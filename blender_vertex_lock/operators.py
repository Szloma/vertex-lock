import bpy
import bmesh
import math
import mathutils
from mathutils import Vector
from bpy_extras import view3d_utils
from bpy.props import BoolProperty, IntProperty, StringProperty
from .draw import tag_redraw_view3d

def get_edit_mesh_and_layers(context):
    """Utility to retrieve active bmesh and vertex/face lock custom layers."""
    obj = context.active_object
    if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
        return None, None, None, None

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    vert_lock_layer = bm.verts.layers.int.get("vertex_lock")
    if not vert_lock_layer:
        vert_lock_layer = bm.verts.layers.int.new("vertex_lock")

    face_lock_layer = bm.faces.layers.int.get("face_lock")
    if not face_lock_layer:
        face_lock_layer = bm.faces.layers.int.new("face_lock")

    return obj, bm, vert_lock_layer, face_lock_layer


def update_mesh_lock_visibility(bm, vert_lock_layer, face_lock_layer):
    """
    Enforces hide / select state consistency for locked and unlocked elements:
    - Locked vertices: hide=True, select=False (preserves Proportional Editing immunity)
    - Unlocked vertices: hide=False
    - Faces: keep hide=False so surrounding mesh and unlocked faces remain visible
    - Locked edges (both endpoints locked): hide=True, select=False (protects from accidental deletion when editing adjacent faces)
    - Unlocked edges (at least one endpoint unlocked): hide=False
    """
    for v in bm.verts:
        if v[vert_lock_layer] >= 1:
            v.hide = True
            v.select = False
        else:
            v.hide = False

    for f in bm.faces:
        f.hide = False

    for e in bm.edges:
        is_locked_edge = (e.verts[0][vert_lock_layer] >= 1 and e.verts[1][vert_lock_layer] >= 1)
        if is_locked_edge:
            e.hide = True
            e.select = False
        else:
            e.hide = False


def find_locked_element_under_mouse(context, event, obj, bm, vert_lock_layer, face_lock_layer):
    """
    Raycasts / projects 3D mesh elements to 2D viewport coordinates.
    Returns ('VERT'|'EDGE'|'FACE', element) if a locked element is under the cursor.
    If an unlocked element is closer to the cursor, returns (None, None) to let Blender handle it.
    """
    region = context.region
    rv3d = getattr(context.space_data, 'region_3d', None) if (context.space_data and context.space_data.type == 'VIEW_3D') else getattr(context, 'region_data', None)
    if not region or not rv3d:
        return None, None

    mouse_x = event.mouse_region_x
    mouse_y = event.mouse_region_y
    mouse_vec = Vector((mouse_x, mouse_y, 0.0))

    select_mode = context.tool_settings.mesh_select_mode
    is_vert_mode, is_edge_mode, is_face_mode = select_mode

    mat = obj.matrix_world
    view_mat = rv3d.view_matrix

    # 1. Vertices
    if is_vert_mode:
        best_v = None
        min_dist_sq = 196.0  # 14px tolerance squared
        best_depth = -float('inf')
        closest_unlocked_dist_sq = float('inf')

        for v in bm.verts:
            world_co = mat @ v.co
            screen_co = view3d_utils.location_3d_to_region_2d(region, rv3d, world_co)
            if screen_co is None:
                continue

            dx = screen_co.x - mouse_x
            dy = screen_co.y - mouse_y
            dist_sq = dx * dx + dy * dy

            is_locked = (v[vert_lock_layer] in {1, 2} or v.hide)
            if not is_locked:
                if dist_sq < closest_unlocked_dist_sq:
                    closest_unlocked_dist_sq = dist_sq
            else:
                if dist_sq <= 196.0:
                    depth = (view_mat @ world_co).z
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        best_depth = depth
                        best_v = v
                    elif abs(dist_sq - min_dist_sq) < 9.0 and depth > best_depth:
                        best_depth = depth
                        best_v = v

        if best_v and closest_unlocked_dist_sq >= min_dist_sq:
            return 'VERT', best_v

    # 2. Edges
    elif is_edge_mode:
        best_e = None
        min_dist = 12.0  # 12px tolerance
        best_depth = -float('inf')

        for e in bm.edges:
            is_locked_e = (e.verts[0][vert_lock_layer] in {1, 2} or e.verts[0].hide) and \
                          (e.verts[1][vert_lock_layer] in {1, 2} or e.verts[1].hide)
            if not is_locked_e:
                continue

            p0 = view3d_utils.location_3d_to_region_2d(region, rv3d, mat @ e.verts[0].co)
            p1 = view3d_utils.location_3d_to_region_2d(region, rv3d, mat @ e.verts[1].co)
            if p0 is None or p1 is None:
                continue

            p0_3d = Vector((p0.x, p0.y, 0.0))
            p1_3d = Vector((p1.x, p1.y, 0.0))
            proj, factor = mathutils.geometry.intersect_point_line(mouse_vec, p0_3d, p1_3d)
            if 0.0 <= factor <= 1.0:
                dist = (mouse_vec - proj).length
            elif factor < 0.0:
                dist = (mouse_vec - p0_3d).length
            else:
                dist = (mouse_vec - p1_3d).length

            if dist <= min_dist:
                mid_world = mat @ ((e.verts[0].co + e.verts[1].co) * 0.5)
                depth = (view_mat @ mid_world).z
                if dist < min_dist or depth > best_depth:
                    min_dist = dist
                    best_depth = depth
                    best_e = e

        if best_e:
            return 'EDGE', best_e

    # 3. Faces
    elif is_face_mode:
        best_f = None
        best_depth = -float('inf')

        for f in bm.faces:
            is_locked_f = (face_lock_layer and f[face_lock_layer] in {1, 2}) or \
                          all(v[vert_lock_layer] in {1, 2} or v.hide for v in f.verts)
            if not is_locked_f:
                continue

            pts_2d = [view3d_utils.location_3d_to_region_2d(region, rv3d, mat @ v.co) for v in f.verts]
            if any(p is None for p in pts_2d):
                continue

            inside = False
            for i in range(1, len(pts_2d) - 1):
                if mathutils.geometry.intersect_point_tri_2d(
                    Vector((mouse_x, mouse_y)), pts_2d[0], pts_2d[i], pts_2d[i + 1]
                ):
                    inside = True
                    break

            if inside:
                center_world = mat @ f.calc_center_median()
                depth = (view_mat @ center_world).z
                if depth > best_depth:
                    best_depth = depth
                    best_f = f

        if best_f:
            return 'FACE', best_f

    return None, None


class MESH_OT_vertex_lock_click_select(bpy.types.Operator):
    """Select or toggle locked vertices, edges, or faces by clicking on them in the 3D Viewport"""
    bl_idname = "mesh.vertex_lock_click_select"
    bl_label = "Select Locked Element"
    bl_options = {'REGISTER', 'UNDO'}

    extend: BoolProperty(
        name="Extend",
        description="Extend selection instead of replacing",
        default=False
    )
    hit_type: StringProperty(
        name="Hit Type",
        description="Type of hit element (VERT, EDGE, FACE)",
        default=""
    )
    hit_index: IntProperty(
        name="Hit Index",
        description="Index of hit element for programmatic selection",
        default=-1
    )

    def _apply_selection(self, elem_type, elem, bm, vert_lock_layer, face_lock_layer):
        if elem_type == 'VERT':
            if self.extend:
                elem[vert_lock_layer] = 1 if elem[vert_lock_layer] == 2 else 2
            else:
                for v in bm.verts:
                    if v[vert_lock_layer] == 2:
                        v[vert_lock_layer] = 1
                    v.select = False
                for f in bm.faces:
                    if f[face_lock_layer] == 2:
                        f[face_lock_layer] = 1
                    f.select = False
                for e in bm.edges:
                    e.select = False
                elem[vert_lock_layer] = 2

        elif elem_type == 'EDGE':
            is_selected = (elem.verts[0][vert_lock_layer] == 2 and elem.verts[1][vert_lock_layer] == 2)
            target_val = 1 if (self.extend and is_selected) else 2
            if not self.extend:
                for v in bm.verts:
                    if v[vert_lock_layer] == 2:
                        v[vert_lock_layer] = 1
                    v.select = False
                for f in bm.faces:
                    if f[face_lock_layer] == 2:
                        f[face_lock_layer] = 1
                    f.select = False
                for e in bm.edges:
                    e.select = False
            elem.verts[0][vert_lock_layer] = target_val
            elem.verts[1][vert_lock_layer] = target_val

        elif elem_type == 'FACE':
            is_selected = (elem[face_lock_layer] == 2) or all(v[vert_lock_layer] == 2 for v in elem.verts)
            target_val = 1 if (self.extend and is_selected) else 2
            if not self.extend:
                for v in bm.verts:
                    if v[vert_lock_layer] == 2:
                        v[vert_lock_layer] = 1
                    v.select = False
                for f in bm.faces:
                    if f[face_lock_layer] == 2:
                        f[face_lock_layer] = 1
                    f.select = False
                for e in bm.edges:
                    e.select = False
            elem[face_lock_layer] = target_val
            for v in elem.verts:
                v[vert_lock_layer] = target_val

    def execute(self, context):
        obj, bm, vert_lock_layer, face_lock_layer = get_edit_mesh_and_layers(context)
        if not bm:
            return {'CANCELLED'}

        if self.hit_index >= 0 and self.hit_type:
            elem = None
            if self.hit_type == 'VERT' and 0 <= self.hit_index < len(bm.verts):
                elem = bm.verts[self.hit_index]
            elif self.hit_type == 'EDGE' and 0 <= self.hit_index < len(bm.edges):
                elem = bm.edges[self.hit_index]
            elif self.hit_type == 'FACE' and 0 <= self.hit_index < len(bm.faces):
                elem = bm.faces[self.hit_index]

            if elem:
                self._apply_selection(self.hit_type, elem, bm, vert_lock_layer, face_lock_layer)
                bmesh.update_edit_mesh(obj.data)
                tag_redraw_view3d()
                return {'FINISHED'}

        return {'FINISHED'}

    def invoke(self, context, event):
        if not (context.space_data and context.space_data.type == 'VIEW_3D'):
            return {'PASS_THROUGH'}
        obj, bm, vert_lock_layer, face_lock_layer = get_edit_mesh_and_layers(context)
        if not bm:
            return {'PASS_THROUGH'}

        elem_type, elem = find_locked_element_under_mouse(context, event, obj, bm, vert_lock_layer, face_lock_layer)

        if not elem:
            # Regular click on empty space or unlocked geometry:
            # Deselect any currently selected locked vertices so selection stays synchronized
            if not self.extend:
                has_sel_locked = any(v[vert_lock_layer] == 2 for v in bm.verts)
                if has_sel_locked:
                    for v in bm.verts:
                        if v[vert_lock_layer] == 2:
                            v[vert_lock_layer] = 1
                    for f in bm.faces:
                        if f[face_lock_layer] == 2:
                            f[face_lock_layer] = 1
                    bmesh.update_edit_mesh(obj.data)
                    tag_redraw_view3d()
            return {'PASS_THROUGH'}

        self._apply_selection(elem_type, elem, bm, vert_lock_layer, face_lock_layer)
        bmesh.update_edit_mesh(obj.data)
        tag_redraw_view3d()
        return {'FINISHED'}


class MESH_OT_vertex_lock_selected(bpy.types.Operator):
    """Lock selected vertices, edges, or faces in place to prevent proportional editing movement"""
    bl_idname = "mesh.vertex_lock_selected"
    bl_label = "Lock Selected"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH' and context.active_object.mode == 'EDIT'

    def execute(self, context):
        obj, bm, vert_lock_layer, face_lock_layer = get_edit_mesh_and_layers(context)
        if not bm:
            return {'CANCELLED'}

        select_mode = context.tool_settings.mesh_select_mode
        is_vert_mode, is_edge_mode, is_face_mode = select_mode

        locked_verts = set()
        locked_faces = 0
        locked_edges = 0

        if is_face_mode:
            selected_faces = [f for f in bm.faces if f.select]
            if selected_faces:
                for f in selected_faces:
                    f[face_lock_layer] = 1
                    f.select = False
                    for v in f.verts:
                        v[vert_lock_layer] = 1
                        v.hide = True
                        v.select = False
                        locked_verts.add(v)
                locked_faces = len(selected_faces)

        if not locked_verts and is_edge_mode:
            selected_edges = [e for e in bm.edges if e.select]
            if selected_edges:
                for e in selected_edges:
                    e.select = False
                    for v in e.verts:
                        v[vert_lock_layer] = 1
                        v.hide = True
                        v.select = False
                        locked_verts.add(v)
                locked_edges = len(selected_edges)

        if not locked_verts:
            selected_verts = [v for v in bm.verts if v.select]
            for v in selected_verts:
                v[vert_lock_layer] = 1
                v.hide = True
                v.select = False
                locked_verts.add(v)

        if not locked_verts:
            self.report({'WARNING'}, "No elements selected to lock")
            return {'CANCELLED'}

        # Enforce hide/select state: locked edges must be hidden to protect from adjacent face deletion
        update_mesh_lock_visibility(bm, vert_lock_layer, face_lock_layer)

        bmesh.update_edit_mesh(obj.data)
        tag_redraw_view3d()

        if locked_faces:
            self.report({'INFO'}, f"Locked {locked_faces} faces ({len(locked_verts)} vertices)")
        elif locked_edges:
            self.report({'INFO'}, f"Locked {locked_edges} edges ({len(locked_verts)} vertices)")
        else:
            self.report({'INFO'}, f"Locked {len(locked_verts)} vertices")

        return {'FINISHED'}


class MESH_OT_vertex_unlock_selected(bpy.types.Operator):
    """Unlock selected vertices, edges, or faces"""
    bl_idname = "mesh.vertex_unlock_selected"
    bl_label = "Unlock Selected"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH' and context.active_object.mode == 'EDIT'

    def execute(self, context):
        obj, bm, vert_lock_layer, face_lock_layer = get_edit_mesh_and_layers(context)
        if not bm:
            return {'CANCELLED'}

        select_mode = context.tool_settings.mesh_select_mode
        is_vert_mode, is_edge_mode, is_face_mode = select_mode

        unlocked_verts = set()

        # 1. Unlock locked elements selected in addon (layer == 2)
        selected_locked_verts = [v for v in bm.verts if v[vert_lock_layer] == 2]
        if selected_locked_verts:
            for v in selected_locked_verts:
                v[vert_lock_layer] = 0
                v.hide = False
                v.select = True
                unlocked_verts.add(v)
            for f in bm.faces:
                if f[face_lock_layer] == 2:
                    f[face_lock_layer] = 0
                    for v in f.verts:
                        v[vert_lock_layer] = 0
                        v.hide = False
                        v.select = True
                        unlocked_verts.add(v)

        # 2. Check if native Blender selection has locked faces or edges
        if not unlocked_verts and is_face_mode:
            for f in bm.faces:
                if f.select:
                    f[face_lock_layer] = 0
                    for v in f.verts:
                        if v[vert_lock_layer] >= 1 or v.hide:
                            v[vert_lock_layer] = 0
                            v.hide = False
                            v.select = True
                            unlocked_verts.add(v)

        if not unlocked_verts and is_edge_mode:
            for e in bm.edges:
                if e.select:
                    for v in e.verts:
                        if v[vert_lock_layer] >= 1 or v.hide:
                            v[vert_lock_layer] = 0
                            v.hide = False
                            v.select = True
                            unlocked_verts.add(v)

        if not unlocked_verts:
            for v in bm.verts:
                if v.select and (v[vert_lock_layer] >= 1 or v.hide):
                    v[vert_lock_layer] = 0
                    v.hide = False
                    v.select = True
                    unlocked_verts.add(v)

        if not unlocked_verts:
            self.report({'INFO'}, "No locked elements selected. Click on a locked element or use 'Select Locked'.")
            return {'CANCELLED'}

        update_mesh_lock_visibility(bm, vert_lock_layer, face_lock_layer)

        bmesh.update_edit_mesh(obj.data)
        tag_redraw_view3d()
        self.report({'INFO'}, f"Unlocked {len(unlocked_verts)} vertices")
        return {'FINISHED'}


class MESH_OT_vertex_unlock_all(bpy.types.Operator):
    """Unlock all vertices on the active mesh"""
    bl_idname = "mesh.vertex_unlock_all"
    bl_label = "Unlock All"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH' and context.active_object.mode == 'EDIT'

    def execute(self, context):
        obj, bm, vert_lock_layer, face_lock_layer = get_edit_mesh_and_layers(context)
        if not bm:
            return {'CANCELLED'}

        unlocked_count = 0
        for v in bm.verts:
            if v[vert_lock_layer] >= 1 or v.hide:
                v[vert_lock_layer] = 0
                v.hide = False
                unlocked_count += 1

        for f in bm.faces:
            f[face_lock_layer] = 0
            f.hide = False

        for e in bm.edges:
            e.hide = False

        bmesh.update_edit_mesh(obj.data)
        tag_redraw_view3d()
        self.report({'INFO'}, f"Unlocked all {unlocked_count} vertices")
        return {'FINISHED'}


class MESH_OT_vertex_lock_toggle(bpy.types.Operator):
    """Toggle lock state for selected vertices, edges, or faces"""
    bl_idname = "mesh.vertex_lock_toggle"
    bl_label = "Toggle Lock"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH' and context.active_object.mode == 'EDIT'

    def execute(self, context):
        obj, bm, vert_lock_layer, face_lock_layer = get_edit_mesh_and_layers(context)
        if not bm:
            return {'CANCELLED'}

        has_locked_selection = any(v[vert_lock_layer] == 2 for v in bm.verts) or \
                               any(f[face_lock_layer] == 2 for f in bm.faces)

        if not has_locked_selection:
            for v in bm.verts:
                if v.select and (v[vert_lock_layer] >= 1 or v.hide):
                    has_locked_selection = True
                    break

        if has_locked_selection:
            return bpy.ops.mesh.vertex_unlock_selected()
        else:
            return bpy.ops.mesh.vertex_lock_selected()


class MESH_OT_vertex_select_locked(bpy.types.Operator):
    """Select all currently locked vertices without breaking their lock protection"""
    bl_idname = "mesh.vertex_select_locked"
    bl_label = "Select Locked"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH' and context.active_object.mode == 'EDIT'

    def execute(self, context):
        obj, bm, vert_lock_layer, face_lock_layer = get_edit_mesh_and_layers(context)
        if not bm:
            return {'CANCELLED'}

        count = 0
        for v in bm.verts:
            if v[vert_lock_layer] >= 1 or v.hide:
                v[vert_lock_layer] = 2  # Mark selected in lock manager
                v.hide = True           # Keep hidden from transform engine for PET immunity
                v.select = False        # Avoid partial mesh crash in Blender 5.2
                count += 1

        for f in bm.faces:
            if f[face_lock_layer] >= 1 or all(v[vert_lock_layer] >= 1 for v in f.verts):
                f[face_lock_layer] = 2

        update_mesh_lock_visibility(bm, vert_lock_layer, face_lock_layer)
        bmesh.update_edit_mesh(obj.data)
        tag_redraw_view3d()
        self.report({'INFO'}, f"Selected {count} locked vertices")
        return {'FINISHED'}


class MESH_OT_vertex_lock_deselect_all(bpy.types.Operator):
    """Deselect all locked vertices"""
    bl_idname = "mesh.vertex_lock_deselect_all"
    bl_label = "Deselect Locked"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH' and context.active_object.mode == 'EDIT'

    def execute(self, context):
        obj, bm, vert_lock_layer, face_lock_layer = get_edit_mesh_and_layers(context)
        if not bm:
            return {'CANCELLED'}

        deselected = 0
        for v in bm.verts:
            if v[vert_lock_layer] == 2:
                v[vert_lock_layer] = 1
                deselected += 1

        for f in bm.faces:
            if f[face_lock_layer] == 2:
                f[face_lock_layer] = 1

        update_mesh_lock_visibility(bm, vert_lock_layer, face_lock_layer)
        bmesh.update_edit_mesh(obj.data)
        tag_redraw_view3d()
        self.report({'INFO'}, f"Deselected {deselected} locked vertices")
        return {'FINISHED'}


class MESH_OT_vertex_relock(bpy.types.Operator):
    """Re-enforce lock state on all marked locked vertices (useful after pressing Alt+H)"""
    bl_idname = "mesh.vertex_relock"
    bl_label = "Re-apply Locks"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH' and context.active_object.mode == 'EDIT'

    def execute(self, context):
        obj, bm, vert_lock_layer, face_lock_layer = get_edit_mesh_and_layers(context)
        if not bm:
            return {'CANCELLED'}

        relocked = sum(1 for v in bm.verts if v[vert_lock_layer] >= 1)
        update_mesh_lock_visibility(bm, vert_lock_layer, face_lock_layer)

        bmesh.update_edit_mesh(obj.data)
        tag_redraw_view3d()
        self.report({'INFO'}, f"Re-applied locks to {relocked} vertices")
        return {'FINISHED'}


class MESH_OT_vertex_invert_locks(bpy.types.Operator):
    """Invert locked and unlocked vertices across the mesh"""
    bl_idname = "mesh.vertex_invert_locks"
    bl_label = "Invert Locks"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH' and context.active_object.mode == 'EDIT'

    def execute(self, context):
        obj, bm, vert_lock_layer, face_lock_layer = get_edit_mesh_and_layers(context)
        if not bm:
            return {'CANCELLED'}

        for v in bm.verts:
            is_locked = (v[vert_lock_layer] >= 1)
            v[vert_lock_layer] = 0 if is_locked else 1

        update_mesh_lock_visibility(bm, vert_lock_layer, face_lock_layer)

        bmesh.update_edit_mesh(obj.data)
        tag_redraw_view3d()
        self.report({'INFO'}, "Inverted vertex locks")
        return {'FINISHED'}


classes = (
    MESH_OT_vertex_lock_click_select,
    MESH_OT_vertex_lock_selected,
    MESH_OT_vertex_unlock_selected,
    MESH_OT_vertex_unlock_all,
    MESH_OT_vertex_lock_toggle,
    MESH_OT_vertex_select_locked,
    MESH_OT_vertex_lock_deselect_all,
    MESH_OT_vertex_relock,
    MESH_OT_vertex_invert_locks,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
