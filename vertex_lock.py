bl_info = {
    "name": "Vertex Lock",
    "author": "Szloma",
    "version": (1, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar (N-Panel) > Vertex Lock",
    "description": "Lock vertices, edges, and faces in place to prevent them from moving during proportional editing",
    "warning": "",
    "doc_url": "",
    "category": "Mesh",
}

import bpy
import bmesh
import gpu
import math
import mathutils
from mathutils import Vector
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from bpy.props import (
    BoolProperty,
    FloatVectorProperty,
    IntProperty,
    StringProperty,
    PointerProperty
)

# ------------------------------------------------------------------------
# Properties
# ------------------------------------------------------------------------

class VertexLockSettings(bpy.types.PropertyGroup):
    show_overlay: BoolProperty(
        name="Overview",
        description="Display visual overlay for locked vertices, edges, and faces",
        default=True
    )

    draw_faces: BoolProperty(
        name="Shade Faces",
        description="Shade locked faces in semi-transparent overlay",
        default=True
    )

    draw_edges: BoolProperty(
        name="Outline Edges",
        description="Highlight locked edges with lines",
        default=True
    )

    draw_verts: BoolProperty(
        name="Show Vertices",
        description="Render locked vertices as visible points",
        default=True
    )

    lock_color_point: FloatVectorProperty(
        name="Vertex Color",
        description="Color and opacity for unselected locked vertex points",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 0.15, 0.15, 1.0)
    )

    lock_color_face: FloatVectorProperty(
        name="Face Color",
        description="Color and opacity for unselected locked faces",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 0.05, 0.05, 0.28)
    )

    lock_color_line: FloatVectorProperty(
        name="Edge Color",
        description="Color and opacity for unselected locked edge outlines",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 0.1, 0.1, 0.8)
    )

    point_size: IntProperty(
        name="Point Size",
        description="Pixel diameter of locked vertex points",
        min=2,
        max=24,
        default=8
    )

    line_width: IntProperty(
        name="Line Width",
        description="Pixel width of locked edge outlines",
        min=1,
        max=10,
        default=1
    )

    x_ray: BoolProperty(
        name="In front",
        description="Draw locked geometry through the mesh without depth occlusion",
        default=False
    )


# ------------------------------------------------------------------------
# GPU Draw Handler (Overlay with Selected Highlight)
# ------------------------------------------------------------------------

_draw_handler = None

def is_backface_culling_active(context):
    """Checks if Backface Culling is enabled in Blender's current 3D Viewport shading settings."""
    try:
        space = getattr(context, 'space_data', None)
        if space and space.type == 'VIEW_3D' and hasattr(space, 'shading'):
            return bool(getattr(space.shading, 'show_backface_culling', False))
        if hasattr(context, 'area') and context.area and context.area.type == 'VIEW_3D':
            for sp in context.area.spaces:
                if sp.type == 'VIEW_3D' and hasattr(sp, 'shading'):
                    return bool(getattr(sp.shading, 'show_backface_culling', False))
    except Exception:
        pass
    return False

def tag_redraw_view3d():
    """Forces all 3D viewports to immediately redraw."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def draw_callback_view3d():
    """Renders overlay for locked vertices, edges, and faces in 3D viewport with selection highlights."""
    context = bpy.context
    scene = context.scene
    settings = getattr(scene, 'vertex_lock_settings', None)
    if not settings or not settings.show_overlay:
        return

    obj = context.active_object
    if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
        return

    try:
        bm = bmesh.from_edit_mesh(obj.data)
    except Exception:
        return

    vert_lock_layer = bm.verts.layers.int.get("vertex_lock")
    if not vert_lock_layer:
        return

    locked_verts = [v for v in bm.verts if v[vert_lock_layer] >= 1 or v.hide]
    if not locked_verts:
        return

    locked_vert_set = set(locked_verts)
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    gpu.matrix.push()
    gpu.matrix.multiply_matrix(obj.matrix_world)

    if settings.x_ray:
        gpu.state.depth_test_set('ALWAYS')
    else:
        gpu.state.depth_test_set('LESS_EQUAL')

    # Prevent transparent overlays from contaminating the depth buffer
    gpu.state.depth_mask_set(False)
    gpu.state.blend_set('ALPHA')

    # Physical surface-normal offset to prevent Z-fighting without camera-distance depth distortion
    scale = max(obj.matrix_world.to_scale()) if hasattr(obj.matrix_world, 'to_scale') else 1.0
    bias = (0.0003 / max(0.001, scale)) if (not settings.x_ray) else 0.0
    bias = min(max(bias, 0.0), 0.001)

    try:
        # 1. Faces (Offset along face normal by bias)
        if settings.draw_faces:
            face_lock_layer = bm.faces.layers.int.get("face_lock")
            unsel_face_coords = []
            unsel_face_indices = []
            sel_face_coords = []
            sel_face_indices = []

            for f in bm.faces:
                is_locked_face = (face_lock_layer and f[face_lock_layer] >= 1) or all(v in locked_vert_set for v in f.verts)
                if is_locked_face and len(f.verts) >= 3:
                    is_sel_face = (face_lock_layer and f[face_lock_layer] == 2) or all(v[vert_lock_layer] == 2 for v in f.verts)
                    f_offset = f.normal * bias if f.normal.length_squared > 1e-4 else None
                    if is_sel_face:
                        base = len(sel_face_coords)
                        if f_offset:
                            sel_face_coords.extend((v.co + f_offset) for v in f.verts)
                        else:
                            sel_face_coords.extend(v.co for v in f.verts)
                        for i in range(1, len(f.verts) - 1):
                            sel_face_indices.append((base, base + i, base + i + 1))
                    else:
                        base = len(unsel_face_coords)
                        if f_offset:
                            unsel_face_coords.extend((v.co + f_offset) for v in f.verts)
                        else:
                            unsel_face_coords.extend(v.co for v in f.verts)
                        for i in range(1, len(f.verts) - 1):
                            unsel_face_indices.append((base, base + i, base + i + 1))

            if is_backface_culling_active(context):
                is_neg_scale = (obj.matrix_world.determinant() < 0)
                gpu.state.face_culling_set('FRONT' if is_neg_scale else 'BACK')
            else:
                gpu.state.face_culling_set('NONE')

            if unsel_face_indices:
                batch_f_unsel = batch_for_shader(shader, 'TRIS', {'pos': unsel_face_coords}, indices=unsel_face_indices)
                shader.uniform_float('color', settings.lock_color_face)
                batch_f_unsel.draw(shader)

            if sel_face_indices:
                batch_f_sel = batch_for_shader(shader, 'TRIS', {'pos': sel_face_coords}, indices=sel_face_indices)
                shader.uniform_float('color', (1.0, 0.70, 0.10, 0.38))
                batch_f_sel.draw(shader)

            gpu.state.face_culling_set('NONE')

        # 2. Edges (Offset along vertex normals by bias * 1.3)
        if settings.draw_edges:
            edge_bias = bias * 1.3
            unsel_edge_coords = []
            sel_edge_coords = []
            for e in bm.edges:
                if e.verts[0] in locked_vert_set and e.verts[1] in locked_vert_set:
                    is_sel_edge = (e.verts[0][vert_lock_layer] == 2 and e.verts[1][vert_lock_layer] == 2)
                    v0 = e.verts[0]
                    v1 = e.verts[1]
                    n0 = v0.normal * edge_bias if v0.normal.length_squared > 1e-4 else None
                    n1 = v1.normal * edge_bias if v1.normal.length_squared > 1e-4 else None
                    co0 = (v0.co + n0) if n0 else v0.co
                    co1 = (v1.co + n1) if n1 else v1.co

                    if is_sel_edge:
                        sel_edge_coords.extend((co0, co1))
                    else:
                        unsel_edge_coords.extend((co0, co1))

            if unsel_edge_coords:
                try:
                    gpu.state.line_width_set(float(settings.line_width))
                except Exception:
                    pass
                batch_e_unsel = batch_for_shader(shader, 'LINES', {'pos': unsel_edge_coords})
                shader.uniform_float('color', settings.lock_color_line)
                batch_e_unsel.draw(shader)

            if sel_edge_coords:
                try:
                    gpu.state.line_width_set(float(settings.line_width))
                except Exception:
                    pass
                batch_e_sel = batch_for_shader(shader, 'LINES', {'pos': sel_edge_coords})
                shader.uniform_float('color', (1.0, 0.75, 0.15, 0.95))
                batch_e_sel.draw(shader)

        # 3. Vertices (Offset along vertex normal by bias * 1.6)
        if settings.draw_verts:
            vert_bias = bias * 1.6
            unsel_vert_coords = []
            sel_vert_coords = []

            for v in locked_verts:
                n = v.normal * vert_bias if v.normal.length_squared > 1e-4 else None
                co = (v.co + n) if n else v.co
                if v[vert_lock_layer] == 2:
                    sel_vert_coords.append(co)
                else:
                    unsel_vert_coords.append(co)

            # Unselected locked vertices (red)
            if unsel_vert_coords:
                try:
                    gpu.state.point_size_set(float(settings.point_size))
                except Exception:
                    pass
                batch_v_unsel = batch_for_shader(shader, 'POINTS', {'pos': unsel_vert_coords})
                shader.uniform_float('color', settings.lock_color_point)
                batch_v_unsel.draw(shader)

            # Selected locked vertices (clean white square)
            if sel_vert_coords:
                try:
                    gpu.state.point_size_set(float(settings.point_size))
                except Exception:
                    pass
                batch_v_sel = batch_for_shader(shader, 'POINTS', {'pos': sel_vert_coords})
                shader.uniform_float('color', (1.0, 1.0, 1.0, 1.0))
                batch_v_sel.draw(shader)

    except Exception:
        pass
    finally:
        gpu.state.depth_mask_set(True)
        gpu.state.face_culling_set('NONE')
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')
        try:
            gpu.state.point_size_set(1.0)
            gpu.state.line_width_set(1.0)
        except Exception:
            pass
        gpu.matrix.pop()


# ------------------------------------------------------------------------
# Operators
# ------------------------------------------------------------------------

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
    Projects 3D mesh elements to 2D viewport coordinates.
    Returns ('VERT'|'EDGE'|'FACE', element) if a locked element is clicked.
    If an unlocked element is closer to the cursor, returns (None, None).
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

        # 1. Unlock elements marked selected via our tool (layer == 2)
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

        # 2. Check native Blender selection for locked faces or edges
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
                v.hide = True           # Keep hidden for proportional editing immunity
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


# ------------------------------------------------------------------------
# UI Panel & Preferences
# ------------------------------------------------------------------------

def get_active_shortcut(idname):
    """Dynamically resolves the active user-assigned shortcut string for an operator."""
    wm = bpy.context.window_manager
    for kc in (wm.keyconfigs.user, wm.keyconfigs.addon):
        if not kc:
            continue
        km = kc.keymaps.get("Mesh")
        if not km:
            continue
        for kmi in km.keymap_items:
            if kmi.idname == idname:
                if not kmi.active:
                    return "Disabled"
                parts = []
                if kmi.ctrl:
                    parts.append("Ctrl")
                if kmi.alt:
                    parts.append("Alt")
                if kmi.shift:
                    parts.append("Shift")
                if kmi.oskey:
                    parts.append("Cmd")
                parts.append(kmi.type)
                return " + ".join(parts)
    return "None"


class VertexLockAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    category: StringProperty(
        name="Tab Category",
        description="Name of the 3D Viewport sidebar (N-Panel) tab",
        default="Vertex Lock"
    )

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="UI Configuration", icon='PREFERENCES')
        box.prop(self, "category", text="Sidebar Tab Name")

        box = layout.box()
        box.label(text="Keymap Shortcuts", icon='KEYINGSET')
        wm = context.window_manager
        kc = wm.keyconfigs.user
        if kc:
            km = kc.keymaps.get("Mesh")
            if km:
                for kmi in km.keymap_items:
                    if kmi.idname in {
                        "mesh.vertex_lock_selected",
                        "mesh.vertex_unlock_selected",
                        "mesh.vertex_lock_toggle",
                    }:
                        row = box.row(align=True)
                        row.label(text=kmi.name)
                        row.prop(kmi, "type", text="", full_event=True)
                        row.prop(kmi, "active", text="")


class MESH_PT_vertex_lock_panel(bpy.types.Panel):
    """Vertex Lock sidebar panel in 3D Viewport"""
    bl_label = "Vertex Lock"
    bl_idname = "MESH_PT_vertex_lock_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Vertex Lock"

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        scene = context.scene
        settings = getattr(scene, 'vertex_lock_settings', None)

        is_edit_mode = (obj.mode == 'EDIT')

        # Actions & Status
        box = layout.box()
        col = box.column(align=True)

        if is_edit_mode:
            total_verts = len(obj.data.vertices)
            locked_count = 0
            sel_locked_count = 0
            try:
                bm = bmesh.from_edit_mesh(obj.data)
                layer = bm.verts.layers.int.get("vertex_lock")
                if layer:
                    locked_count = sum(1 for v in bm.verts if v[layer] >= 1 or v.hide)
                    sel_locked_count = sum(1 for v in bm.verts if v[layer] == 2)
                else:
                    locked_count = sum(1 for v in bm.verts if v.hide)
                total_verts = len(bm.verts)
            except Exception:
                pass

            row = col.row(align=True)
            pct = (locked_count / total_verts * 100) if total_verts > 0 else 0
            if sel_locked_count > 0:
                row.label(text=f"Status: {locked_count} / {total_verts} ({sel_locked_count} selected)", icon='INFO')
            else:
                row.label(text=f"Status: {locked_count} / {total_verts} ({pct:.0f}%)", icon='INFO')

            col.separator()

            # Actions
            col_actions = col.column(align=True)
            col_actions.scale_y = 1.3
            col_actions.operator("mesh.vertex_lock_selected", text="Lock Selected", icon='LOCKED')

            row = col_actions.row(align=True)
            row.operator("mesh.vertex_unlock_selected", text="Unlock Selected", icon='UNLOCKED')
            row.operator("mesh.vertex_lock_toggle", text="Toggle Lock", icon='ARROW_LEFTRIGHT')

            col.separator()

            # Management & selection utilities
            row = col.row(align=True)
            row.operator("mesh.vertex_select_locked", text="Select Locked", icon='RESTRICT_SELECT_OFF')
            row.operator("mesh.vertex_lock_deselect_all", text="Deselect Locked", icon='RESTRICT_SELECT_ON')

            row = col.row(align=True)
            row.operator("mesh.vertex_unlock_all", text="Unlock All", icon='X')
            row.operator("mesh.vertex_invert_locks", text="Invert Locks", icon='MOD_MIRROR')

            row = col.row(align=True)
            row.operator("mesh.vertex_relock", text="Re-apply Locks", icon='FILE_REFRESH')
        else:
            col.label(text="Enter Edit Mode to Lock/Unlock", icon='INFO')

        # Render / Overview Display Settings
        if settings:
            box = layout.box()
            box.label(text="Render", icon='RESTRICT_RENDER_OFF')
            header_row = box.row(align=True)
            header_row.prop(settings, "show_overlay", text="Overview", icon='HIDE_OFF' if settings.show_overlay else 'HIDE_ON')

            if settings.show_overlay:
                col = box.column(align=True)
                col.prop(settings, "x_ray", text="In front", icon='SHADING_WIRE')
                row = col.row(align=True)
                row.prop(settings, "draw_verts", text="Vertices", toggle=True)
                row.prop(settings, "draw_edges", text="Edges", toggle=True)
                row.prop(settings, "draw_faces", text="Faces", toggle=True)

        # Shortcuts Live Reference
        box = layout.box()
        box.label(text="Shortcuts", icon='KEYINGSET')
        col = box.column(align=True)
        row = col.row()
        row.label(text="Lock Selected:")
        row.label(text=get_active_shortcut("mesh.vertex_lock_selected"))
        row = col.row()
        row.label(text="Unlock Selected:")
        row.label(text=get_active_shortcut("mesh.vertex_unlock_selected"))
        row = col.row()
        row.label(text="Toggle Lock:")
        row.label(text=get_active_shortcut("mesh.vertex_lock_toggle"))


# ------------------------------------------------------------------------
# Keymaps & Registration
# ------------------------------------------------------------------------

addon_keymaps = []

def register_keymaps():
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if not kc:
        return

    km = kc.keymaps.new(name="Mesh", space_type='EMPTY')

    kmi_lock = km.keymap_items.new(
        "mesh.vertex_lock_selected",
        type='L',
        value='PRESS',
        shift=True
    )
    addon_keymaps.append((km, kmi_lock))

    kmi_unlock = km.keymap_items.new(
        "mesh.vertex_unlock_selected",
        type='L',
        value='PRESS',
        alt=True,
        shift=True
    )
    addon_keymaps.append((km, kmi_unlock))

    kmi_toggle = km.keymap_items.new(
        "mesh.vertex_lock_toggle",
        type='L',
        value='PRESS',
        ctrl=True,
        shift=True
    )
    addon_keymaps.append((km, kmi_toggle))

    # Click Select Locked Elements: Left Mouse Click
    kmi_click = km.keymap_items.new(
        "mesh.vertex_lock_click_select",
        type='LEFTMOUSE',
        value='CLICK'
    )
    kmi_click.properties.extend = False
    addon_keymaps.append((km, kmi_click))

    # Shift-Click Multi-Select: Shift + Left Mouse Click
    kmi_shift_click = km.keymap_items.new(
        "mesh.vertex_lock_click_select",
        type='LEFTMOUSE',
        value='CLICK',
        shift=True
    )
    kmi_shift_click.properties.extend = True
    addon_keymaps.append((km, kmi_shift_click))

    try:
        if bpy.context.preferences.inputs.mouse_select_button == 'RIGHT':
            kmi_rc = km.keymap_items.new("mesh.vertex_lock_click_select", type='RIGHTMOUSE', value='CLICK')
            kmi_rc.properties.extend = False
            addon_keymaps.append((km, kmi_rc))

            kmi_rc_shift = km.keymap_items.new("mesh.vertex_lock_click_select", type='RIGHTMOUSE', value='CLICK', shift=True)
            kmi_rc_shift.properties.extend = True
            addon_keymaps.append((km, kmi_rc_shift))
    except Exception:
        pass


def unregister_keymaps():
    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    addon_keymaps.clear()


classes = (
    VertexLockSettings,
    MESH_OT_vertex_lock_click_select,
    MESH_OT_vertex_lock_selected,
    MESH_OT_vertex_unlock_selected,
    MESH_OT_vertex_unlock_all,
    MESH_OT_vertex_lock_toggle,
    MESH_OT_vertex_select_locked,
    MESH_OT_vertex_lock_deselect_all,
    MESH_OT_vertex_relock,
    MESH_OT_vertex_invert_locks,
    VertexLockAddonPreferences,
    MESH_PT_vertex_lock_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.vertex_lock_settings = PointerProperty(type=VertexLockSettings)

    global _draw_handler
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback_view3d, (), 'WINDOW', 'POST_VIEW'
        )

    register_keymaps()


def unregister():
    unregister_keymaps()

    global _draw_handler
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        _draw_handler = None

    tag_redraw_view3d()

    if hasattr(bpy.types.Scene, 'vertex_lock_settings'):
        del bpy.types.Scene.vertex_lock_settings

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


if __name__ == "__main__":
    register()
