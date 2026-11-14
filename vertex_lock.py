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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
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

def _update_overlay_geom(self, context):
    try:
        tag_redraw_view3d(invalidate_cache=True)
    except Exception:
        pass

def _update_overlay_uniform(self, context):
    try:
        tag_redraw_view3d(invalidate_cache=False)
    except Exception:
        pass

class VertexLockSettings(bpy.types.PropertyGroup):
    show_overlay: BoolProperty(
        name="Overview",
        description="Display visual overlay for locked vertices, edges, and faces",
        default=True,
        update=_update_overlay_geom
    )

    draw_faces: BoolProperty(
        name="Shade Faces",
        description="Shade locked faces in semi-transparent overlay",
        default=True,
        update=_update_overlay_geom
    )

    draw_edges: BoolProperty(
        name="Outline Edges",
        description="Highlight locked edges with lines",
        default=True,
        update=_update_overlay_geom
    )

    draw_verts: BoolProperty(
        name="Show Vertices",
        description="Render locked vertices as visible points",
        default=True,
        update=_update_overlay_geom
    )

    lock_color_point: FloatVectorProperty(
        name="Vertex Color",
        description="Color and opacity for unselected locked vertex points",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 0.15, 0.15, 1.0),
        update=_update_overlay_uniform
    )

    lock_color_face: FloatVectorProperty(
        name="Face Color",
        description="Color and opacity for unselected locked faces",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 0.05, 0.05, 0.28),
        update=_update_overlay_uniform
    )

    lock_color_line: FloatVectorProperty(
        name="Edge Color",
        description="Color and opacity for unselected locked edge outlines",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 0.1, 0.1, 0.8),
        update=_update_overlay_uniform
    )

    point_size: IntProperty(
        name="Point Size",
        description="Pixel diameter of locked vertex points",
        min=2,
        max=24,
        default=8,
        update=_update_overlay_uniform
    )

    line_width: IntProperty(
        name="Line Width",
        description="Pixel width of locked edge outlines",
        min=1,
        max=10,
        default=1,
        update=_update_overlay_uniform
    )

    x_ray: BoolProperty(
        name="In front",
        description="Draw locked geometry through the mesh without depth occlusion",
        default=False,
        update=_update_overlay_geom
    )


# ------------------------------------------------------------------------
# GPU Draw Handler (Overlay with Batch Caching & Selection Highlight)
# ------------------------------------------------------------------------

_draw_handler = None


class OverlayBatchCache:
    """
    Caches GPU batches for locked faces, edges, and vertices (Single Responsibility).
    Prevents redundant per-frame BMesh traversals and GPU VBO allocations during viewport navigation.
    """
    def __init__(self):
        self.cached_obj_name = None
        self.cached_mesh_name = None
        self.cached_bias = 0.0
        self.cached_draw_faces = True
        self.cached_draw_edges = True
        self.cached_draw_verts = True
        self.is_dirty = True

        self.batch_f_unsel = None
        self.batch_f_sel = None
        self.batch_e_unsel = None
        self.batch_e_sel = None
        self.batch_v_unsel = None
        self.batch_v_sel = None

        self.locked_count = 0
        self.sel_locked_count = 0
        self.total_verts = 0
        self.has_locked = False

    def invalidate(self):
        self.is_dirty = True

    def clear(self):
        self.batch_f_unsel = None
        self.batch_f_sel = None
        self.batch_e_unsel = None
        self.batch_e_sel = None
        self.batch_v_unsel = None
        self.batch_v_sel = None
        self.has_locked = False
        self.locked_count = 0
        self.sel_locked_count = 0
        self.total_verts = 0

    def update_cache(self, obj, bm, vert_lock_layer, face_lock_layer, settings, bias, shader):
        locked_verts = [v for v in bm.verts if v[vert_lock_layer] >= 1 or v.hide]
        self.total_verts = len(bm.verts)
        self.locked_count = len(locked_verts)
        self.sel_locked_count = sum(1 for v in locked_verts if v[vert_lock_layer] == 2)

        if not locked_verts:
            self.clear()
            self.cached_obj_name = obj.name
            self.cached_mesh_name = obj.data.name
            self.cached_bias = bias
            self.cached_draw_faces = settings.draw_faces
            self.cached_draw_edges = settings.draw_edges
            self.cached_draw_verts = settings.draw_verts
            self.is_dirty = False
            return

        self.has_locked = True
        locked_vert_set = set(locked_verts)

        # 1. Faces (Selective extraction from candidate faces connected to locked vertices)
        self.batch_f_unsel = None
        self.batch_f_sel = None
        if settings.draw_faces:
            candidate_faces = {f for v in locked_verts for f in v.link_faces}
            unsel_face_coords = []
            unsel_face_indices = []
            sel_face_coords = []
            sel_face_indices = []

            for f in candidate_faces:
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

            if unsel_face_indices:
                self.batch_f_unsel = batch_for_shader(shader, 'TRIS', {'pos': unsel_face_coords}, indices=unsel_face_indices)
            if sel_face_indices:
                self.batch_f_sel = batch_for_shader(shader, 'TRIS', {'pos': sel_face_coords}, indices=sel_face_indices)

        # 2. Edges (Selective extraction from candidate edges connected to locked vertices)
        self.batch_e_unsel = None
        self.batch_e_sel = None
        if settings.draw_edges:
            candidate_edges = {e for v in locked_verts for e in v.link_edges}
            edge_bias = bias * 1.3
            unsel_edge_coords = []
            sel_edge_coords = []
            for e in candidate_edges:
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
                self.batch_e_unsel = batch_for_shader(shader, 'LINES', {'pos': unsel_edge_coords})
            if sel_edge_coords:
                self.batch_e_sel = batch_for_shader(shader, 'LINES', {'pos': sel_edge_coords})

        # 3. Vertices
        self.batch_v_unsel = None
        self.batch_v_sel = None
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

            if unsel_vert_coords:
                self.batch_v_unsel = batch_for_shader(shader, 'POINTS', {'pos': unsel_vert_coords})
            if sel_vert_coords:
                self.batch_v_sel = batch_for_shader(shader, 'POINTS', {'pos': sel_vert_coords})

        self.cached_obj_name = obj.name
        self.cached_mesh_name = obj.data.name
        self.cached_bias = bias
        self.cached_draw_faces = settings.draw_faces
        self.cached_draw_edges = settings.draw_edges
        self.cached_draw_verts = settings.draw_verts
        self.is_dirty = False


_batch_cache = OverlayBatchCache()


def invalidate_overlay_cache():
    """Forces the batch cache to re-extract geometry on the next redraw."""
    global _batch_cache
    _batch_cache.invalidate()


def get_cached_counts(obj):
    """Returns cached (locked_count, sel_locked_count, total_verts) if available."""
    global _batch_cache
    if _batch_cache.cached_obj_name == obj.name and not _batch_cache.is_dirty:
        return _batch_cache.locked_count, _batch_cache.sel_locked_count, _batch_cache.total_verts
    return None


@bpy.app.handlers.persistent
def on_depsgraph_update(scene, depsgraph):
    """Invalidates the batch cache when mesh geometry changes."""
    global _batch_cache
    for update in depsgraph.updates:
        if isinstance(update.id, (bpy.types.Object, bpy.types.Mesh)):
            if getattr(update, 'is_updated_geometry', False):
                _batch_cache.invalidate()
                break


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


def tag_redraw_view3d(invalidate_cache: bool = True):
    """Forces all 3D viewports to immediately redraw."""
    if invalidate_cache:
        invalidate_overlay_cache()
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def draw_callback_view3d():
    """Renders visual overlay for locked vertices, edges, and faces in 3D viewport."""
    global _batch_cache
    context = bpy.context
    scene = context.scene
    settings = getattr(scene, 'vertex_lock_settings', None)
    if not settings or not settings.show_overlay:
        return

    obj = context.active_object
    if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
        return

    scale = max(obj.matrix_world.to_scale()) if hasattr(obj.matrix_world, 'to_scale') else 1.0
    bias = (0.0003 / max(0.001, scale)) if (not settings.x_ray) else 0.0
    bias = min(max(bias, 0.0), 0.001)

    needs_rebuild = (
        _batch_cache.is_dirty
        or _batch_cache.cached_obj_name != obj.name
        or _batch_cache.cached_mesh_name != obj.data.name
        or abs(_batch_cache.cached_bias - bias) > 1e-6
        or _batch_cache.cached_draw_faces != settings.draw_faces
        or _batch_cache.cached_draw_edges != settings.draw_edges
        or _batch_cache.cached_draw_verts != settings.draw_verts
    )

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    if needs_rebuild:
        try:
            bm = bmesh.from_edit_mesh(obj.data)
        except Exception:
            return

        vert_lock_layer = bm.verts.layers.int.get("vertex_lock")
        if not vert_lock_layer:
            _batch_cache.clear()
            _batch_cache.cached_obj_name = obj.name
            _batch_cache.cached_mesh_name = obj.data.name
            _batch_cache.is_dirty = False
            return

        face_lock_layer = bm.faces.layers.int.get("face_lock")
        _batch_cache.update_cache(obj, bm, vert_lock_layer, face_lock_layer, settings, bias, shader)

    if not _batch_cache.has_locked:
        return

    gpu.matrix.push()
    gpu.matrix.multiply_matrix(obj.matrix_world)

    if settings.x_ray:
        gpu.state.depth_test_set('ALWAYS')
    else:
        gpu.state.depth_test_set('LESS_EQUAL')

    # Prevent transparent overlays from contaminating the depth buffer
    gpu.state.depth_mask_set(False)
    gpu.state.blend_set('ALPHA')

    try:
        # 1. Faces
        if settings.draw_faces and (_batch_cache.batch_f_unsel or _batch_cache.batch_f_sel):
            if is_backface_culling_active(context):
                is_neg_scale = (obj.matrix_world.determinant() < 0)
                gpu.state.face_culling_set('FRONT' if is_neg_scale else 'BACK')
            else:
                gpu.state.face_culling_set('NONE')

            if _batch_cache.batch_f_unsel:
                shader.uniform_float('color', settings.lock_color_face)
                _batch_cache.batch_f_unsel.draw(shader)

            if _batch_cache.batch_f_sel:
                shader.uniform_float('color', (1.0, 0.70, 0.10, 0.38))
                _batch_cache.batch_f_sel.draw(shader)

            gpu.state.face_culling_set('NONE')

        # 2. Edges
        if settings.draw_edges and (_batch_cache.batch_e_unsel or _batch_cache.batch_e_sel):
            try:
                gpu.state.line_width_set(float(settings.line_width))
            except Exception:
                pass

            if _batch_cache.batch_e_unsel:
                shader.uniform_float('color', settings.lock_color_line)
                _batch_cache.batch_e_unsel.draw(shader)

            if _batch_cache.batch_e_sel:
                shader.uniform_float('color', (1.0, 0.75, 0.15, 0.95))
                _batch_cache.batch_e_sel.draw(shader)

        # 3. Vertices
        if settings.draw_verts and (_batch_cache.batch_v_unsel or _batch_cache.batch_v_sel):
            try:
                gpu.state.point_size_set(float(settings.point_size))
            except Exception:
                pass

            if _batch_cache.batch_v_unsel:
                shader.uniform_float('color', settings.lock_color_point)
                _batch_cache.batch_v_unsel.draw(shader)

            if _batch_cache.batch_v_sel:
                shader.uniform_float('color', (1.0, 1.0, 1.0, 1.0))
                _batch_cache.batch_v_sel.draw(shader)

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


@dataclass
class SelectionRay:
    """Encapsulates 3D ray information for viewport selection (Interface Segregation)."""
    origin_world: Vector
    direction_world: Vector
    origin_local: Vector
    direction_local: Vector
    view_matrix: Matrix
    mouse_x: float
    mouse_y: float
    is_perspective: bool


@dataclass
class CandidateHit:
    """Represents a candidate mesh element hit during selection testing."""
    elem_type: str  # 'VERT', 'EDGE', 'FACE'
    elem: Union[bmesh.types.BMVert, bmesh.types.BMEdge, bmesh.types.BMFace]
    depth: float  # Camera space distance along ray (smaller = closer to camera)
    is_locked: bool
    screen_dist: float = 0.0  # 2D pixel distance to cursor (for verts/edges)


class IRaycastService(ABC):
    """Interface for viewport raycasting."""
    @abstractmethod
    def create_selection_ray(self, context, event, obj: bpy.types.Object) -> Optional[SelectionRay]:
        pass

    @abstractmethod
    def raycast_mesh(self, bm: bmesh.types.BMesh, ray: SelectionRay) -> Tuple[Optional[Vector], Optional[Vector], Optional[int], Optional[float]]:
        pass


def is_backface_culling_active(context) -> bool:
    """Checks if Backface Culling is enabled in Blender's current 3D Viewport shading settings."""
    try:
        space = getattr(context, 'space_data', None)
        if space and getattr(space, 'type', None) == 'VIEW_3D' and hasattr(space, 'shading'):
            return bool(getattr(space.shading, 'show_backface_culling', False))
        if hasattr(context, 'area') and context.area and context.area.type == 'VIEW_3D':
            for sp in context.area.spaces:
                if sp.type == 'VIEW_3D' and hasattr(sp, 'shading'):
                    return bool(getattr(sp.shading, 'show_backface_culling', False))
    except Exception:
        pass
    return False


def get_viewport_rv3d(context):
    space = getattr(context, 'space_data', None)
    if space and getattr(space, 'type', None) == 'VIEW_3D':
        return getattr(space, 'region_3d', None)
    return getattr(context, 'region_data', None)


class RaycastService(IRaycastService):
    """Responsible solely for viewport coordinate conversion and 3D raycasting (Single Responsibility)."""

    def create_selection_ray(self, context, event, obj: bpy.types.Object) -> Optional[SelectionRay]:
        region = getattr(context, 'region', None)
        rv3d = get_viewport_rv3d(context)
        if not region or not rv3d:
            return None

        mouse_x = event.mouse_region_x
        mouse_y = event.mouse_region_y

        ray_origin_world = view3d_utils.region_2d_to_origin_3d(region, rv3d, (mouse_x, mouse_y))
        ray_direction_world = view3d_utils.region_2d_to_vector_3d(region, rv3d, (mouse_x, mouse_y))
        if ray_origin_world is None or ray_direction_world is None:
            return None

        ray_direction_world = ray_direction_world.normalized()

        mat_inv = obj.matrix_world.inverted()
        origin_local = mat_inv @ ray_origin_world
        direction_local = (mat_inv.to_3x3() @ ray_direction_world).normalized()

        return SelectionRay(
            origin_world=ray_origin_world,
            direction_world=ray_direction_world,
            origin_local=origin_local,
            direction_local=direction_local,
            view_matrix=rv3d.view_matrix.copy(),
            mouse_x=mouse_x,
            mouse_y=mouse_y,
            is_perspective=getattr(rv3d, 'is_perspective', True),
        )

    def raycast_mesh(self, bm: bmesh.types.BMesh, ray: SelectionRay) -> Tuple[Optional[Vector], Optional[Vector], Optional[int], Optional[float]]:
        try:
            bvh = BVHTree.FromBMesh(bm)
            loc, norm, idx, dist = bvh.ray_cast(ray.origin_local, ray.direction_local)
            return loc, norm, idx, dist
        except Exception:
            return None, None, None, None


class IOcclusionService(ABC):
    """Interface for depth and occlusion verification."""
    @abstractmethod
    def get_bvh(self, bm: bmesh.types.BMesh) -> Optional[BVHTree]:
        pass

    @abstractmethod
    def is_point_occluded(
        self,
        bm: bmesh.types.BMesh,
        point_local: Vector,
        ray_origin_local: Vector,
        tolerance: float = 0.005,
    ) -> bool:
        pass


class OcclusionService(IOcclusionService):
    """Evaluates whether a point in space is occluded by mesh geometry along the direct line of sight."""

    def __init__(self, bvh: Optional[BVHTree] = None):
        self._bvh = bvh

    def get_bvh(self, bm: bmesh.types.BMesh) -> Optional[BVHTree]:
        if self._bvh is None:
            try:
                self._bvh = BVHTree.FromBMesh(bm)
            except Exception:
                self._bvh = None
        return self._bvh

    def is_point_occluded(
        self,
        bm: bmesh.types.BMesh,
        point_local: Vector,
        ray_origin_local: Vector,
        tolerance: float = 0.005,
    ) -> bool:
        bvh = self.get_bvh(bm)
        if bvh is None:
            return False

        view_vec = point_local - ray_origin_local
        dist_to_point = view_vec.length
        if dist_to_point <= 1e-5:
            return False

        direction = view_vec / dist_to_point
        loc, norm, idx, dist = bvh.ray_cast(ray_origin_local, direction)
        if loc is not None and dist is not None:
            if dist < (dist_to_point - tolerance):
                return True
        return False


class IElementPicker(ABC):
    """Abstract Strategy for picking candidates of a specific mesh element type (Open/Closed, Liskov Substitution)."""
    @abstractmethod
    def pick_candidate(
        self,
        context,
        obj: bpy.types.Object,
        bm: bmesh.types.BMesh,
        ray: SelectionRay,
        vert_lock_layer,
        face_lock_layer,
        occlusion: IOcclusionService,
        xray_active: bool,
    ) -> Optional[CandidateHit]:
        pass


class FacePicker(IElementPicker):
    """Picks faces using 3D raycasting with backface culling and depth verification."""

    def is_face_locked(self, face, vert_lock_layer, face_lock_layer) -> bool:
        if face_lock_layer and face[face_lock_layer] >= 1:
            return True
        return all(v[vert_lock_layer] >= 1 or v.hide for v in face.verts)

    def pick_candidate(
        self,
        context,
        obj: bpy.types.Object,
        bm: bmesh.types.BMesh,
        ray: SelectionRay,
        vert_lock_layer,
        face_lock_layer,
        occlusion: IOcclusionService,
        xray_active: bool,
    ) -> Optional[CandidateHit]:
        bm.faces.ensure_lookup_table()
        bvh = occlusion.get_bvh(bm)
        if bvh is not None:
            loc, norm, idx, dist = bvh.ray_cast(ray.origin_local, ray.direction_local)

            if loc is not None and idx is not None and 0 <= idx < len(bm.faces):
                hit_face = bm.faces[idx]
                cull_backface = (
                    not xray_active
                    and is_backface_culling_active(context)
                    and norm is not None
                    and norm.dot(ray.direction_local) > 0
                )
                if not cull_backface:
                    is_locked = self.is_face_locked(hit_face, vert_lock_layer, face_lock_layer)
                    return CandidateHit(
                        elem_type='FACE',
                        elem=hit_face,
                        depth=dist if dist is not None else 0.0,
                        is_locked=is_locked,
                        screen_dist=0.0
                    )

        region = getattr(context, 'region', None)
        rv3d = get_viewport_rv3d(context)
        if not region or not rv3d:
            return None

        mat = obj.matrix_world
        view_mat = rv3d.view_matrix
        mouse_vec = Vector((ray.mouse_x, ray.mouse_y))

        best_hit = None
        min_depth = float('inf')

        for f in bm.faces:
            pts_2d = [view3d_utils.location_3d_to_region_2d(region, rv3d, mat @ v.co) for v in f.verts]
            if any(p is None for p in pts_2d):
                continue

            inside = False
            for i in range(1, len(pts_2d) - 1):
                if mathutils.geometry.intersect_point_tri_2d(mouse_vec, pts_2d[0], pts_2d[i], pts_2d[i + 1]):
                    inside = True
                    break

            if inside:
                center_local = f.calc_center_median()
                center_world = mat @ center_local
                view_z = (view_mat @ center_world).z
                cam_dist = -view_z

                if not xray_active:
                    if occlusion.is_point_occluded(bm, center_local, ray.origin_local):
                        continue

                is_locked = self.is_face_locked(f, vert_lock_layer, face_lock_layer)
                if cam_dist < min_depth:
                    min_depth = cam_dist
                    best_hit = CandidateHit(
                        elem_type='FACE',
                        elem=f,
                        depth=cam_dist,
                        is_locked=is_locked,
                        screen_dist=0.0
                    )

        return best_hit


class VertexPicker(IElementPicker):
    """Picks vertices within screen radius, sorted by camera depth and screen distance."""

    def __init__(self, tolerance_px: float = 14.0):
        self.tolerance_px = tolerance_px

    def is_vert_locked(self, v, vert_lock_layer) -> bool:
        return v[vert_lock_layer] >= 1 or v.hide

    def pick_candidate(
        self,
        context,
        obj: bpy.types.Object,
        bm: bmesh.types.BMesh,
        ray: SelectionRay,
        vert_lock_layer,
        face_lock_layer,
        occlusion: IOcclusionService,
        xray_active: bool,
    ) -> Optional[CandidateHit]:
        region = getattr(context, 'region', None)
        rv3d = get_viewport_rv3d(context)
        if not region or not rv3d:
            return None

        mat = obj.matrix_world
        view_mat = rv3d.view_matrix
        mouse_x = ray.mouse_x
        mouse_y = ray.mouse_y
        tol_sq = self.tolerance_px * self.tolerance_px

        candidates = []

        for v in bm.verts:
            world_co = mat @ v.co
            screen_co = view3d_utils.location_3d_to_region_2d(region, rv3d, world_co)
            if screen_co is None:
                continue

            dx = screen_co.x - mouse_x
            dy = screen_co.y - mouse_y
            dist_sq = dx * dx + dy * dy

            if dist_sq <= tol_sq:
                cam_dist = -(view_mat @ world_co).z

                if not xray_active:
                    if occlusion.is_point_occluded(bm, v.co, ray.origin_local):
                        continue

                is_locked = self.is_vert_locked(v, vert_lock_layer)
                candidates.append(CandidateHit(
                    elem_type='VERT',
                    elem=v,
                    depth=cam_dist,
                    is_locked=is_locked,
                    screen_dist=dist_sq ** 0.5
                ))

        if not candidates:
            return None

        def sort_key(c: CandidateHit):
            return (round(c.depth, 3), c.screen_dist)

        candidates.sort(key=sort_key)
        return candidates[0]


class EdgePicker(IElementPicker):
    """Picks edges within screen radius, sorted by camera depth and screen distance."""

    def __init__(self, tolerance_px: float = 12.0):
        self.tolerance_px = tolerance_px

    def is_edge_locked(self, e, vert_lock_layer) -> bool:
        return (e.verts[0][vert_lock_layer] >= 1 or e.verts[0].hide) and \
               (e.verts[1][vert_lock_layer] >= 1 or e.verts[1].hide)

    def pick_candidate(
        self,
        context,
        obj: bpy.types.Object,
        bm: bmesh.types.BMesh,
        ray: SelectionRay,
        vert_lock_layer,
        face_lock_layer,
        occlusion: IOcclusionService,
        xray_active: bool,
    ) -> Optional[CandidateHit]:
        region = getattr(context, 'region', None)
        rv3d = get_viewport_rv3d(context)
        if not region or not rv3d:
            return None

        mat = obj.matrix_world
        view_mat = rv3d.view_matrix
        mouse_vec = Vector((ray.mouse_x, ray.mouse_y, 0.0))

        candidates = []

        for e in bm.edges:
            p0 = view3d_utils.location_3d_to_region_2d(region, rv3d, mat @ e.verts[0].co)
            p1 = view3d_utils.location_3d_to_region_2d(region, rv3d, mat @ e.verts[1].co)
            if p0 is None or p1 is None:
                continue

            p0_3d = Vector((p0.x, p0.y, 0.0))
            p1_3d = Vector((p1.x, p1.y, 0.0))
            proj, factor = mathutils.geometry.intersect_point_line(mouse_vec, p0_3d, p1_3d)

            if 0.0 <= factor <= 1.0:
                dist = (mouse_vec - proj).length
                mid_co_local = (e.verts[0].co + e.verts[1].co) * 0.5
            elif factor < 0.0:
                dist = (mouse_vec - p0_3d).length
                mid_co_local = e.verts[0].co
            else:
                dist = (mouse_vec - p1_3d).length
                mid_co_local = e.verts[1].co

            if dist <= self.tolerance_px:
                mid_world = mat @ mid_co_local
                cam_dist = -(view_mat @ mid_world).z

                if not xray_active:
                    if occlusion.is_point_occluded(bm, mid_co_local, ray.origin_local):
                        continue

                is_locked = self.is_edge_locked(e, vert_lock_layer)
                candidates.append(CandidateHit(
                    elem_type='EDGE',
                    elem=e,
                    depth=cam_dist,
                    is_locked=is_locked,
                    screen_dist=dist
                ))

        if not candidates:
            return None

        def sort_key(c: CandidateHit):
            return (round(c.depth, 3), c.screen_dist)

        candidates.sort(key=sort_key)
        return candidates[0]


class LockSelectionResolver:
    """
    Coordinates selection picking across active mesh selection modes (Dependency Inversion).
    Enforces Blender priority:
    1. The closest element to the camera always takes priority.
    2. If the closest element is unlocked, returns (None, None) so Blender's native select handles it.
    3. If elements have near-identical depth, Blender priority is Vert > Edge > Face.
    """

    def __init__(
        self,
        raycast_service: Optional[IRaycastService] = None,
        vert_picker: Optional[IElementPicker] = None,
        edge_picker: Optional[IElementPicker] = None,
        face_picker: Optional[IElementPicker] = None,
    ):
        self.raycast_service = raycast_service or RaycastService()
        self.vert_picker = vert_picker or VertexPicker()
        self.edge_picker = edge_picker or EdgePicker()
        self.face_picker = face_picker or FacePicker()

    def resolve(
        self,
        context,
        event,
        obj: bpy.types.Object,
        bm: bmesh.types.BMesh,
        vert_lock_layer,
        face_lock_layer,
    ) -> Tuple[Optional[str], Optional[Union[bmesh.types.BMVert, bmesh.types.BMEdge, bmesh.types.BMFace]]]:
        ray = self.raycast_service.create_selection_ray(context, event, obj)
        if not ray:
            return None, None

        settings = getattr(context.scene, 'vertex_lock_settings', None)
        xray_active = self._is_xray(context, settings)

        occlusion = OcclusionService()

        select_mode = context.tool_settings.mesh_select_mode
        is_vert_mode, is_edge_mode, is_face_mode = select_mode

        mode_hits: List[CandidateHit] = []

        if is_vert_mode:
            v_hit = self.vert_picker.pick_candidate(
                context, obj, bm, ray, vert_lock_layer, face_lock_layer, occlusion, xray_active
            )
            if v_hit:
                mode_hits.append(v_hit)

        if is_edge_mode:
            e_hit = self.edge_picker.pick_candidate(
                context, obj, bm, ray, vert_lock_layer, face_lock_layer, occlusion, xray_active
            )
            if e_hit:
                mode_hits.append(e_hit)

        if is_face_mode:
            f_hit = self.face_picker.pick_candidate(
                context, obj, bm, ray, vert_lock_layer, face_lock_layer, occlusion, xray_active
            )
            if f_hit:
                mode_hits.append(f_hit)

        if not mode_hits:
            return None, None

        type_priority = {'VERT': 0, 'EDGE': 1, 'FACE': 2}

        def overall_sort_key(hit: CandidateHit):
            return (round(hit.depth, 2), type_priority.get(hit.elem_type, 3), hit.screen_dist)

        mode_hits.sort(key=overall_sort_key)
        winning_hit = mode_hits[0]

        if not winning_hit.is_locked:
            return None, None

        return winning_hit.elem_type, winning_hit.elem

    def _is_xray(self, context, settings) -> bool:
        if settings and getattr(settings, 'x_ray', False):
            return True
        space = getattr(context, 'space_data', None)
        if space and space.type == 'VIEW_3D' and hasattr(space, 'shading'):
            if getattr(space.shading, 'show_xray', False):
                return True
            if getattr(space.shading, 'type', '') == 'WIREFRAME':
                return True
        return False


_selection_resolver = LockSelectionResolver()


def find_locked_element_under_mouse(context, event, obj, bm, vert_lock_layer, face_lock_layer):
    """
    Resolves selection under mouse cursor using LockSelectionResolver.
    Returns ('VERT'|'EDGE'|'FACE', element) if a locked element is under the cursor and has priority.
    If an unlocked element is closer to the camera or in front, returns (None, None) to let Blender handle it.
    """
    return _selection_resolver.resolve(context, event, obj, bm, vert_lock_layer, face_lock_layer)


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
            cached_counts = get_cached_counts(obj)
            if cached_counts is not None:
                locked_count, sel_locked_count, total_verts = cached_counts
            else:
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

    # Lock Selected: Ctrl + Alt + L (avoids overriding Blender's native Shift+L Deselect Linked)
    kmi_lock = km.keymap_items.new(
        "mesh.vertex_lock_selected",
        type='L',
        value='PRESS',
        ctrl=True,
        alt=True
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

    if on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)

    register_keymaps()


def unregister():
    unregister_keymaps()

    global _draw_handler, _batch_cache
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        _draw_handler = None

    if on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update)

    _batch_cache.clear()
    tag_redraw_view3d(invalidate_cache=True)

    if hasattr(bpy.types.Scene, 'vertex_lock_settings'):
        del bpy.types.Scene.vertex_lock_settings

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


if __name__ == "__main__":
    register()
