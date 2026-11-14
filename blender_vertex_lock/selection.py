"""
Selection and Occlusion Services for Vertex Lock Add-on.
Implements SOLID principles for robust 3D mesh element picking in Blender's 3D Viewport.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from bpy_extras import view3d_utils


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
                
                # Check backface culling only when explicitly enabled in Viewport Shading
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

        # Fallback for 2D polygon projection (orthographic bounds or boundary edge cases)
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

        # Sort primarily by depth (closest to camera wins, matching user preference)
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

        # Priority ordering when depth is tied: VERT (0) > EDGE (1) > FACE (2)
        type_priority = {'VERT': 0, 'EDGE': 1, 'FACE': 2}

        def overall_sort_key(hit: CandidateHit):
            return (round(hit.depth, 2), type_priority.get(hit.elem_type, 3), hit.screen_dist)

        mode_hits.sort(key=overall_sort_key)
        winning_hit = mode_hits[0]

        # If winning hit is UNLOCKED: pass through to Blender!
        if not winning_hit.is_locked:
            return None, None

        # Winning hit is LOCKED: return it for add-on selection!
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
