import bpy
import bmesh
import gpu
from gpu_extras.batch import batch_for_shader

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
    """Renders visual overlay for locked vertices, edges, and faces in 3D viewport."""
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


def register():
    global _draw_handler
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback_view3d, (), 'WINDOW', 'POST_VIEW'
        )

def unregister():
    global _draw_handler
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        _draw_handler = None
    tag_redraw_view3d()
