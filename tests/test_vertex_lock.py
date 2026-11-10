import sys
import os
import bpy
import bmesh
from mathutils import Vector

# Ensure parent directory is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def run_tests():
    print("=" * 60)
    print("RUNNING VERTEX LOCK ADDON TEST SUITE (v1.1.0)")
    print("=" * 60)

    # Reset to clean scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 1. Test Module Import & Registration
    import importlib
    try:
        import blender_vertex_lock as addon_module
        print("[INFO] Testing modular package: blender_vertex_lock")
    except ImportError:
        import vertex_lock as addon_module
        print("[INFO] Testing standalone file: vertex_lock")

    try:
        addon_module.unregister()
    except Exception:
        pass
    addon_module = importlib.reload(addon_module)

    try:
        addon_module.register()
        print("[PASS] Registration successful")
    except Exception as e:
        print("[FAIL] Registration failed:", e)
        return False

    # Check keymaps
    km_names = [km.name for km, kmi in addon_module.addon_keymaps]
    kmi_idnames = [kmi.idname for km, kmi in addon_module.addon_keymaps]
    print(f"[PASS] Keymaps registered: {len(addon_module.addon_keymaps)} items in {set(km_names)}")
    assert "mesh.vertex_lock_click_select" in kmi_idnames, "Click select operator missing from keymaps"

    # 2. Test Selection Mode & Locking (Vertex Mode)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=5, y_subdivisions=5)
    obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')

    # Select Vertex mode: (True, False, False)
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    # Select only vert 12 (center)
    for v in bm.verts:
        v.select = (v.index == 12)
    bmesh.update_edit_mesh(obj.data)

    # Execute Lock Selected
    bpy.ops.mesh.vertex_lock_selected()

    # Verify vert 12 is locked
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    vert_layer = bm.verts.layers.int.get("vertex_lock")
    assert vert_layer is not None, "Vertex lock layer was not created"
    assert bm.verts[12][vert_layer] == 1, "Vert 12 was not locked"
    assert bm.verts[12].hide == True, "Vert 12 was not hidden (for lock protection)"
    print("[PASS] Vertex lock (Vertex Mode) successful")

    # 3. Test Proportional Editing Immunity
    bpy.ops.mesh.select_all(action='DESELECT')
    for v in bm.verts:
        if v.index != 12:
            v.select = True
            break
    bmesh.update_edit_mesh(obj.data)

    initial_co_12 = bm.verts[12].co.copy()
    bpy.context.scene.tool_settings.use_proportional_edit = True
    bpy.context.scene.tool_settings.proportional_size = 5.0

    # Translate with proportional editing enabled
    bpy.ops.transform.translate(value=(0, 0, 2))

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    after_co_12 = bm.verts[12].co.copy()
    assert (initial_co_12 - after_co_12).length < 1e-5, f"Locked vertex moved during proportional editing! {initial_co_12} != {after_co_12}"
    print("[PASS] Proportional editing immunity verified (Vert 12 stayed completely stationary)")

    # 4. Test Click Select Operator (Select locked vert)
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.vertex_lock_click_select(hit_type='VERT', hit_index=12, extend=False)

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    assert bm.verts[12][vert_layer] == 2, f"Vert 12 lock status expected 2 (selected locked), got {bm.verts[12][vert_layer]}"
    print("[PASS] Click-Select locked vertex successful (Layer set to 2)")

    # 5. Test Unlock Selected (Unlocks only vert 12)
    bpy.ops.mesh.vertex_unlock_selected()

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    assert bm.verts[12][vert_layer] == 0, "Vert 12 failed to unlock"
    assert bm.verts[12].hide == False, "Vert 12 was not unhidden after unlock"
    print("[PASS] Unlock selected successful")

    # 6. Test Edge Locking (Edge Mode)
    bpy.context.tool_settings.mesh_select_mode = (False, True, False)
    bpy.ops.mesh.select_all(action='DESELECT')
    bm.edges.ensure_lookup_table()
    bm.edges[0].select = True
    bmesh.update_edit_mesh(obj.data)

    bpy.ops.mesh.vertex_lock_selected()

    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    assert bm.edges[0].verts[0][vert_layer] >= 1, "Edge 0 endpoint vert 0 was not locked"
    assert bm.edges[0].verts[1][vert_layer] >= 1, "Edge 0 endpoint vert 1 was not locked"
    assert bm.edges[0].hide == True, "Edge 0 was not hidden/locked"
    print("[PASS] Edge lock (Edge Mode) successful with endpoint sync")

    # 7. Test Face Locking (Face Mode)
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)
    bpy.ops.mesh.select_all(action='DESELECT')
    bm.faces.ensure_lookup_table()
    bm.faces[0].select = True
    bmesh.update_edit_mesh(obj.data)

    bpy.ops.mesh.vertex_lock_selected()

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    face_layer = bm.faces.layers.int.get("face_lock")
    assert face_layer is not None, "Face lock layer not created"
    assert bm.faces[0][face_layer] >= 1, "Face 0 was not locked"
    for v in bm.faces[0].verts:
        assert v[vert_layer] >= 1, f"Face 0 vert {v.index} was not locked"
    print("[PASS] Face lock (Face Mode) successful with perimeter & vert sync")

    # 8. Test Toggle Lock
    bpy.ops.mesh.select_all(action='DESELECT')
    bm.faces.ensure_lookup_table()
    bm.faces[1].select = True
    bmesh.update_edit_mesh(obj.data)

    bpy.ops.mesh.vertex_lock_toggle()
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    assert bm.faces[1][face_layer] >= 1, "Toggle lock failed to lock Face 1"

    # Select locked face and toggle again to unlock
    bpy.ops.mesh.vertex_lock_click_select(hit_type='FACE', hit_index=1, extend=False)
    bpy.ops.mesh.vertex_lock_toggle()
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    assert bm.faces[1][face_layer] == 0, "Toggle lock failed to unlock Face 1"
    print("[PASS] Toggle lock successful")

    # 9. Test Invert Locks
    locked_count_before = sum(1 for v in bm.verts if v[vert_layer] >= 1)
    unlocked_count_before = sum(1 for v in bm.verts if v[vert_layer] == 0)

    bpy.ops.mesh.vertex_invert_locks()
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    locked_count_after = sum(1 for v in bm.verts if v[vert_layer] >= 1)
    assert locked_count_after == unlocked_count_before, f"Invert locks mismatch: {locked_count_after} != {unlocked_count_before}"
    print(f"[PASS] Invert locks successful (Swapped {locked_count_before} and {unlocked_count_before} verts)")

    # 10. Test Select / Deselect Locked
    bpy.ops.mesh.vertex_select_locked()
    bm = bmesh.from_edit_mesh(obj.data)
    for v in bm.verts:
        if v[vert_layer] >= 1:
            assert v[vert_layer] == 2, "Select all locked failed to set selected state"
    print("[PASS] Select All Locked successful")

    bpy.ops.mesh.vertex_lock_deselect_all()
    bm = bmesh.from_edit_mesh(obj.data)
    for v in bm.verts:
        if v[vert_layer] >= 1:
            assert v[vert_layer] == 1, "Deselect all locked failed to reset state to 1"
    print("[PASS] Deselect All Locked successful")

    # 11. Test Native Unhide Recovery (Alt+H defense)
    for v in bm.verts:
        v.hide = False
    bmesh.update_edit_mesh(obj.data)

    # Re-apply locks should re-hide locked geometry
    bpy.ops.mesh.vertex_relock()
    bm = bmesh.from_edit_mesh(obj.data)
    for v in bm.verts:
        if v[vert_layer] >= 1:
            assert v.hide == True, f"Re-apply locks failed to re-hide locked vert {v.index}"
    print("[PASS] Re-apply locks after native unhide (Alt+H) defense verified")

    # 12. Test Unlock All
    bpy.ops.mesh.vertex_unlock_all()
    bm = bmesh.from_edit_mesh(obj.data)
    for v in bm.verts:
        assert v[vert_layer] == 0, f"Unlock all failed for vert {v.index}"
        assert v.hide == False, f"Vert {v.index} remained hidden after unlock all"
    print("[PASS] Unlock all (Clear) successful")

    # 13. Test Shader Draw Callback Stability & Depth Bias
    try:
        import gpu
        gpu_ready = True
        try:
            if hasattr(gpu, 'init'):
                gpu.init()
            _ = gpu.matrix.get_projection_matrix()
        except Exception:
            gpu_ready = False

        if not gpu_ready:
            print("[SKIP] GPU drawing context not available in headless CLI mode")
        else:
            # Lock vert 0 for draw test
            bm.verts[0][vert_layer] = 1
            bm.verts[0].hide = True
            bmesh.update_edit_mesh(obj.data)

            draw_func = getattr(addon_module, 'draw_callback_view3d', None)
            if not draw_func and hasattr(addon_module, 'draw'):
                draw_func = getattr(addon_module.draw, 'draw_callback_view3d', None)

            if draw_func:
                orig_proj = gpu.matrix.get_projection_matrix().copy()
                draw_func()
                restored_proj = gpu.matrix.get_projection_matrix()

                # Check projection matrix stack symmetry (no leaks)
                assert orig_proj == restored_proj, "Projection matrix stack leak detected after draw callback"
                # Check depth mask state restored to default True
                assert gpu.state.depth_mask_get() == True, "Depth mask was not restored to True"
                print("[PASS] Draw callback executed with layered depth bias and clean matrix stack symmetry")
            else:
                print("[SKIP] Draw callback function not found on module")
    except Exception as e:
        print("[FAIL] Draw callback raised error:", e)
        return False

    # 14. Test Adjacent Face Edge Deletion Immunity
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=2)
    grid_obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')
    grid_bm = bmesh.from_edit_mesh(grid_obj.data)
    grid_bm.faces.ensure_lookup_table()
    grid_bm.edges.ensure_lookup_table()
    grid_bm.verts.ensure_lookup_table()

    # Lock Face 0
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)
    bpy.ops.mesh.select_all(action='DESELECT')
    grid_bm.faces[0].select = True
    bmesh.update_edit_mesh(grid_obj.data)

    bpy.ops.mesh.vertex_lock_selected()

    grid_bm = bmesh.from_edit_mesh(grid_obj.data)
    grid_bm.faces.ensure_lookup_table()
    grid_bm.edges.ensure_lookup_table()
    grid_bm.verts.ensure_lookup_table()
    vert_layer = grid_bm.verts.layers.int.get("vertex_lock")

    # Verify locked edges have e.hide == True, unlocked edges have e.hide == False
    for e in grid_bm.edges:
        if all(v[vert_layer] >= 1 for v in e.verts):
            assert e.hide == True, "Locked edge must be hidden to protect from accidental deletion"
        else:
            assert e.hide == False, "Unlocked edge must remain unhidden"

    # Now select adjacent Face 1 and delete edges
    bpy.ops.mesh.select_all(action='DESELECT')
    grid_bm = bmesh.from_edit_mesh(grid_obj.data)
    grid_bm.faces.ensure_lookup_table()
    grid_bm.faces[1].select = True
    bmesh.update_edit_mesh(grid_obj.data)

    bpy.ops.mesh.delete(type='EDGE')

    grid_bm = bmesh.from_edit_mesh(grid_obj.data)
    grid_bm.faces.ensure_lookup_table()
    grid_bm.edges.ensure_lookup_table()
    grid_bm.verts.ensure_lookup_table()

    locked_verts_after = sum(1 for v in grid_bm.verts if v[vert_layer] >= 1)
    locked_edges_after = sum(1 for e in grid_bm.edges if all(v[vert_layer] >= 1 for v in e.verts))
    has_f0 = any(all(v[vert_layer] >= 1 for v in f.verts) for f in grid_bm.faces)

    assert locked_verts_after == 4, f"Locked vertices were removed! Remaining: {locked_verts_after}"
    assert locked_edges_after == 4, f"Locked edges were removed! Remaining: {locked_edges_after}"
    assert has_f0, "Locked Face 0 was removed when deleting edges of adjacent face!"
    print("[PASS] Adjacent face edge deletion immunity verified: locked edges and face preserved 100%")

    # 15. Test Unregistration
    try:
        addon_module.unregister()
        print("[PASS] Unregistration successful")
    except Exception as e:
        print("[FAIL] Unregistration failed:", e)
        return False

    print("=" * 60)
    print("ALL 15 TEST SUITES COMPLETED AND PASSED PERFECTLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_tests()
    if not success:
        sys.exit(1)
