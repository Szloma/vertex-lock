bl_info = {
    "name": "Vertex Lock",
    "author": "Antigravity",
    "version": (1, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar (N-Panel) > Vertex Lock",
    "description": "Lock vertices, edges, and faces in place to prevent them from moving during proportional editing",
    "warning": "",
    "doc_url": "",
    "category": "Mesh",
}

import bpy
from . import properties
from . import operators
from . import ui
from . import draw

addon_keymaps = []

def register_keymaps():
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if not kc:
        return

    # Register within Mesh keymap (active in Edit Mode)
    km = kc.keymaps.new(name="Mesh", space_type='EMPTY')

    # Lock Selected: Shift + L
    kmi_lock = km.keymap_items.new(
        "mesh.vertex_lock_selected",
        type='L',
        value='PRESS',
        shift=True
    )
    addon_keymaps.append((km, kmi_lock))

    # Unlock Selected: Alt + Shift + L
    kmi_unlock = km.keymap_items.new(
        "mesh.vertex_unlock_selected",
        type='L',
        value='PRESS',
        alt=True,
        shift=True
    )
    addon_keymaps.append((km, kmi_unlock))

    # Toggle Lock: Ctrl + Shift + L
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

    # Shift-Click Multi-Select Locked Elements: Shift + Left Mouse Click
    kmi_shift_click = km.keymap_items.new(
        "mesh.vertex_lock_click_select",
        type='LEFTMOUSE',
        value='CLICK',
        shift=True
    )
    kmi_shift_click.properties.extend = True
    addon_keymaps.append((km, kmi_shift_click))

    # Support Right-Click Select preference if configured
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


def register():
    properties.register()
    operators.register()
    ui.register()
    draw.register()
    register_keymaps()


def unregister():
    unregister_keymaps()
    draw.unregister()
    ui.unregister()
    operators.unregister()
    properties.unregister()


if __name__ == "__main__":
    register()
