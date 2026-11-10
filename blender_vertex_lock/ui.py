import bpy
import bmesh
from bpy.props import StringProperty
from .draw import tag_redraw_view3d

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
    bl_idname = __package__

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

        # Shortcuts Section
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


classes = (
    VertexLockAddonPreferences,
    MESH_PT_vertex_lock_panel,
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
