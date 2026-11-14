import bpy
from bpy.props import BoolProperty, FloatVectorProperty, IntProperty, StringProperty, PointerProperty

def _update_overlay_geom(self, context):
    try:
        from .draw import tag_redraw_view3d
        tag_redraw_view3d(invalidate_cache=True)
    except Exception:
        pass

def _update_overlay_uniform(self, context):
    try:
        from .draw import tag_redraw_view3d
        tag_redraw_view3d(invalidate_cache=False)
    except Exception:
        pass

class VertexLockSettings(bpy.types.PropertyGroup):
    show_overlay: BoolProperty(
        name="Overview",
        description="Display red visual overlay for locked vertices, edges, and faces",
        default=True,
        update=_update_overlay_geom
    )
    
    draw_faces: BoolProperty(
        name="Shade Faces",
        description="Shade locked faces in semi-transparent red",
        default=True,
        update=_update_overlay_geom
    )
    
    draw_edges: BoolProperty(
        name="Outline Edges",
        description="Highlight locked edges with red lines",
        default=True,
        update=_update_overlay_geom
    )
    
    draw_verts: BoolProperty(
        name="Show Vertices",
        description="Render locked vertices as visible red points",
        default=True,
        update=_update_overlay_geom
    )
    
    lock_color_point: FloatVectorProperty(
        name="Vertex Color",
        description="Color and opacity for locked vertex points",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 0.15, 0.15, 1.0),
        update=_update_overlay_uniform
    )
    
    lock_color_face: FloatVectorProperty(
        name="Face Color",
        description="Color and opacity for locked faces",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 0.05, 0.05, 0.28),
        update=_update_overlay_uniform
    )
    
    lock_color_line: FloatVectorProperty(
        name="Edge Color",
        description="Color and opacity for locked edge outlines",
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


def register():
    bpy.utils.register_class(VertexLockSettings)
    bpy.types.Scene.vertex_lock_settings = PointerProperty(type=VertexLockSettings)


def unregister():
    if hasattr(bpy.types.Scene, 'vertex_lock_settings'):
        del bpy.types.Scene.vertex_lock_settings
    try:
        bpy.utils.unregister_class(VertexLockSettings)
    except Exception:
        pass
