import bpy
import glob
import sys
import os
import math

# --- CONFIGURATION ---
RENDER_RESOLUTION = 2048
SAMPLES = 256
USE_DENOISER = True
CAMERA_FOCAL_LENGTH = 120  
BEVEL_RADIUS = 0.002
BUMP_STRENGTH = 0.03 
ROTATION_ANGLE_Z = 30  # Rotation in Degrees
# ---------------------

def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def apply_realism_modifiers(obj):
    bpy.context.view_layer.objects.active = obj
    bevel = obj.modifiers.new(name="Bevel", type='BEVEL')
    bevel.width = BEVEL_RADIUS
    bevel.segments = 3
    bevel.profile = 0.7
    bevel.limit_method = 'ANGLE'
    bevel.angle_limit = math.radians(30)
    bevel.harden_normals = True 
    
    for poly in obj.data.polygons:
        poly.use_smooth = True
        
    if bpy.app.version >= (4, 1, 0):
        if "Smooth by Angle" not in obj.modifiers:
            try:
                bpy.ops.object.modifier_add_node_group(asset_library_type='ESSENTIALS', asset_library_identifier="", relative_asset_identifier="geometry_nodes/smooth_by_angle.blend/NodeTree/Smooth by Angle")
            except: pass 
    else:
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = math.radians(30)

def create_material(name, color, metallic, roughness, transmission=0.0, alpha=1.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)
    
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Transmission Weight'].default_value = transmission
    bsdf.inputs['Alpha'].default_value = alpha
    if alpha < 1.0:
        mat.blend_method = 'BLEND'

    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    if transmission == 0.0:
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        tex_coord.location = (-800, 200)
        noise = nodes.new(type='ShaderNodeTexNoise')
        noise.location = (-600, 200)
        noise.inputs['Scale'].default_value = 100.0
        noise.inputs['Detail'].default_value = 15.0
        bump = nodes.new(type='ShaderNodeBump')
        bump.location = (-300, -100)
        bump.inputs['Strength'].default_value = BUMP_STRENGTH
        
        links.new(tex_coord.outputs['Object'], noise.inputs['Vector'])
        links.new(noise.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    
    return mat

def get_base_color_from_material(mat):
    default_color = (0.5, 0.5, 0.5, 1.0)
    if not mat or not mat.use_nodes: return default_color
    bsdf = None
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            bsdf = node
            break
    if bsdf: return bsdf.inputs['Base Color'].default_value[:]
    return default_color

def setup_scene(gltf_path, output_path, hdri_path=None):
    clean_scene()
    bpy.ops.import_scene.gltf(filepath=gltf_path)
    
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not mesh_objects: return

    print("\n--- Auto-Assigning Materials ---")
    
    mat_lib_defs = {
        "copper":   ("Copper", (1.0, 0.55, 0.35, 1.0), 1.0, 0.25),
        "cu":       ("Copper", (1.0, 0.55, 0.35, 1.0), 1.0, 0.25),
        "steel":    ("Steel",  (0.25, 0.25, 0.27, 1.0), 0.9, 0.35),
        "aluminum": ("Aluminum", (0.8, 0.82, 0.85, 1.0), 0.9, 0.3),
        "al":       ("Aluminum", (0.8, 0.82, 0.85, 1.0), 0.9, 0.3),
        "gold":     ("Gold",   (1.0, 0.76, 0.2, 1.0), 1.0, 0.15),
        "au":       ("Gold",   (1.0, 0.76, 0.2, 1.0), 1.0, 0.15),
        "wood":     ("Wood",   (0.26, 0.14, 0.06, 1.0), 0.0, 0.7),
        "oak":      ("Wood",   (0.26, 0.14, 0.06, 1.0), 0.0, 0.7),
        "pine":     ("Pine",   (0.60, 0.45, 0.25, 1.0), 0.0, 0.6),
        "stone":    ("Stone",  (0.4, 0.4, 0.42, 1.0),   0.0, 0.9),
        "granite":  ("Stone",  (0.4, 0.4, 0.42, 1.0),   0.0, 0.9),
        "bronze":   ("Bronze", (0.65, 0.35, 0.15, 1.0), 1.0, 0.3),
        "brass":    ("Brass",    (0.85, 0.70, 0.35, 1.0), 1.0, 0.2),
        "chrome":   ("Chrome",   (0.9, 0.9, 0.92, 1.0),   1.0, 0.05),
        "titanium": ("Titanium", (0.55, 0.53, 0.51, 1.0), 0.9, 0.35),
        "rubber":   ("Rubber",   (0.05, 0.05, 0.05, 1.0), 0.0, 0.8),
        "ceramic":  ("Ceramic",  (0.92, 0.92, 0.92, 1.0), 0.0, 0.1),
        "concrete": ("Concrete", (0.5, 0.5, 0.52, 1.0),   0.0, 0.95),
        "black":    ("SatinBlack", (0.02, 0.02, 0.02, 1.0), 0.2, 0.5),
        "plastic":  ("Plastic", (0.1, 0.1, 0.1, 1.0), 0.0, 0.5),
    }
    
    cached_mats = {}

    for obj in mesh_objects:
        obj_name_lower = obj.name.lower()
        assigned = False

        if "glass" in obj_name_lower:
            print(f"  Object '{obj.name}' -> Detected GLASS")
            if "Glass50" not in cached_mats:
                cached_mats["Glass50"] = create_material("Glass50", (1,1,1,1), 0.0, 0.05, transmission=0.2, alpha=0.8)
            obj.data.materials.clear()
            obj.data.materials.append(cached_mats["Glass50"])
            assigned = True
            
        if not assigned:
            for keyword, data in mat_lib_defs.items():
                if keyword in obj_name_lower:
                    mat_name, color, metal, rough = data
                    print(f"  Object '{obj.name}' -> Detected '{keyword.upper()}' -> Assigning {mat_name}")
                    if mat_name not in cached_mats:
                        cached_mats[mat_name] = create_material(mat_name, color, metal, rough)
                    obj.data.materials.clear()
                    obj.data.materials.append(cached_mats[mat_name])
                    assigned = True
                    break
        
        if not assigned:
            original_color = (0.5, 0.5, 0.5, 1.0)
            if obj.data.materials:
                original_color = get_base_color_from_material(obj.data.materials[0])
            
            color_key = f"CustomMetal_{original_color[0]:.2f}_{original_color[1]:.2f}_{original_color[2]:.2f}"
            if color_key not in cached_mats:
                print(f"  Object '{obj.name}' -> No keyword. Enhancing imported color.")
                cached_mats[color_key] = create_material(color_key, original_color, 0.8, 0.4)
            obj.data.materials.clear()
            obj.data.materials.append(cached_mats[color_key])

    print("--------------------------------\n")
    
    # Join
    anchor = mesh_objects[0]
    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objects: obj.select_set(True)
    bpy.context.view_layer.objects.active = anchor
    
    if len(mesh_objects) > 1: bpy.ops.object.join()
    main_obj = bpy.context.active_object
    
    apply_realism_modifiers(main_obj)
    
    # --- TRANSFORM FIXES ---
    
    # 1. Clear any Animation Data (glTF often locks rotation with animations)
    if main_obj.animation_data:
        print("Clearing imported animation data...")
        main_obj.animation_data_clear()

    # 2. Reset Origin
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    main_obj.location = (0, 0, 0)
    
    # 3. Normalize Scale
    max_dim = max(main_obj.dimensions)
    scale_factor = 2.0 / max_dim if max_dim > 0 else 1.0
    main_obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(scale=True)
    
    # 4. Position on Floor
    main_obj.location.z = main_obj.dimensions.z / 2
    
    # 5. ROTATION FIX: FORCE EULER MODE
    # glTF defaults to 'QUATERNION', which ignores 'rotation_euler' commands.
    print(f"Rotating object by {ROTATION_ANGLE_Z} degrees...")
    main_obj.rotation_mode = 'XYZ' 
    main_obj.rotation_euler.z += math.radians(ROTATION_ANGLE_Z)
    
    # -----------------------

    # Environment
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    
    if hdri_path and os.path.exists(hdri_path):
        print(f"Using HDRI: {hdri_path}")
        tree = world.node_tree
        bg_node = tree.nodes['Background']
        env_node = tree.nodes.new('ShaderNodeTexEnvironment')
        env_node.image = bpy.data.images.load(hdri_path)
        hue_sat = tree.nodes.new('ShaderNodeHueSaturation')
        hue_sat.inputs['Saturation'].default_value = 0.0 
        tex_coord = tree.nodes.new('ShaderNodeTexCoord')
        mapping = tree.nodes.new('ShaderNodeMapping')
        mapping.inputs['Rotation'].default_value[2] = math.radians(220)
        tree.links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        tree.links.new(mapping.outputs['Vector'], env_node.inputs['Vector'])
        tree.links.new(env_node.outputs['Color'], hue_sat.inputs['Color'])
        tree.links.new(hue_sat.outputs['Color'], bg_node.inputs['Color'])
        
        bpy.ops.mesh.primitive_plane_add(size=100)
        floor = bpy.context.active_object
        floor.is_shadow_catcher = True
    else:
        print("No HDRI found. Using Studio Lights.")
        bpy.ops.mesh.primitive_plane_add(size=100)
        key = bpy.data.lights.new("Key", 'AREA')
        key.energy = 500
        key.size = 2
        k_obj = bpy.data.objects.new("Key", key)
        bpy.context.scene.collection.objects.link(k_obj)
        k_obj.location = (4, -4, 5)
        k_obj.constraints.new(type='TRACK_TO').target = main_obj

    # Camera
    cam_data = bpy.data.cameras.new('Camera')
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_data.lens = CAMERA_FOCAL_LENGTH
    cam_obj.location = (8, -8, 6)
    const = cam_obj.constraints.new(type='TRACK_TO')
    const.target = main_obj
    const.track_axis = 'TRACK_NEGATIVE_Z'
    const.up_axis = 'UP_Y'

    # Settings
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = SAMPLES
    scene.render.film_transparent = True 
    
    try:
        scene.view_settings.view_transform = 'AgX'
        scene.view_settings.look = 'AgX - High Contrast'
    except TypeError:
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look = 'High Contrast'
        
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.refresh_devices()
    if prefs.devices:
        scene.cycles.device = 'GPU'
        for dev in prefs.devices: dev.use = True
    else:
        scene.cycles.device = 'CPU'

    scene.render.resolution_x = RENDER_RESOLUTION
    scene.render.resolution_y = RENDER_RESOLUTION
    scene.render.filepath = output_path
    
    print(f"Rendering {output_path}...")
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    hdri_file = None
    potential_hdris = glob.glob("*.exr") + glob.glob("*.hdr")
    if potential_hdris:
        hdri_file = os.path.abspath(potential_hdris[0])
    argv = sys.argv
    try:
        index = argv.index("--") + 1
    except ValueError:
        index = len(argv)
    args = argv[index:]
    if len(args) < 1:
        print("Usage: blender -b -P auto_material_render_v3.py -- <gltf_filename>")
    else:
        output_dir = os.path.join(os.getcwd(), "renders_auto_mat_v3")
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        files = glob.glob(args[0])
        print(f"Found {len(files)} files.")
        for f in files:
            name_root = os.path.splitext(os.path.basename(f))[0]
            out_name = os.path.join(output_dir, name_root + ".png")
            setup_scene(f, out_name, hdri_file)