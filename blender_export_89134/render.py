import bpy
import json
import math
import os
import sys


def arg(name, default):
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
    return default


def set_input(node, names, value):
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return node.inputs[n]
    return None


def find_input(node, names):
    for n in names:
        if n in node.inputs:
            return node.inputs[n]
    return None


def load_image(path, colorspace):
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = colorspace
    return img


here = os.path.dirname(os.path.abspath(__file__))
meta = json.load(open(os.path.join(here, "canvas.json")))
width_m, height_m = meta["size_m"]
cells = meta["cells"]
height_scale = meta["height_scale_m"]
height_max = meta["height_max_m"]
view = arg("--view", "close")
out = arg("--out", os.path.join(here, "render_%s.png" % view))
samples = int(arg("--samples", "256"))
hdri = arg("--hdri", "")
grid = int(arg("--grid", "1536"))
res = int(arg("--res", "3000"))

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = samples
scene.cycles.use_denoising = True
try:
    scene.cycles.device = "GPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            for d in prefs.devices:
                d.use = True
            break
        except Exception:
            continue
except Exception:
    scene.cycles.device = "CPU"
for name in ("AgX", "Filmic"):
    try:
        scene.view_settings.view_transform = name
        break
    except Exception:
        continue
scene.view_settings.look = "None"
scene.view_settings.exposure = float(arg("--exposure", "-0.3"))
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_depth = "16"
scene.render.film_transparent = False

bpy.ops.mesh.primitive_grid_add(x_subdivisions=grid, y_subdivisions=max(2, int(grid * height_m / width_m)), size=1.0)
canvas = bpy.context.active_object
canvas.name = "canvas"
canvas.scale = (width_m, height_m, 1.0)
canvas.location = (width_m * 0.5, height_m * 0.5, 0.0)
bpy.ops.object.transform_apply(scale=True, location=True)
bpy.ops.object.shade_smooth()

height_img = load_image(os.path.join(here, "height.png"), "Non-Color")
albedo_img = load_image(os.path.join(here, "albedo.png"), "Linear Rec.709" if "Linear Rec.709" in [c.name for c in bpy.types.ColorManagedInputColorspaceSettings.bl_rna.properties["name"].enum_items] else "Linear")
rough_img = load_image(os.path.join(here, "roughness.png"), "Non-Color")
mask_img = load_image(os.path.join(here, "mask.png"), "Non-Color")

tex = bpy.data.textures.new("height", type="IMAGE")
tex.image = height_img
disp = canvas.modifiers.new("relief", type="DISPLACE")
disp.texture = tex
disp.texture_coords = "UV"
disp.direction = "Z"
disp.mid_level = 0.0
disp.strength = height_scale

mat = bpy.data.materials.new("paint")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()
outn = nodes.new("ShaderNodeOutputMaterial")
bsdf = nodes.new("ShaderNodeBsdfPrincipled")
links.new(bsdf.outputs["BSDF"], outn.inputs["Surface"])
uv = nodes.new("ShaderNodeTexCoord")


def image_node(img, interp="Linear"):
    n = nodes.new("ShaderNodeTexImage")
    n.image = img
    n.interpolation = interp
    n.extension = "EXTEND"
    links.new(uv.outputs["UV"], n.inputs["Vector"])
    return n


n_albedo = image_node(albedo_img, "Cubic")
n_rough = image_node(rough_img)
n_mask = image_node(mask_img)
n_height = image_node(height_img, "Cubic")
links.new(n_albedo.outputs["Color"], bsdf.inputs["Base Color"])
links.new(n_rough.outputs["Color"], bsdf.inputs["Roughness"])
sep = nodes.new("ShaderNodeSeparateColor") if hasattr(bpy.types, "ShaderNodeSeparateColor") else nodes.new("ShaderNodeSeparateRGB")
links.new(n_mask.outputs["Color"], sep.inputs[0])
wet_out, angle_out, aniso_out = sep.outputs[0], sep.outputs[1], sep.outputs[2]

coat_w = find_input(bsdf, ["Coat Weight", "Clearcoat"])
if coat_w is not None:
    scale = nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = 0.55
    links.new(wet_out, scale.inputs[0])
    links.new(scale.outputs[0], coat_w)
set_input(bsdf, ["Coat Roughness", "Clearcoat Roughness"], 0.12)
set_input(bsdf, ["Coat IOR"], 1.5)
set_input(bsdf, ["IOR"], 1.48)
set_input(bsdf, ["Specular IOR Level", "Specular"], 0.5)
ani = find_input(bsdf, ["Anisotropic"])
if ani is not None:
    scale2 = nodes.new("ShaderNodeMath")
    scale2.operation = "MULTIPLY"
    scale2.inputs[1].default_value = 0.6
    links.new(aniso_out, scale2.inputs[0])
    links.new(scale2.outputs[0], ani)
rot = find_input(bsdf, ["Anisotropic Rotation"])
if rot is not None:
    links.new(angle_out, rot)
tangent = nodes.new("ShaderNodeTangent")
tangent.direction_type = "UV_MAP"
if "Tangent" in bsdf.inputs:
    links.new(tangent.outputs["Tangent"], bsdf.inputs["Tangent"])
ss = find_input(bsdf, ["Subsurface Weight", "Subsurface"])
if ss is not None:
    ss.default_value = 0.04
set_input(bsdf, ["Subsurface Radius"], (0.0008, 0.0005, 0.0003))
set_input(bsdf, ["Subsurface Scale"], 1.0)
bump = nodes.new("ShaderNodeBump")
bump.inputs["Strength"].default_value = 0.35
bump.inputs["Distance"].default_value = height_scale * 0.5
links.new(n_height.outputs["Color"], bump.inputs["Height"])
links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
canvas.data.materials.append(mat)

world = bpy.data.worlds.new("studio")
scene.world = world
world.use_nodes = True
wn = world.node_tree.nodes
wl = world.node_tree.links
wn.clear()
wout = wn.new("ShaderNodeOutputWorld")
bg = wn.new("ShaderNodeBackground")
wl.new(bg.outputs["Background"], wout.inputs["Surface"])
if not hdri:
    base = bpy.utils.system_resource("DATAFILES")
    for name in ("interior.exr", "studio.exr", "city.exr", "forest.exr"):
        p = os.path.join(base, "studiolights", "world", name)
        if os.path.exists(p):
            hdri = p
            break
if hdri and os.path.exists(hdri):
    env = wn.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(hdri)
    mapping = wn.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(float(arg("--hdri-rot", "40"))))
    coord = wn.new("ShaderNodeTexCoord")
    wl.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    wl.new(mapping.outputs["Vector"], env.inputs["Vector"])
    wl.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = float(arg("--hdri-strength", "0.5"))
else:
    bg.inputs["Color"].default_value = (0.55, 0.55, 0.58, 1.0)
    bg.inputs["Strength"].default_value = 1.0

bpy.ops.object.light_add(type="AREA", location=(width_m * 0.5 - 0.9, height_m * 0.5 + 0.5, 0.75))
key = bpy.context.active_object
key.data.energy = 70.0
key.data.size = 0.5
key.data.shape = "RECTANGLE"
key.data.size_y = 1.2
key.rotation_euler = (math.radians(-35.0), math.radians(-52.0), math.radians(-20.0))
bpy.ops.object.light_add(type="AREA", location=(width_m * 0.5 + 1.0, height_m * 0.5 - 0.3, 1.1))
fill = bpy.context.active_object
fill.data.energy = 20.0
fill.data.size = 1.5
fill.rotation_euler = (math.radians(20.0), math.radians(45.0), 0.0)

bpy.ops.object.camera_add()
cam = bpy.context.active_object
scene.camera = cam
if view == "top":
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = width_m * 1.02
    cam.location = (width_m * 0.5, height_m * 0.5, 2.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    scene.render.resolution_x = res
    scene.render.resolution_y = int(res * height_m / width_m)
else:
    at = arg("--at", "%f,%f" % (width_m * 0.55, height_m * 0.45)).split(",")
    cx = float(at[0])
    cy = float(at[1])
    dist = float(arg("--dist", "0.28"))
    cam.data.lens = 60.0
    cam.data.sensor_width = 36.0
    cam.data.dof.use_dof = True
    cam.data.dof.aperture_fstop = float(arg("--fstop", "11.0"))
    elev = math.radians(float(arg("--elev", "60")))
    azim = math.radians(float(arg("--azim", "-30")))
    cam.location = (cx + dist * math.cos(elev) * math.cos(azim), cy + dist * math.cos(elev) * math.sin(azim), dist * math.sin(elev))
    direction = (cx - cam.location[0], cy - cam.location[1], -cam.location[2])
    rot_quat = bpy.context.scene.cursor.rotation_quaternion.copy()
    import mathutils
    cam.rotation_euler = mathutils.Vector(direction).to_track_quat("-Z", "Y").to_euler()
    cam.data.dof.focus_distance = dist
    scene.render.resolution_x = res
    scene.render.resolution_y = int(res * 2 / 3)
scene.render.resolution_percentage = 100
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print("rendered", out)
