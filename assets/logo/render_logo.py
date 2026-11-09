####################################################################################################
#                                          render_logo.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 28/10/26                                                                                #
#                                                                                                  #
# Purpose: The MRSandbox logo (logo.png) as a real object, path-traced in Blender's Cycles on a    #
#          transparent film: a wide, low tray of black lacquer with a copper lip and inlay, filled #
#          with sand that rises into a spectrum's lines, NAA the tall one in the middle, creatine  #
#          and choline to its left, the smaller lines around. Each line is a pile shaped like a    #
#          Lorentzian with a flared foot, its ridges sweeping forward, left and right; the sand is #
#          made of                                                                                 #
#          grains, each its own shade and bump. Writes render-<LOGO_NAME>.png beside this script   #
#          (LOGO_SAMPLES and LOGO_SIZE set the quality).                                           #
#          Run: blender -b -P render_logo.py                                                       #
#                                                                                                  #
####################################################################################################
import math
import os

import bmesh
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = os.environ.get("LOGO_NAME", "sand")
SAMPLES = int(os.environ.get("LOGO_SAMPLES", "512"))
SIZE = int(os.environ.get("LOGO_SIZE", "2048"))
RADIUS, WALL, DEPTH = 1.35, 0.06, 0.26      # the tray's inner radius, wall and height
LEVEL = 0.21                                 # the flat sand's height in the tray
# the piles: (x, y, height, half width) on the tray, y away from the viewer
PILES = ((-1.13, 0.55, 0.28, 0.055), (-1.00, 0.30, 0.26, 0.055), (-0.80, 0.65, 0.26, 0.055),
         (-0.66, 0.15, 0.60, 0.070), (-0.12, 0.95, 0.32, 0.060), (0.11, 0.40, 1.36, 0.095),
         (0.40, 0.00, 0.20, 0.055), (0.59, -0.45, 0.45, 0.065), (0.92, 0.30, 0.35, 0.060))
FLARE, FOOT = 0.7, 5.0                       # the Lorentzian's power (below 1 the foot spreads)
                                             # and the foot's reach in half widths
RIDGE, RUN, SPREAD = 0.35, 0.8, 0.15         # a ridge's height (of its pile's), length, widening
LEAN, SWEEP = 0.35, 0.30                     # how a ridge drifts sideways as it comes forward
RIGHT = 0.5                                  # the right ridge's height against the left one's
GRAIN = 170.0                                # grains per unit length
RISE = 1.35                                  # the piles over the tray: the mark fills a square


#************#
#   linear   #
#************#
def linear(hexcolour):
    """An sRGB hex colour in Blender's linear space, with alpha."""
    def one(v):
        v = int(v, 16) / 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    h = hexcolour.lstrip("#")
    return (one(h[0:2]), one(h[2:4]), one(h[4:6]), 1.0)


#**************#
#   material   #
#**************#
def material(name, colour, roughness, metallic=0.0, coat=0.0):
    """A Principled material of one colour, with an optional clear coat."""
    m = bpy.data.materials.new(name)
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = linear(colour)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Coat Roughness"].default_value = 0.04
    return m


#**********#
#   sand   #
#**********#
def sand_material():
    """Sand as grains: a Voronoi cell per grain gives it its own shade and a small dome."""
    m = material("sand", "#B87A45", 0.95)
    nodes, links = m.node_tree.nodes, m.node_tree.links
    bsdf = nodes["Principled BSDF"]
    coords = nodes.new("ShaderNodeTexCoord")
    grains = nodes.new("ShaderNodeTexVoronoi")
    grains.inputs["Scale"].default_value = GRAIN
    links.new(coords.outputs["Object"], grains.inputs["Vector"])
    shade = nodes.new("ShaderNodeValToRGB")
    stops = shade.color_ramp.elements
    stops[0].position, stops[0].color = 0.0, linear("#2E1C0C")
    stops[1].position, stops[1].color = 1.0, linear("#D6A468")
    stops.new(0.35).color = linear("#7E4E24")
    stops.new(0.70).color = linear("#AC7038")
    links.new(grains.outputs["Color"], shade.inputs["Fac"])
    links.new(shade.outputs["Color"], bsdf.inputs["Base Color"])
    bump = nodes.new("ShaderNodeBump")
    bump.invert = True                                  # the cells' centres stand up
    bump.inputs["Strength"].default_value = 0.7
    bump.inputs["Distance"].default_value = 0.003
    links.new(grains.outputs["Distance"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return m


#*************#
#   surface   #
#*************#
def height(x, y):
    """The sand's height over the tray's floor at (x, y): the level, a pile per line (a
    Lorentzian raised to FLARE, its foot ending within FOOT half widths) and two ridges, which
    rise from the pile's foot and fade as they run forward, one drifting left, a lower one
    right, both widening; all settle to the level along the wall."""
    total = 0.0
    for x0, y0, a, w in PILES:
        near = ((x - x0) ** 2 + (y - y0) ** 2) / w ** 2
        total += a / (1.0 + near) ** FLARE * math.exp(-near / FOOT ** 2)
        ahead = y0 - y
        if ahead > 0:
            rise = a * RIDGE * (1 - math.exp(-ahead / w)) * math.exp(-ahead / RUN)
            for side, share in ((1, 1.0), (-1, RIGHT)):
                drift = side * (LEAN * ahead + SWEEP * ahead ** 2)
                total += share * rise * math.exp(-abs(x - x0 + drift) / (w + SPREAD * ahead))
    return LEVEL + RISE * total * (1.0 - (x * x + y * y) ** 4 / RADIUS ** 8)


def sand(steps=640):
    """The sand as a height field on a disc just inside the tray's wall."""
    bm = bmesh.new()
    inner = RADIUS - 0.004
    grid = {}
    for i in range(steps + 1):
        for j in range(steps + 1):
            x, y = -inner + 2 * inner * i / steps, -inner + 2 * inner * j / steps
            if x * x + y * y <= inner * inner:
                grid[i, j] = bm.verts.new((x, y, height(x, y)))
    for i in range(steps):
        for j in range(steps):
            quad = [grid.get(k) for k in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))]
            if all(quad):
                bm.faces.new(quad).smooth = True
    mesh = bpy.data.meshes.new("sand")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("sand", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(sand_material())


#**********#
#   tray   #
#**********#
def tray():
    """The tray: a lacquered dish (a cylinder with its inside cut away above a floor), bevelled,
    a copper lip on its rim and a copper inlay round its wall."""
    bpy.ops.mesh.primitive_cylinder_add(vertices=384, radius=RADIUS + WALL, depth=DEPTH,
                                        location=(0, 0, DEPTH / 2))
    dish = bpy.context.active_object
    bpy.ops.mesh.primitive_cylinder_add(vertices=384, radius=RADIUS, depth=DEPTH,
                                        location=(0, 0, DEPTH / 2 + 0.05))
    cut = bpy.context.active_object
    hollow = dish.modifiers.new("hollow", "BOOLEAN")
    hollow.operation, hollow.object = "DIFFERENCE", cut
    bpy.context.view_layer.objects.active = dish
    bpy.ops.object.modifier_apply(modifier="hollow")
    bpy.data.objects.remove(cut)
    soft = dish.modifiers.new("soft", "BEVEL")
    soft.width, soft.segments = 0.02, 6
    bpy.ops.object.shade_smooth()
    dish.data.materials.clear()                       # the boolean leaves an empty slot
    dish.data.materials.append(material("lacquer", "#120D0A", 0.35, metallic=0.2, coat=1.0))
    copper = material("copper", "#C8773F", 0.22, metallic=1.0)
    for radius, thickness, z in ((RADIUS + WALL / 2, WALL * 0.45, DEPTH),
                                 (RADIUS + WALL + 0.001, 0.007, 0.42 * DEPTH)):
        bpy.ops.mesh.primitive_torus_add(major_radius=radius, minor_radius=thickness,
                                         major_segments=384, minor_segments=16, location=(0, 0, z))
        bpy.ops.object.shade_smooth()
        bpy.context.active_object.data.materials.append(copper)


#***********#
#   light   #
#***********#
def area(name, location, target, size, power, colour):
    """An area light at `location` aimed at `target`."""
    data = bpy.data.lights.new(name, "AREA")
    data.size, data.energy, data.color = size, power, colour
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (Vector(target) - Vector(location)).to_track_quat("-Z", "Y").to_euler()


#***********#
#   scene   #
#***********#
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
sand()
tray()
# a warm key from the left, a warm rim from behind on the right, a faint cool fill
area("key", (-6.0, -0.5, 3.0), (0, 0, 0.4), 1.8, 2800, (1.0, 0.82, 0.62))
area("rim", (2.5, 4.5, 2.2), (0, 0, 0.6), 2.0, 1000, (1.0, 0.68, 0.42))
area("fill", (3.5, -5.0, 1.5), (0, 0, 0.3), 3.0, 40, (0.85, 0.90, 1.0))
# a dim studio that the lacquer and copper reflect; the film stays transparent
world = bpy.data.worlds.new("studio")
world.node_tree.nodes["Background"].inputs["Color"].default_value = linear("#6E625A")
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.10
scene.world = world
camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
bpy.context.collection.objects.link(camera)
camera.location = (0.0, -8.6, 2.3)
camera.rotation_euler = (Vector((0, 0, 0.6)) - camera.location).to_track_quat("-Z", "Y").to_euler()
camera.data.lens = 95
scene.camera = camera
scene.render.engine = "CYCLES"
try:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "METAL"
    preferences.get_devices()
    for device in preferences.devices:
        device.use = True
    scene.cycles.device = "GPU"
except Exception:          # no GPU: the CPU renders the same image, slower
    scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.render.film_transparent = True
scene.render.resolution_x = scene.render.resolution_y = SIZE
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "Standard"
scene.render.filepath = os.path.join(HERE, f"render-{NAME}.png")
bpy.ops.render.render(write_still=True)
print("wrote", scene.render.filepath, flush=True)
