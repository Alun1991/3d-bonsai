"""
SAKURA BONSAI — Blender build pipeline
======================================
Prunus incisa style cherry bonsai: gnarled trunk, zigzag ramification,
5-petal blossom clusters forming cloud pads, moss mound with embedded rocks,
celadon bowl. Everything procedural, exported as GLB for the Three.js viewer.

Run headless:
    Blender -b -P blender/build_bonsai.py
"""

import bpy
import bmesh
import math
import os
import random
import numpy as np
from mathutils import Vector, Matrix, noise

SEED = 20261114
rng = random.Random(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")
TEX = os.path.join(ASSETS, "tex")
os.makedirs(TEX, exist_ok=True)

TEX_RES = 1024

# ----------------------------------------------------------------------------
# scene reset
# ----------------------------------------------------------------------------

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def log(*a):
    print("[bonsai]", *a, flush=True)


# ----------------------------------------------------------------------------
# noise helpers (numpy, tileable)
# ----------------------------------------------------------------------------


def vnoise_tile(h, w, fx, fy, seed):
    g = np.random.default_rng(seed).random((fy + 1, fx + 1))
    g[-1, :] = g[0, :]
    g[:, -1] = g[:, 0]
    ys = np.linspace(0, fy, h, endpoint=False)
    xs = np.linspace(0, fx, w, endpoint=False)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy_t = ys - y0
    fx_t = xs - x0
    fy_t = fy_t * fy_t * (3 - 2 * fy_t)
    fx_t = fx_t * fx_t * (3 - 2 * fx_t)
    a = g[np.ix_(y0, x0)]
    b = g[np.ix_(y0, x0 + 1)]
    c = g[np.ix_(y0 + 1, x0)]
    d = g[np.ix_(y0 + 1, x0 + 1)]
    fyv = fy_t[:, None]
    fxv = fx_t[None, :]
    top = a * (1 - fxv) + b * fxv
    bot = c * (1 - fxv) + d * fxv
    return top * (1 - fyv) + bot * fyv


def fbm(h, w, fx, fy, octaves=5, seed=0, gain=0.5):
    total = np.zeros((h, w))
    amp = 1.0
    norm = 0.0
    for i in range(octaves):
        total += amp * vnoise_tile(h, w, max(1, int(fx * 2 ** i)), max(1, int(fy * 2 ** i)), seed + i * 131)
        norm += amp
        amp *= gain
    return total / norm


def normal_from_height(height, strength=2.0):
    gy, gx = np.gradient(height)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(height)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack(
        [(nx / ln * 0.5 + 0.5), (ny / ln * 0.5 + 0.5), (nz / ln * 0.5 + 0.5)], axis=-1
    )


def write_image(name, rgb, srgb=True):
    """rgb: (h, w, 3) float 0..1, row 0 = top"""
    h, w, _ = rgb.shape
    img = bpy.data.images.get(name)
    if img:
        bpy.data.images.remove(img)
    img = bpy.data.images.new(name, width=w, height=h, alpha=False)
    img.colorspace_settings.name = "sRGB" if srgb else "Non-Color"
    flat = np.concatenate([np.flipud(rgb), np.ones((h, w, 1))], axis=2).astype(np.float32)
    img.pixels.foreach_set(flat.ravel())
    path = os.path.join(TEX, name + ".png")
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    log("wrote", path)
    return img


# ----------------------------------------------------------------------------
# textures
# ----------------------------------------------------------------------------


def tex_bark():
    h = w = TEX_RES
    n = fbm(h, w, 26, 3, octaves=5, seed=11)
    fine = fbm(h, w, 120, 14, octaves=3, seed=27)
    fiss = np.clip(1.0 - fbm(h, w, 16, 2, octaves=4, seed=41) ** 0.5, 0, 1)
    height = np.clip(0.60 * n + 0.40 * fine - 0.45 * fiss, 0, 1)

    # dark grey-brown, like aged prunus bark
    vb = 0.16 + 0.26 * height
    base = np.stack([vb * 1.05, vb * 0.88, vb * 0.76], axis=-1)
    grey = fbm(h, w, 60, 30, octaves=3, seed=55)
    base *= (0.70 + 0.5 * grey)[:, :, None]
    base *= np.stack([1.0 - 0.55 * fiss, 1.0 - 0.42 * fiss, 1.0 - 0.55 * fiss], axis=-1)

    # pale lichen patches, like the reference trunk
    lichen = np.clip((fbm(h, w, 10, 10, octaves=4, seed=61) - 0.56) * 2.6, 0, 1)
    lichen *= np.clip((fbm(h, w, 90, 90, octaves=2, seed=63) - 0.35) * 2.0, 0, 1)
    lc = np.array([0.46, 0.47, 0.40])
    base = base * (1 - 0.5 * lichen[:, :, None]) + lc * (0.5 * lichen[:, :, None])

    return write_image("bark_base", np.clip(base, 0, 1)), write_image(
        "bark_normal", normal_from_height(height, 6.0), srgb=False
    )


def tex_stone():
    h = w = TEX_RES
    n = fbm(h, w, 7, 7, octaves=6, seed=71)
    speck = fbm(h, w, 150, 150, octaves=2, seed=83)
    height = np.clip(0.75 * n + 0.25 * speck, 0, 1)
    v = 0.21 + 0.24 * height
    base = np.stack([v * 1.00, v * 0.98, v * 0.92], axis=-1)
    base *= (0.78 + 0.34 * speck)[:, :, None]
    return write_image("stone_base", np.clip(base, 0, 1)), write_image(
        "stone_normal", normal_from_height(height, 2.6), srgb=False
    )


def tex_soil():
    h = w = 512
    n = fbm(h, w, 30, 30, octaves=5, seed=97)
    gran = fbm(h, w, 90, 90, octaves=2, seed=99)
    v = 0.30 + 0.45 * gran
    base = np.stack([v * 0.52 + 0.10, v * 0.40 + 0.06, v * 0.28 + 0.04], axis=-1) * (0.6 + 0.7 * n)[:, :, None]
    return write_image("soil_base", np.clip(base, 0, 1)), write_image(
        "soil_normal", normal_from_height(np.clip(0.6 * n + 0.4 * gran, 0, 1), 3.0), srgb=False
    )


def tex_moss():
    h = w = TEX_RES
    clump = fbm(h, w, 34, 34, octaves=5, seed=113)
    fine = fbm(h, w, 220, 220, octaves=2, seed=127)
    height = np.clip(0.55 * clump + 0.45 * fine, 0, 1)
    # rich living moss: two populations, yellow-green and deep green
    g = 0.14 + 0.34 * height
    dark = np.array([0.045, 0.10, 0.022])
    lite = np.array([0.155, 0.255, 0.070])
    mixm = np.clip(fbm(h, w, 18, 18, octaves=3, seed=131) * 1.4 - 0.25, 0, 1)[:, :, None]
    base = dark * (1 - mixm) + lite * mixm
    base *= (0.6 + 0.7 * g)[:, :, None]
    # occasional soil showing through
    soilmask = np.clip((fbm(h, w, 7, 7, octaves=4, seed=137) - 0.60) * 3.0, 0, 1)[:, :, None]
    base = base * (1 - 0.6 * soilmask) + np.array([0.16, 0.11, 0.07]) * (0.6 * soilmask)
    return write_image("moss_base", np.clip(base, 0, 1)), write_image(
        "moss_normal", normal_from_height(height, 3.4), srgb=False
    )


def tex_clay():
    h = w = TEX_RES
    n = fbm(h, w, 8, 3, octaves=5, seed=149)
    grain = fbm(h, w, 200, 200, octaves=2, seed=151)
    height = np.clip(0.6 * n + 0.4 * grain, 0, 1)
    # celadon glaze with darker pooling
    base = np.stack(
        [0.42 + 0.07 * height, 0.50 + 0.07 * height, 0.44 + 0.06 * height], axis=-1
    )
    glaze = np.clip(fbm(h, w, 4, 2, octaves=4, seed=163) * 1.3, 0, 1)
    cel = np.array([0.30, 0.39, 0.33])
    base = base * (1 - 0.6 * glaze[:, :, None]) + cel * (0.6 * glaze[:, :, None])
    # fine crazing
    cr = np.clip(1.0 - np.abs(fbm(h, w, 60, 60, octaves=3, seed=167) - 0.5) * 40, 0, 1)
    base *= (1 - 0.22 * cr)[:, :, None]
    return write_image("clay_base", np.clip(base, 0, 1)), write_image(
        "clay_normal", normal_from_height(height, 1.0), srgb=False
    )


log("generating textures …")
IMG_BARK, NRM_BARK = tex_bark()
IMG_STONE, NRM_STONE = tex_stone()
IMG_SOIL, NRM_SOIL = tex_soil()
IMG_MOSS, NRM_MOSS = tex_moss()
IMG_CLAY, NRM_CLAY = tex_clay()


# ----------------------------------------------------------------------------
# materials
# ----------------------------------------------------------------------------


def pbr_material(name, base_img, normal_img, roughness=0.85, normal_strength=1.0, vcol=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0

    tb = nt.nodes.new("ShaderNodeTexImage")
    tb.image = base_img
    tn = nt.nodes.new("ShaderNodeTexImage")
    tn.image = normal_img
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = normal_strength
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"

    nt.links.new(uv.outputs["UV"], tb.inputs["Vector"])
    nt.links.new(uv.outputs["UV"], tn.inputs["Vector"])
    nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(tb.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def vcol_material(name, roughness=0.6, backface=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = not backface
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    ca = nt.nodes.new("ShaderNodeVertexColor")
    ca.layer_name = "Col"
    nt.links.new(ca.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


MAT_BARK = pbr_material("Bark", IMG_BARK, NRM_BARK, 0.92, 1.3)
MAT_STONE = pbr_material("Stone", IMG_STONE, NRM_STONE, 0.78, 1.0)
MAT_SOIL = pbr_material("Soil", IMG_SOIL, NRM_SOIL, 1.0, 0.7)
MAT_MOSS = pbr_material("Moss", IMG_MOSS, NRM_MOSS, 1.0, 1.0)
MAT_CLAY = pbr_material("Clay", IMG_CLAY, NRM_CLAY, 0.30, 0.4)
MAT_BLOSSOM = vcol_material("Blossom", 0.55, backface=True)
MAT_MOSSTUFT = vcol_material("MossTuft", 0.9, backface=True)
MAT_CORE = vcol_material("Core", 1.0, backface=False)


# ----------------------------------------------------------------------------
# mesh helpers
# ----------------------------------------------------------------------------


def new_object(name, verts, faces, uvs=None, mat=None, smooth=True, recalc=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.validate()
    if uvs is not None:
        uvl = me.uv_layers.new(name="UVMap")
        data = uvl.data
        for poly in me.polygons:
            for li in poly.loop_indices:
                vi = me.loops[li].vertex_index
                data[li].uv = uvs[vi]
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    if recalc:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
    if mat:
        ob.data.materials.append(mat)
    return ob


def parallel_frames(pts):
    n = len(pts)
    tang = []
    for i in range(n):
        if i == 0:
            t = pts[1] - pts[0]
        elif i == n - 1:
            t = pts[-1] - pts[-2]
        else:
            t = pts[i + 1] - pts[i - 1]
        tang.append(t.normalized())
    ref = Vector((0, 0.15, 1))
    if abs(tang[0].dot(ref)) > 0.9:
        ref = Vector((1, 0, 0))
    nrm = (ref - tang[0] * ref.dot(tang[0])).normalized()
    norms = [nrm]
    for i in range(1, n):
        a, b = tang[i - 1], tang[i]
        axis = a.cross(b)
        if axis.length > 1e-9:
            ang = math.atan2(axis.length, a.dot(b))
            nrm = (Matrix.Rotation(ang, 4, axis.normalized()) @ nrm).normalized()
        norms.append(nrm)
    return tang, norms


class MeshBuilder:
    def __init__(self):
        self.verts = []
        self.faces = []
        self.uvs = []

    def tube(self, pts, radii, ring=10, v_scale=3.0, u_scale=1.0, ridge=0.10, ridge_seed=0.0):
        tang, norms = parallel_frames(pts)
        base = len(self.verts)
        n = len(pts)
        for i in range(n):
            p = pts[i]
            r0 = radii[i]
            t = tang[i]
            nr = norms[i]
            bn = t.cross(nr).normalized()
            for j in range(ring):
                a = 2 * math.pi * j / ring
                ca, sa = math.cos(a), math.sin(a)
                rr = r0 * (1.0 + ridge * noise.noise(Vector((ca * 2.6 + ridge_seed, i * 0.22, sa * 2.6))))
                v = p + nr * (ca * rr) + bn * (sa * rr)
                self.verts.append(v)
                self.uvs.append((j / ring * u_scale, i / (n - 1) * v_scale))
        for i in range(n - 1):
            for j in range(ring):
                j2 = (j + 1) % ring
                self.faces.append(
                    (
                        base + i * ring + j,
                        base + i * ring + j2,
                        base + (i + 1) * ring + j2,
                        base + (i + 1) * ring + j,
                    )
                )
        for end, idx in ((0, base), (-1, base + (n - 1) * ring)):
            p = pts[end]
            ci = len(self.verts)
            self.verts.append(p)
            self.uvs.append((0.5 * u_scale, (0.0 if end == 0 else 1.0) * v_scale))
            for j in range(ring):
                j2 = (j + 1) % ring
                a0 = idx + j
                a1 = idx + j2
                if end == 0:
                    self.faces.append((ci, a1, a0))
                else:
                    self.faces.append((ci, a0, a1))
        return self


def sphere_uv(positions):
    uvs = []
    for v in positions:
        n = v.normalized()
        u = (math.atan2(n.z, n.x) / (2 * math.pi)) + 0.5
        vv = math.acos(max(-1.0, min(1.0, n.y))) / math.pi
        uvs.append((u * 2.0, vv * 2.0))
    return uvs


def catmull(pts, samples):
    out = []
    n = len(pts)
    for i in range(samples):
        t = i / (samples - 1) * (n - 1)
        seg = min(int(t), n - 2)
        f = t - seg
        p0 = pts[max(0, seg - 1)]
        p1 = pts[seg]
        p2 = pts[seg + 1]
        p3 = pts[min(n - 1, seg + 2)]
        out.append(
            0.5
            * (
                (2 * p1)
                + (-p0 + p2) * f
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f
                + (-p0 + 3 * p1 - 3 * p2 + p3) * f * f * f
            )
        )
    return out


# ----------------------------------------------------------------------------
# skeleton — metres, tree ~50 cm
# ----------------------------------------------------------------------------

S = 0.19
TREE_Y = 0.110
R_S = 0.0955
H_S = 0.21


def V(x, y, z):
    return Vector((x * S, (y - 0.78) * S + TREE_Y, z * S))


trunk_pts = [
    V(0.00, 0.78, 0.10),
    V(0.16, 1.02, 0.06),
    V(0.30, 1.30, -0.01),
    V(0.14, 1.58, -0.06),
    V(-0.12, 1.86, -0.05),
    V(-0.16, 2.14, 0.01),
    V(0.02, 2.40, 0.05),
    V(0.10, 2.62, 0.03),
    V(0.05, 2.80, 0.00),
]

N_TRUNK = 150
trunk_samples = catmull(trunk_pts, N_TRUNK)


def trunk_radius(u):
    base = 0.245 * (1 - u) ** 0.80 + 0.014 * u
    flare = 1 + 0.50 * math.exp(-u * 8.5)
    wobble = 1 + 0.05 * noise.noise(Vector((u * 9.0, 3.3, 7.7)))
    return base * flare * wobble * S


def trunk_info_at_y(world_y):
    best = 0
    bestd = 1e9
    for i, p in enumerate(trunk_samples):
        d = abs(p.y - world_y)
        if d < bestd:
            bestd = d
            best = i
    return best / (N_TRUNK - 1), trunk_samples[best].copy()


bark = MeshBuilder()

trunk_radii = [trunk_radius(i / (N_TRUNK - 1)) for i in range(N_TRUNK)]
bark.tube(trunk_samples, trunk_radii, ring=20, v_scale=6.0, u_scale=1.5, ridge=0.14, ridge_seed=1.7)
log("trunk built")

# ---- pads: the cloud platforms the tree is built around --------------------

pads = [
    (0.02, 2.84, 0.00, 0.36),
    (-0.46, 2.58, 0.04, 0.33),
    (0.52, 2.38, 0.10, 0.34),
    (-0.72, 2.12, 0.10, 0.33),
    (0.78, 1.98, 0.14, 0.34),
    (-0.52, 1.68, 0.12, 0.28),
    (0.52, 1.54, 0.30, 0.26),
]

branch_paths = []
for (px_, py_, pz_, pr_) in pads:
    u, anchor = trunk_info_at_y(V(px_, py_, pz_).y - 0.02)
    r_anchor = trunk_radius(u)
    target = V(px_, py_ - pr_ * 0.25, pz_)
    mid = (anchor + target) * 0.5
    mid.y += 0.05 * S + (target - anchor).length * 0.12  # gentle upward arc
    pts = [anchor, mid, target]
    full = catmull(pts, 44)
    radii = []
    for i in range(len(full)):
        t = i / (len(full) - 1)
        grow = min(1.0, t / 0.15)
        ramp = grow * grow * (3 - 2 * grow)
        radii.append(max(0.0005, (r_anchor * 0.55 * (1 - t) ** 0.6 + 0.0022) * ramp))
    bark.tube(full, radii, ring=12, v_scale=4.0, ridge=0.11, ridge_seed=3.1 + pr_)
    branch_paths.append((full, radii))
log("branches built")

# ---- nebari roots -----------------------------------------------------------

root_specs = [
    [[0.00, 0.92, 0.10], [0.30, 0.86, 0.34], [0.52, 0.80, 0.52]],
    [[0.00, 0.92, 0.09], [-0.32, 0.87, 0.36], [-0.56, 0.80, 0.54]],
    [[0.02, 0.92, 0.08], [0.22, 0.86, -0.28], [0.38, 0.80, -0.52]],
    [[0.00, 0.92, 0.08], [-0.28, 0.87, -0.30], [-0.50, 0.80, -0.50]],
    [[0.01, 0.92, 0.10], [0.48, 0.87, 0.02], [0.74, 0.80, -0.10]],
    [[0.00, 0.92, 0.08], [-0.16, 0.90, 0.44], [-0.24, 0.84, 0.72]],
]
for pts_units in root_specs:
    pts = [V(*p) for p in pts_units]
    full = catmull(pts, 40)
    radii = []
    for i in range(len(full)):
        t = i / (len(full) - 1)
        grow = min(1.0, t / 0.18)
        ramp = grow * grow * (3 - 2 * grow)
        radii.append(max(0.0004, (0.095 * S * (1 - t) ** 0.58 + 0.0014) * ramp))
    bark.tube(full, radii, ring=12, v_scale=3.0, ridge=0.20, ridge_seed=5.5)
log("roots built")

# ---- zigzag twigs around pad cores -----------------------------------------

twig_rng = random.Random(4242)
anchors = []
for full, radii in branch_paths:
    for k in range(6):
        t = 0.35 + k / 6 * 0.62
        i = min(len(full) - 1, int(t * (len(full) - 1)))
        anchors.append((full[i].copy(), radii[i], (full[min(len(full) - 1, i + 1)] - full[i])))


def grow_twig(origin, direction, length, radius, depth):
    pts = [origin.copy()]
    d = direction.normalized()
    steps = twig_rng.randint(2, 3)
    cur = origin.copy()
    rad = radius
    for i in range(steps):
        axis = Vector(
            (twig_rng.random() - 0.5, twig_rng.random() - 0.5, twig_rng.random() - 0.5)
        ).normalized()
        ang = (1 if twig_rng.random() < 0.5 else -1) * (0.55 + twig_rng.random() * 0.6)
        d = (Matrix.Rotation(ang, 4, axis) @ d).normalized()
        d = (d + Vector((0, 0.10, 0))).normalized()
        seg = length / steps * (1 - i * 0.1)
        cur = cur + d * seg
        pts.append(cur.copy())
        rad *= 0.62
        if depth < 1 and twig_rng.random() < 0.75:
            cd = d.copy()
            axis2 = Vector(
                (twig_rng.random() - 0.5, twig_rng.random() - 0.5, twig_rng.random() - 0.5)
            ).normalized()
            cd = (Matrix.Rotation(
                (1 if twig_rng.random() < 0.5 else -1) * (0.6 + twig_rng.random() * 0.5), 4, axis2
            ) @ cd).normalized()
            grow_twig(cur, cd, length * 0.6, rad * 1.1, depth + 1)
    full = catmull(pts, 12)
    radii = []
    for i in range(len(full)):
        t = i / (len(full) - 1)
        grow = min(1.0, t / 0.3)
        ramp = grow * grow * (3 - 2 * grow)
        radii.append(max(0.0003, (radius * (1 - t) + 0.0005) * ramp))
    bark.tube(full, radii, ring=6, v_scale=2.0, ridge=0.10, ridge_seed=7.0)


for (p, r, tan) in anchors:
    if twig_rng.random() < 0.7:
        d = tan.normalized()
        d = (d + Vector((twig_rng.uniform(-0.8, 0.8), 0.55 + twig_rng.random() * 0.5, twig_rng.uniform(-0.8, 0.8)))).normalized()
        grow_twig(p, d, 0.018 + twig_rng.random() * 0.02, max(0.0006, r * 0.5), 0)
log("twigs built")

bark_obj = new_object("Bark", bark.verts, bark.faces, bark.uvs, MAT_BARK)
log("bark object:", len(bark.verts), "verts", len(bark.faces), "faces")


# ----------------------------------------------------------------------------
# blossoms — real 5-petal flowers clustered into cloud pads
# ----------------------------------------------------------------------------


def build_flower(hue_t, vertex_jitter):
    """returns verts, faces, per-vertex tint multipliers for one flower"""
    verts = []
    faces = []
    tints = []
    n_petals = 5
    cup = 0.22
    # petal profile rows: (radial, half-width, lift)
    prof = [(0.10, 0.06, 0.0), (0.45, 0.25, cup), (0.82, 0.22, cup * 2.1), (0.96, 0.10, cup * 3.0)]
    for pi in range(n_petals):
        ang = pi * math.tau / n_petals + (hue_t - 0.5) * 0.25
        ca, sa = math.cos(ang), math.sin(ang)
        u = Vector((ca, sa, 0.0))
        w = Vector((-sa, ca, 0.0))
        idxmap = []
        for (rd, hw, lift) in prof:
            hw_j = hw * (0.85 + 0.3 * vertex_jitter())
            lift_j = lift * (0.8 + 0.4 * vertex_jitter())
            L = u * rd + w * hw_j + Vector((0, 0, lift_j))
            R = u * rd - w * hw_j + Vector((0, 0, lift_j))
            iL = len(verts)
            verts.append(L)
            tints.append(0.94 + 0.1 * vertex_jitter())
            iR = len(verts)
            verts.append(R)
            tints.append(0.94 + 0.1 * vertex_jitter())
            idxmap.append((iL, iR))
        tip = u * 1.0 + Vector((0, 0, cup * 3.4))
        itip = len(verts)
        verts.append(tip)
        tints.append(1.02)
        for a in range(len(idxmap) - 1):
            L0, R0 = idxmap[a]
            L1, R1 = idxmap[a + 1]
            faces.append((L0, R0, R1))
            faces.append((L0, R1, L1))
        L0, R0 = idxmap[-1]
        faces.append((L0, R0, itip))
    return verts, faces, tints


flower_rng = random.Random(9091)
blossoms = MeshBuilder()
blossom_cols = []


def place_flower(pos, normal, scale, hue_t, open_amt):
    verts, faces, tints = build_flower(hue_t, lambda: flower_rng.random())
    # orient +Z -> normal
    z = Vector((0, 0, 1))
    n = normal.normalized()
    axis = z.cross(n)
    if axis.length < 1e-6:
        mtx = Matrix.Identity(4) if n.z > 0 else Matrix.Rotation(math.pi, 4, Vector((1, 0, 0)))
    else:
        ang = math.atan2(axis.length, z.dot(n))
        mtx = Matrix.Rotation(ang, 4, axis.normalized())
    # openness: flatten petals when fully open
    sx = scale * (0.75 + 0.5 * open_amt)
    base = len(blossoms.verts)
    for v, tint in zip(verts, tints):
        wv = mtx @ Vector((v.x * sx, v.y * sx, v.z * scale * (1.2 - 0.5 * open_amt)))
        blossoms.verts.append(pos + wv)
        blossoms.uvs.append((0.5, 0.5))
        # white -> pale pink by hue_t, deeper at the base rows
        r = 0.99 - 0.02 * hue_t
        g = 0.86 - 0.44 * hue_t
        b = 0.855 - 0.26 * hue_t
        blossom_cols.append((r * tint, g * tint, b * tint))
    for f in faces:
        blossoms.faces.append(tuple(i + base for i in f))


# dark cores so pads are never see-through
core_verts = []
core_faces = []
core_cols = []
core_rng = random.Random(1213)


def add_core(center, sx, sy, sz):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
    bm.verts.index_update()
    local = [Vector((v.co.x * sx, v.co.y * sy, v.co.z * sz)) for v in bm.verts]
    faces = [tuple(vi.index for vi in f.verts) for f in bm.faces]
    bm.free()
    base = len(core_verts)
    for v in local:
        core_verts.append(center + v)
        shade = 0.8 + 0.4 * core_rng.random()
        core_cols.append((0.16 * shade, 0.09 * shade, 0.10 * shade))
    for f in faces:
        core_faces.append(tuple(i + base for i in f))


flower_count = 0
for (px_, py_, pz_, pr_) in pads:
    center = V(px_, py_, pz_)
    R = pr_ * S
    # core mass
    add_core(center, R * 0.46, R * 0.18, R * 0.40)
    add_core(center + Vector((R * 0.3, -R * 0.04, R * 0.12)), R * 0.32, R * 0.14, R * 0.28)
    add_core(center + Vector((-R * 0.28, R * 0.01, -R * 0.1)), R * 0.32, R * 0.14, R * 0.3)

    n_flowers = max(190, int(pr_ * 640))
    for _ in range(n_flowers):
        # biased to the upper shell so pads read from above
        a = flower_rng.random() * math.tau
        b = math.acos(2 * flower_rng.random() - 1)
        shell = 0.50 + flower_rng.random() * 0.58
        px = center.x + math.sin(b) * math.cos(a) * R * 1.05 * shell
        py = center.y + math.cos(b) * R * 0.30 * shell
        pz = center.z + math.sin(b) * math.sin(a) * R * 0.90 * shell
        outward = Vector(
            (math.sin(b) * math.cos(a), max(0.15, math.cos(b)) * 0.9, math.sin(b) * math.sin(a))
        ).normalized()
        outward = (outward + Vector((
            flower_rng.uniform(-0.4, 0.4), flower_rng.uniform(-0.25, 0.4), flower_rng.uniform(-0.4, 0.4)
        ))).normalized()
        scale = 0.0105 + flower_rng.random() * 0.006
        place_flower(Vector((px, py, pz)), outward, scale,
                     flower_rng.random(), flower_rng.random())
        flower_count += 1

# a few % buds: small pink spheres among the flowers
bud_rng = random.Random(777)
for (px_, py_, pz_, pr_) in pads:
    center = V(px_, py_, pz_)
    R = pr_ * S
    for _ in range(int(pr_ * 26)):
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
        bm.verts.index_update()
        s = 0.0028 + bud_rng.random() * 0.0022
        pos = center + Vector((bud_rng.uniform(-R, R), bud_rng.uniform(-R * 0.25, R * 0.35), bud_rng.uniform(-R, R) * 0.85))
        base = len(blossoms.verts)
        for v in bm.verts:
            blossoms.verts.append(pos + Vector((v.co.x, v.co.y * 1.35, v.co.z)) * s)
            blossoms.uvs.append((0.5, 0.5))
            blossom_cols.append((0.86, 0.42, 0.55))
        bm.verts.index_update()
        for f in bm.faces:
            blossoms.faces.append(tuple(vi.index + base for vi in f.verts))
        bm.free()

log("blossoms:", flower_count)

blossom_obj = new_object("Blossoms", blossoms.verts, blossoms.faces, blossoms.uvs, MAT_BLOSSOM, smooth=False, recalc=False)
col_layer = blossom_obj.data.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
for i, c in enumerate(blossom_cols):
    col_layer.data[i].color = (*c, 1.0)

core_obj = new_object("PadCores", core_verts, core_faces, None, MAT_CORE, smooth=True, recalc=True)
core_col = core_obj.data.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
for i, c in enumerate(core_cols):
    core_col.data[i].color = (*c, 1.0)


# ----------------------------------------------------------------------------
# pot (shallow celadon bowl)
# ----------------------------------------------------------------------------

profile = [
    (0.00, 0.030), (0.55, 0.018), (0.95, 0.0), (1.25, 0.020), (1.42, 0.068),
    (1.54, 0.135), (1.60, 0.205), (1.625, 0.262), (1.635, 0.290), (1.615, 0.305),
    (1.578, 0.296), (1.52, 0.240), (1.38, 0.215), (1.10, 0.205), (0.60, 0.198),
    (0.00, 0.192),
]
pot_verts = []
pot_faces = []
pot_uvs = []
RING = 96
for (r, y) in profile:
    for j in range(RING):
        a = math.tau * j / RING
        pot_verts.append(Vector((math.cos(a) * r * R_S, y * H_S, math.sin(a) * r * R_S)))
        pot_uvs.append((j / RING * 3.0, y * 1.4))
for i in range(len(profile) - 1):
    for j in range(RING):
        j2 = (j + 1) % RING
        pot_faces.append((i * RING + j, i * RING + j2, (i + 1) * RING + j2, (i + 1) * RING + j))
pot = new_object("Pot", pot_verts, pot_faces, pot_uvs, MAT_CLAY)

# ---- soil disc --------------------------------------------------------------
SOIL_Y = 0.20 * H_S
soil_verts = [Vector((0, SOIL_Y, 0))]
soil_uvs = [(0.5, 0.5)]
soil_faces = []
SOIL_R = 1.56 * R_S
for j in range(RING):
    a = math.tau * j / RING
    soil_verts.append(Vector((math.cos(a) * SOIL_R, SOIL_Y, math.sin(a) * SOIL_R)))
    soil_uvs.append((math.cos(a) * 1.5, math.sin(a) * 1.5))
for j in range(RING):
    soil_faces.append((0, 1 + (j + 1) % RING, 1 + j))
soil = new_object("Soil", soil_verts, soil_faces, soil_uvs, MAT_SOIL)


# ----------------------------------------------------------------------------
# moss mound (flattened dome, irregular)
# ----------------------------------------------------------------------------

MOUND_R = 1.30 * R_S
MOUND_SY = 0.60
MOUND_CY = SOIL_Y

# keep tree base in sync with this mound
TREE_DELTA = (MOUND_CY + MOUND_R * MOUND_SY) - TREE_Y
for name in ("bark",):
    pass

mound_verts = []
mound_faces = []
mound_uvs = []
LAT, LON = 44, 88
for i in range(LAT + 1):
    theta = math.pi * i / LAT
    for j in range(LON):
        phi = math.tau * j / LON
        nx = math.sin(theta) * math.cos(phi)
        ny = math.cos(theta)
        nz = math.sin(theta) * math.sin(phi)
        n = noise.noise(Vector((nx * 2.2 + 11, ny * 2.2 + 7, nz * 2.2 + 3)))
        disp = 1 + (n - 0.5) * 0.13
        mound_verts.append(
            Vector((nx * MOUND_R * disp, ny * MOUND_R * MOUND_SY * disp + MOUND_CY, nz * MOUND_R * disp))
        )
        if ny < -0.4:
            mound_verts[-1].y = max(mound_verts[-1].y, SOIL_Y - 0.004)
        mound_uvs.append((j / LON * 4.0, i / LAT * 2.0))
for i in range(LAT):
    for j in range(LON):
        j2 = (j + 1) % LON
        mound_faces.append((i * LON + j, i * LON + j2, (i + 1) * LON + j2, (i + 1) * LON + j))
mound = new_object("Mound", mound_verts, mound_faces, mound_uvs, MAT_MOSS)


def mound_surface_y(x, z):
    r2 = (x * x + z * z) / (MOUND_R * MOUND_R)
    if r2 >= 1:
        return SOIL_Y
    ny = math.sqrt(1 - r2)
    for _ in range(2):
        n = noise.noise(Vector((x / MOUND_R * 2.2 + 11, ny * 2.2 + 7, z / MOUND_R * 2.2 + 3)))
        disp = 1 + (n - 0.5) * 0.13
        nx = x / (MOUND_R * disp)
        nz = z / (MOUND_R * disp)
        ny = math.sqrt(max(0.0, 1 - nx * nx - nz * nz))
    n = noise.noise(Vector((x / MOUND_R * 2.2 + 11, ny * 2.2 + 7, z / MOUND_R * 2.2 + 3)))
    disp = 1 + (n - 0.5) * 0.22
    return MOUND_CY + MOUND_R * MOUND_SY * ny * disp


# tree base must sit on the mound top
_tree_lift = (MOUND_CY + MOUND_R * MOUND_SY * 0.96) - V(0.0, 0.92, 0.10).y
for ob in [bark_obj]:
    for v in ob.data.vertices:
        v.co.y += _tree_lift
for coll in (blossoms,):
    pass
for v in blossoms_verts_placeholder if False else []:
    pass
log("tree lifted by", round(_tree_lift, 4))

# NOTE: blossoms/cores were placed in pre-lift coordinates; lift them too
for ob in [core_obj, blossom_obj]:
    for v in ob.data.vertices:
        v.co.y += _tree_lift

# ---- moss tufts hugging the mound ------------------------------------------

moss_verts = []
moss_faces = []
moss_uvs = []
mrng = random.Random(3131)
for _ in range(2100):
    a = mrng.random() * math.tau
    r = MOUND_R * (mrng.random() ** 0.60) * 0.94
    x, z = math.cos(a) * r, math.sin(a) * r
    y = mound_surface_y(x, z) - 0.0015
    origin = Vector((x, y, z))
    nrm = Vector((
        x / (MOUND_R ** 2),
        (y - MOUND_CY) / ((MOUND_R * MOUND_SY) ** 2) + 1e-6,
        z / (MOUND_R ** 2),
    ))
    if nrm.length < 1e-6:
        nrm = Vector((0, 1, 0))
    nrm.normalize()
    ref = Vector((0, 1, 0)) if abs(nrm.y) < 0.95 else Vector((1, 0, 0))
    t1 = nrm.cross(ref)
    t1.normalize()
    t2 = nrm.cross(t1)
    t2.normalize()
    for k in range(mrng.randint(5, 9)):
        ang = mrng.random() * math.tau
        tilt = mrng.uniform(0.15, 0.7)
        d = (nrm * math.cos(tilt) + (t1 * math.cos(ang) + t2 * math.sin(ang)) * math.sin(tilt))
        d.normalize()
        ln = 0.0010 + mrng.random() * 0.0022
        side = d.cross(nrm)
        if side.length < 1e-6:
            side = t1.copy()
        side.normalize()
        side *= 0.00055
        base = len(moss_verts)
        moss_verts += [origin - side, origin + side, origin + d * ln]
        moss_uvs += [(0, 0), (1, 0), (0.5, 1)]
        moss_faces.append((base, base + 1, base + 2))
moss_obj = new_object("MossTufts", moss_verts, moss_faces, moss_uvs, MAT_MOSSTUFT, recalc=False)
mcol = moss_obj.data.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
mr2 = random.Random(717)
for i in range(len(moss_obj.data.vertices)):
    v = 0.7 + mr2.random() * 0.6
    mcol.data[i].color = (v * 0.10, v * 0.16, v * 0.045, 1.0)


# ----------------------------------------------------------------------------
# rocks — partially embedded in the moss
# ----------------------------------------------------------------------------

rocks = []
rock_defs = [
    (-0.72, 0.42, 0.30, 0.20, 0.26, 0.6, 0.55),
    (-0.28, 0.66, 0.20, 0.13, 0.18, 1.8, 0.45),
    (0.52, 0.52, 0.24, 0.15, 0.20, 2.9, 0.50),
    (0.86, 0.10, 0.15, 0.10, 0.13, 4.1, 0.55),
    (-0.80, -0.06, 0.18, 0.12, 0.16, 5.2, 0.50),
    (0.10, 0.74, 0.13, 0.09, 0.11, 0.9, 0.40),
]
for (px_, pz_, sx, sy, sz, rot, bury) in rock_defs:
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=4, radius=1.0)
    bm.verts.index_update()
    local = []
    for v in bm.verts:
        p = v.co
        n = noise.noise(Vector((p.x * 1.5 + rot * 5, p.y * 1.5 + 3.1, p.z * 1.5 + rot)))
        p = p * (0.94 + n * 0.12)
        if p.y < -0.35:
            p.y = -0.35 + (p.y + 0.35) * 0.35
        local.append(Vector((p.x * sx * R_S, p.y * sy * R_S, p.z * sz * R_S)))
    faces = [tuple(vi.index for vi in f.verts) for f in bm.faces]
    bm.free()
    gy = mound_surface_y(px_ * R_S, pz_ * R_S)
    for v in local:
        v.x = v.x * 1.35 + px_ * R_S
        v.y = v.y * 1.35 + gy - sy * R_S * bury * 2.0 + sy * R_S * 0.62
        v.z = v.z * 1.35 + pz_ * R_S
    uvs = sphere_uv(local)
    rocks.append(new_object("Rock%d" % int(rot * 10), local, faces, uvs, MAT_STONE))

# pebbles along the moss edge
peb_rng = random.Random(2027)
for i in range(26):
    a = peb_rng.random() * math.tau
    r = MOUND_R * (0.85 + peb_rng.random() * 0.2)
    x, z = math.cos(a) * r, math.sin(a) * r
    gy = mound_surface_y(x, z)
    s = 0.004 + peb_rng.random() * 0.007
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    bm.verts.index_update()
    local = [Vector((v.co.x * s * 1.3, v.co.y * s * 0.7, v.co.z * s)) for v in bm.verts]
    faces = [tuple(vi.index for vi in f.verts) for f in bm.faces]
    bm.free()
    for v in local:
        v.x += x
        v.y += gy + s * 0.15
        v.z += z
    uvs = sphere_uv(local)
    rocks.append(new_object("Pebble%d" % i, local, faces, uvs, MAT_STONE))
log("rocks built:", len(rocks))

# fallen petals on the moss
petal_verts = []
petal_faces = []
petal_cols = []
prng = random.Random(4321)
for i in range(60):
    a = prng.random() * math.tau
    r = prng.random() * MOUND_R * 0.9
    x, z = math.cos(a) * r, math.sin(a) * r
    y = mound_surface_y(x, z) + 0.0012
    s = 0.0035 + prng.random() * 0.0025
    rot = prng.random() * math.tau
    ca, sa = math.cos(rot), math.sin(rot)
    base = len(petal_verts)
    pts = [(-s, 0, 0), (0, 0, s * 0.6), (s, 0, 0), (0, 0, -s * 0.6)]
    for (dx, dy, dz) in pts:
        petal_verts.append(Vector((x + dx * ca - dz * sa, y, z + dx * sa + dz * ca)))
        petal_cols.append((0.95, 0.62, 0.70))
    petal_faces.append((base, base + 1, base + 2))
    petal_faces.append((base, base + 2, base + 3))
petal_obj = new_object("FallenPetals", petal_verts, petal_faces, None, MAT_BLOSSOM, smooth=False, recalc=False)
pcol = petal_obj.data.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
for i, c in enumerate(petal_cols):
    pcol.data[i].color = (*c, 1.0)


# ----------------------------------------------------------------------------
# export
# ----------------------------------------------------------------------------

for ob in bpy.context.scene.objects:
    ob.select_set(True)

glb_path = os.path.join(ASSETS, "bonsai.glb")
bpy.ops.export_scene.gltf(
    filepath=glb_path,
    export_format="GLB",
    export_apply=True,
    export_yup=False,
)
log("exported", glb_path, round(os.path.getsize(glb_path) / 1e6, 2), "MB")

tris = 0
for ob in bpy.context.scene.objects:
    if ob.type == "MESH":
        tris += sum(len(p.vertices) - 2 for p in ob.data.polygons)
log("triangles:", tris)
