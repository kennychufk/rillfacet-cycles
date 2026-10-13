"""Blender Cycles camera renderer for rillfacet scenes.

    from rillfacet_cycles import CyclesRenderer
    r = CyclesRenderer(spp=16)
    scene.add_backend(r)          # or Scene(r)
    out = r.render("wrist", outputs=("rgb", "depth", "segmentation"))

A :class:`rillfacet.render.Renderer` backend with the same outputs as
``rillfacet.render.mitsuba.MitsubaRenderer`` (torch tensors ``[I, H, W,
...]`` for every instance of a camera). This package imports Blender's
``bpy`` (GPL-3.0-or-later) and is licensed GPL-3.0-or-later; it lives
outside rillfacet (MIT) for that reason.

How it maps the scene onto Blender, and why:

- One Blender mesh per body of a rigid mesh, shared by one object per
  instance (env); envs sit on rillfacet's grid ``env_spacing`` apart.
  Per-env appearance lives in OBJECT attributes (custom properties
  ``rf_c<slot>`` / ``rf_r<slot>`` / ``rf_m<slot>``: linear base color,
  roughness, metallic per material slot) that one shared material per slot
  reads, so randomizing appearance never rebuilds a shader. A textured
  slot multiplies its texture by the per-env color.
- Cycles renders one camera per call, so a frame is one render per env.
  ``use_persistent_data`` keeps the scene on the GPU between renders: a
  render after moving objects or the camera re-syncs only what changed.
- Cycles has no OpenCV lens. Each env renders a centered PINHOLE proxy that
  covers the camera's field of view at its focal length (``oversample``
  scales it), with Cycles' own thin-lens depth of field and (``denoise``)
  denoiser; torch then resamples it through the exact OpenCV distortion
  (:class:`rillfacet.render.camera.PixelRays`), ``supersample`` x
  ``supersample`` points per output pixel. Wide lenses cost more proxy
  pixels (the 120-degree Pi Camera Module 3 Wide: ~1.8x the output's).
- Depth and segmentation come from a second render of the same scene: one
  sample per pixel at the pixel center, the compositor passing on the
  ``rf_geo`` AOV (camera z-depth, label, 1) every material writes. Only
  film settings change between the two, so the session stays warm (~8 ms
  beauty + ~3.5 ms geometry per env at 432 x 244 on an RTX 5070 Ti, the
  oven-grasp scene). Each output pixel's center ray
  takes the label of its nearest proxy pixel and a depth interpolated in
  1/z between its 4 neighbors (exact on planes) where they agree.
- Images go through OpenEXR files in a tmpfs directory (``/dev/shm``):
  bpy exposes no render pixels in background mode.
- Lights are shared by all envs (directional = sun, ambient = world color,
  envmap = world environment texture, Blender's equirectangular mapping).

Not modelled yet: motion blur.
"""
from __future__ import annotations

import contextlib
import itertools
import os
import shutil
import tempfile
import time
import warnings

import numpy as np
from rillfacet.render import Renderer, linear_to_srgb, srgb_to_linear
from rillfacet.render.camera import PixelRays
from rillfacet.scene_api import Light

__all__ = ["CyclesRenderer"]
__version__ = "0.1.0"

_DEFAULT_LIGHTS = (
    dict(type="ambient", intensity=0.35, color=(1.0, 1.0, 1.0)),
    dict(type="directional", intensity=2.5, color=(1.0, 1.0, 1.0),
         direction=(0.3, 0.2, -1.0)),
)
_ids = itertools.count()

#: Largest tangent of the angle off the optical axis a pinhole proxy covers
#: (about 79 degrees).
MAX_TAN = 5.0


@contextlib.contextmanager
def _quiet(on=True):
    """Silence Blender's per-render prints (C-level stdout / stderr)."""
    if not on:
        yield
        return
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, null):
            os.close(fd)


def _use_nodes(block):
    """Node trees are always on from Blender 5 (``use_nodes`` deprecated)."""
    import bpy
    if bpy.app.version < (5, 0, 0):
        block.use_nodes = True


def _quat_mat(q):
    """``[..., 4]`` xyzw -> ``[..., 3, 3]``."""
    x, y, z, w = np.moveaxis(np.asarray(q, np.float64), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                  2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                  2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w),
                  1 - 2 * (x * x + y * y)], -1)], -2)


class _Proxy:
    """A camera's pinhole proxy and its resampling grids."""

    def __init__(self, cam, rays, oversample, supersample, torch, dev):
        self.torch = torch
        K = cam.K
        c = rays.corners.reshape(-1, 2)
        tx = float(c[:, 0].abs().max())
        ty = float(c[:, 1].abs().max())
        if max(tx, ty) > MAX_TAN:
            raise ValueError(
                f"camera {cam.name!r}: its rays reach {np.degrees(np.arctan(max(tx, ty))):.0f}"
                " degrees off axis, beyond what a pinhole proxy covers")
        f = float(oversample) * max(K[0, 0], K[1, 1])
        self.f = f
        self.width = 2 * int(np.ceil(f * tx + 1.0))
        self.height = 2 * int(np.ceil(f * ty + 1.0))
        self.cx, self.cy = (self.width - 1) / 2, (self.height - 1) / 2
        W, H, S = cam.width, cam.height, int(supersample)
        o = (torch.arange(S, device=dev, dtype=torch.float32) + 0.5) / S - 0.5
        a, b = torch.meshgrid(o, o, indexing="ij")
        a, b = a.reshape(1, -1, 1), b.reshape(1, -1, 1)          # [1, S2, 1]
        xy = (rays.center[:, None] + a * rays.d_du[:, None]
              + b * rays.d_dv[:, None])                           # [P, S2, 2]
        self.grid_rgb = self._grid(xy).reshape(1, H * W, S * S, 2)
        # The center rays' 4 proxy neighbors (flat indices, bilinear
        # weights) and the nearest of them, for the geometry pass.
        u = self.f * rays.center[:, 0] + self.cx
        v = self.f * rays.center[:, 1] + self.cy
        x0, y0 = u.floor(), v.floor()
        fx, fy = u - x0, v - y0
        idx, wts = [], []
        for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            xi = (x0 + dx).clamp(0, self.width - 1).long()
            yi = (y0 + dy).clamp(0, self.height - 1).long()
            idx.append(yi * self.width + xi)
            wts.append((fx if dx else 1 - fx) * (fy if dy else 1 - fy))
        self.nb_idx = torch.stack(idx, -1)                       # [P, 4]
        self.nb_w = torch.stack(wts, -1)
        self.nearest = self.nb_w.argmax(-1, keepdim=True)        # [P, 1]

    def _grid(self, xy):
        """Normalized coordinates (``grid_sample``, ``align_corners=False``)
        of camera-plane points in the proxy image (row 0 = top)."""
        u = self.f * xy[..., 0] + self.cx
        v = self.f * xy[..., 1] + self.cy
        return self.torch.stack([(2 * u + 1) / self.width - 1,
                                 (2 * v + 1) / self.height - 1], -1)


class CyclesRenderer(Renderer):
    """Blender Cycles renderer (see the module docstring).

    ``spp`` samples per (proxy) pixel; ``max_bounces`` the path length (0 =
    direct light only); ``denoise`` runs Cycles' denoiser (OptiX on an
    OptiX device, else OpenImageDenoise) on each proxy image; ``device``
    ``"OPTIX"``, ``"CUDA"`` or ``"CPU"`` (falls back to CPU with a warning);
    ``sun_angle`` (degrees) the directional lights' angular diameter;
    ``oversample`` / ``supersample`` the proxy resolution and resampling
    (module docstring); ``quiet`` silences Blender's per-render prints.
    Without :class:`~rillfacet.scene_api.Light` s the scene gets a sky and
    a sun (the Mitsuba backend's defaults).

    Blender is a process-wide singleton; every renderer gets its own
    Blender scene and datablocks (prefix ``rf<N>_``), and :meth:`close`
    deletes them.
    """

    def __init__(self, *, spp=16, max_bounces=4, denoise=False,
                 device="OPTIX", env_spacing=50.0, sun_angle=0.53,
                 oversample=1.0, supersample=3, seed=0, quiet=True,
                 tmpdir=None):
        super().__init__(env_spacing=env_spacing, seed=seed)
        import torch
        with _quiet(quiet):           # Cycles' HIP probe warns on NVIDIA hosts
            import bpy
        self.bpy, self.torch = bpy, torch
        self.dev = torch.device("cuda" if torch.cuda.is_available()
                                else "cpu")
        self.spp, self.max_bounces = int(spp), int(max_bounces)
        self.denoise = bool(denoise)
        self.sun_angle = float(sun_angle)
        self.oversample, self.supersample = float(oversample), int(supersample)
        self.quiet = bool(quiet)
        self._seed = int(seed) * 7919
        self._prefix = f"rf{next(_ids)}_"
        base = tmpdir or ("/dev/shm" if os.path.isdir("/dev/shm") else None)
        self._tmp = tempfile.mkdtemp(prefix="rillfacet_cycles_", dir=base)
        self._device = self._setup_device(device)
        self._lights_dirty = True
        self._blocks = []                      # our per-build datablocks
        self._objects: dict = {}               # mesh name -> [(i, b, ob)]
        self._slots: dict = {}                 # mesh name -> [slots] per body
        self._proxies: dict = {}               # camera -> (key, rays, proxy)
        self._images: dict = {}
        self._hidden: dict = {}
        self.last_timing: dict = {}
        self._aperture = 0.0
        self._make_scenes()

    # -- setup ----------------------------------------------------------------

    def _setup_device(self, device):
        device = device.upper()
        if device == "CPU":
            return "CPU"
        prefs = self.bpy.context.preferences.addons["cycles"].preferences
        for kind in dict.fromkeys((device, "CUDA")):
            try:
                prefs.compute_device_type = kind
            except TypeError:
                continue
            prefs.get_devices()
            if any(d.type == kind for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type == kind
                self._kind = kind
                return "GPU"
        warnings.warn("CyclesRenderer: no GPU device, rendering on the CPU")
        return "CPU"

    def _make_scenes(self):
        """One scene for both passes. The beauty pass renders ``spp``
        samples; the geometry pass one sample at the pixel center, the
        compositor writing the ``rf_geo`` AOV (camera z-depth, label, 1)
        that every material emits alongside its BSDF. Switching passes
        touches only film / sampling settings and the compositor flag, so
        Cycles' persistent session stays warm (a material override or an
        emissive switch rebuilds the light distribution, ~70 ms)."""
        bpy = self.bpy
        self._coll = bpy.data.collections.new(self._prefix + "objects")
        self._light_coll = bpy.data.collections.new(self._prefix + "lights")
        sc = self._sc = bpy.data.scenes.new(self._prefix + "scene")
        r = sc.render
        r.engine = "CYCLES"
        sc.cycles.device = self._device
        r.resolution_percentage = 100
        r.pixel_aspect_x = r.pixel_aspect_y = 1.0
        r.use_persistent_data = True
        r.use_sequencer = False
        r.use_file_extension = False
        im = r.image_settings
        im.file_format = "OPEN_EXR"
        im.color_depth = "32"
        im.color_mode = "RGBA"
        im.exr_codec = "NONE"
        sc.view_settings.view_transform = "Standard"
        c = sc.cycles
        c.use_adaptive_sampling = False
        if self.denoise:
            c.denoiser = ("OPTIX" if getattr(self, "_kind", "") == "OPTIX"
                          else "OPENIMAGEDENOISE")
        cam = bpy.data.cameras.new(self._prefix + "cam")
        cam.sensor_fit = "HORIZONTAL"
        cam.sensor_width = 36.0
        ob = bpy.data.objects.new(self._prefix + "cam", cam)
        ob.rotation_mode = "QUATERNION"
        sc.collection.objects.link(ob)
        sc.camera = ob
        sc.collection.children.link(self._coll)
        sc.collection.children.link(self._light_coll)
        self._world = bpy.data.worlds.new(self._prefix + "world")
        _use_nodes(self._world)
        sc.world = self._world
        aov = sc.view_layers[0].aovs.add()
        aov.name, aov.type = "rf_geo", "COLOR"
        ng = bpy.data.node_groups.new(self._prefix + "comp",
                                      "CompositorNodeTree")
        ng.interface.new_socket("Image", in_out="OUTPUT",
                                socket_type="NodeSocketColor")
        rl = ng.nodes.new("CompositorNodeRLayers")
        rl.scene = sc
        go = ng.nodes.new("NodeGroupOutput")
        ng.links.new(rl.outputs["rf_geo"], go.inputs[0])
        sc.compositing_node_group = ng
        self._mode = None
        self._static = [sc, self._coll, self._light_coll, self._world, ng,
                        cam]

    def _set_mode(self, geom, spp):
        """Beauty (``geom=False``, ``spp`` samples) or geometry pass."""
        mode = (geom, None if geom else spp)
        if mode == self._mode:
            return
        sc, c = self._sc, self._sc.cycles
        sc.render.use_compositing = geom
        c.samples = 1 if geom else spp
        c.filter_width = 0.01 if geom else 1.5        # 0.01: the pixel center
        c.max_bounces = 0 if geom else self.max_bounces
        c.use_denoising = self.denoise and not geom
        self._mode = mode

    def close(self) -> None:
        """Delete this renderer's Blender data and temporary files."""
        self._clear()
        bpy = self.bpy
        self._clear_lights()
        for img in self._images.values():
            bpy.data.images.remove(img)
        self._images = {}
        bpy.data.objects.remove(self._sc.camera)
        for blk in self._static:
            coll = {bpy.types.Scene: bpy.data.scenes,
                    bpy.types.Collection: bpy.data.collections,
                    bpy.types.World: bpy.data.worlds,
                    bpy.types.Camera: bpy.data.cameras,
                    bpy.types.CompositorNodeTree: bpy.data.node_groups,
                    }[type(blk)]
            coll.remove(blk)
        self._static = []
        shutil.rmtree(self._tmp, ignore_errors=True)

    # -- Backend: lights do not rebuild the geometry ---------------------------

    def create(self, obj) -> None:
        was = self._structure_dirty
        super().create(obj)
        if isinstance(obj, Light):
            self._structure_dirty = was
            self._lights_dirty = True

    def update(self, obj, what: str) -> None:
        if isinstance(obj, Light):
            self._lights_dirty = True
            return
        super().update(obj, what)

    def remove(self, name: str) -> None:
        was, was_light = self._structure_dirty, name in self.lights
        super().remove(name)
        if was_light:
            self._structure_dirty = was
            self._lights_dirty = True

    # -- build ----------------------------------------------------------------

    def _clear(self):
        bpy = self.bpy
        for objs in self._objects.values():
            for _, _, ob in objs:
                bpy.data.objects.remove(ob)
        self._objects = {}
        for blk in self._blocks:
            {bpy.types.Mesh: bpy.data.meshes,
             bpy.types.Material: bpy.data.materials,
             bpy.types.Image: bpy.data.images}[type(blk)].remove(blk)
        self._blocks = []
        self._hidden = {}

    def _build(self):
        self._clear()
        bases = self.label_bases()
        for mi, m in enumerate(self.meshes.values()):
            self._build_mesh(mi, m, bases[m.name])
        self._structure_dirty = False
        self._dirty_poses = set(self.meshes)
        self._dirty_appearance = set(self.meshes)

    def _material(self, name, mat, slot, textured):
        """A shared Principled BSDF reading the per-object attributes of
        ``slot`` (and multiplying ``mat``'s texture by its color)."""
        bpy = self.bpy
        bm = bpy.data.materials.new(name)
        self._blocks.append(bm)
        _use_nodes(bm)
        nt = bm.node_tree
        nt.nodes.clear()
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
        # the geometry pass's AOV: (camera z-depth, label, 1)
        cd = nt.nodes.new("ShaderNodeCameraData")
        lab = nt.nodes.new("ShaderNodeAttribute")
        lab.attribute_type = "OBJECT"
        lab.attribute_name = "rf_label"
        xyz = nt.nodes.new("ShaderNodeCombineXYZ")
        xyz.inputs[2].default_value = 1.0
        geo = nt.nodes.new("ShaderNodeOutputAOV")
        geo.aov_name = "rf_geo"
        nt.links.new(cd.outputs["View Z Depth"], xyz.inputs[0])
        nt.links.new(lab.outputs["Fac"], xyz.inputs[1])
        nt.links.new(xyz.outputs[0], geo.inputs["Color"])

        def attr(prefix):
            a = nt.nodes.new("ShaderNodeAttribute")
            a.attribute_type = "OBJECT"
            a.attribute_name = f"{prefix}{slot}"
            return a
        color = attr("rf_c").outputs["Color"]
        if textured:
            img = self._texture_image(name, mat.base_color_texture)
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = img
            tex.interpolation = "Linear"
            mul = nt.nodes.new("ShaderNodeVectorMath")
            mul.operation = "MULTIPLY"
            nt.links.new(tex.outputs["Color"], mul.inputs[0])
            nt.links.new(color, mul.inputs[1])
            color = mul.outputs["Vector"]
        nt.links.new(color, bsdf.inputs["Base Color"])
        nt.links.new(attr("rf_r").outputs["Fac"], bsdf.inputs["Roughness"])
        nt.links.new(attr("rf_m").outputs["Fac"], bsdf.inputs["Metallic"])
        return bm

    def _texture_image(self, name, tex):
        bpy = self.bpy
        if isinstance(tex, str):
            from PIL import Image
            tex = np.asarray(Image.open(tex).convert("RGB"))
        tex = np.asarray(tex)
        tex = (tex.astype(np.float32) / 255.0 if tex.dtype == np.uint8
               else tex.astype(np.float32))
        if tex.shape[-1] == 3:
            tex = np.concatenate([tex, np.ones_like(tex[..., :1])], -1)
        h, w = tex.shape[:2]
        img = bpy.data.images.new(name + "_tex", w, h, alpha=True)
        self._blocks.append(img)
        img.colorspace_settings.name = "sRGB"
        img.pixels.foreach_set(np.ascontiguousarray(tex[::-1]).ravel())
        img.pack()
        return img

    def _build_mesh(self, mi, m, label_base):
        bpy = self.bpy
        I, B = m.instances, m.num_bodies
        mats = m.materials
        textured = {s for s, mat in enumerate(mats)
                    if mat.base_color_texture is not None and m.uvs is not None}
        if mats and m.uvs is None and any(
                mat.base_color_texture is not None for mat in mats):
            warnings.warn(f"rigid mesh {m.name!r}: textures without UVs are "
                          "drawn untextured")
        tag = f"{self._prefix}{mi}_"
        nslots = max(len(mats), 1)
        bmats = [self._material(f"{tag}mat{s}", mats[s] if mats else None, s,
                                s in textured) for s in range(nslots)]
        tri_mat = (m.triangle_material if m.triangle_material is not None
                   else np.zeros(len(m.triangles), np.int32))
        objs, body_slots = [], []
        off = m.body_triangle_offsets
        for b in range(B):
            t0, t1 = int(off[b]), int(off[b + 1])
            if t1 <= t0:
                body_slots.append([])
                continue
            tris = m.triangles[t0:t1].astype(np.int64)
            gv, inv = np.unique(tris.ravel(), return_inverse=True)
            used, local = np.unique(tri_mat[t0:t1], return_inverse=True)
            me = bpy.data.meshes.new(f"{tag}b{b}")
            self._blocks.append(me)
            nv, nt = len(gv), t1 - t0
            me.vertices.add(nv)
            me.vertices.foreach_set(
                "co", m.local_vertices[gv].astype(np.float32).ravel())
            me.loops.add(3 * nt)
            me.loops.foreach_set("vertex_index", inv.astype(np.int32))
            me.polygons.add(nt)
            me.polygons.foreach_set("loop_start",
                                    np.arange(0, 3 * nt, 3, dtype=np.int32))
            me.polygons.foreach_set("material_index", local.astype(np.int32))
            for s in used:
                me.materials.append(bmats[int(s)])
            if m.uvs is not None and any(int(s) in textured for s in used):
                uv = m.uvs[gv][inv].astype(np.float32)
                uv[:, 1] = 1.0 - uv[:, 1]            # glTF's v is top-down
                layer = me.uv_layers.new(name="UVMap")
                layer.data.foreach_set("uv", uv.ravel())
            me.update(calc_edges=True)
            # Flat unless normals are given (Blender >= 4.1 meshes are
            # smooth by default).
            me.polygons.foreach_set("use_smooth", np.full(
                nt, m.normals is not None, bool))
            if m.normals is not None:
                me.normals_split_custom_set_from_vertices(
                    m.normals[gv].astype(np.float32))
            body_slots.append([int(s) for s in used])
            for i in range(I):
                ob = bpy.data.objects.new(f"{tag}i{i}_b{b}", me)
                ob.rotation_mode = "QUATERNION"
                ob["rf_label"] = float(label_base + b)
                self._coll.objects.link(ob)
                objs.append((i, b, ob))
        self._objects[m.name] = objs
        self._slots[m.name] = body_slots

    def _clear_lights(self):
        bpy = self.bpy
        for ob in list(self._light_coll.objects):
            data = ob.data
            bpy.data.objects.remove(ob)
            bpy.data.lights.remove(data)
        nt = self._world.node_tree
        for n in list(nt.nodes):
            if n.type == "TEX_ENVIRONMENT" and n.image is not None:
                img = n.image
                nt.nodes.remove(n)
                if img.users == 0:
                    bpy.data.images.remove(img)

    def _build_lights(self):
        import mathutils
        bpy = self.bpy
        self._clear_lights()
        lights = ([dict(type=l.type, intensity=l.intensity, color=l.color,
                        direction=l.direction, envmap=l.envmap)
                   for l in self.lights.values()]
                  or [dict(d, envmap=None) for d in _DEFAULT_LIGHTS])
        nt = self._world.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputWorld")
        ambient = np.zeros(3)
        shader = None
        for k, l in enumerate(lights):
            c = np.asarray(l["color"], np.float64) * l["intensity"]
            if l["type"] == "ambient":
                ambient += c
            elif l["type"] == "directional":
                data = bpy.data.lights.new(f"{self._prefix}sun{k}", "SUN")
                data.color = tuple(np.asarray(l["color"], np.float64))
                data.energy = float(l["intensity"])
                data.angle = np.radians(self.sun_angle)
                ob = bpy.data.objects.new(f"{self._prefix}sun{k}", data)
                d = mathutils.Vector(l["direction"]).normalized()
                ob.rotation_mode = "QUATERNION"
                ob.rotation_quaternion = d.to_track_quat("-Z", "Y")
                self._light_coll.objects.link(ob)
            else:
                env = nt.nodes.new("ShaderNodeTexEnvironment")
                env.image = bpy.data.images.load(str(l["envmap"]))
                bg = nt.nodes.new("ShaderNodeBackground")
                bg.inputs["Strength"].default_value = float(l["intensity"])
                nt.links.new(env.outputs["Color"], bg.inputs["Color"])
                shader = self._add_shader(nt, shader, bg.outputs[0])
        if ambient.any() or shader is None:
            bg = nt.nodes.new("ShaderNodeBackground")
            bg.inputs["Color"].default_value = (*ambient, 1.0)
            bg.inputs["Strength"].default_value = 1.0
            shader = self._add_shader(nt, shader, bg.outputs[0])
        nt.links.new(shader, out.inputs["Surface"])
        self._lights_dirty = False

    @staticmethod
    def _add_shader(nt, a, b):
        if a is None:
            return b
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(a, add.inputs[0])
        nt.links.new(b, add.inputs[1])
        return add.outputs[0]

    # -- per-render sync ------------------------------------------------------

    def _sync(self):
        if self._structure_dirty:
            self._build()
        if self._lights_dirty:
            self._build_lights()
        for name in self._dirty_appearance:
            m = self.meshes.get(name)
            if m is not None:
                self._write_appearance(m)
        for name in self._dirty_poses:
            m = self.meshes.get(name)
            if m is not None:
                self._write_poses(m)
        self._dirty_poses.clear()
        self._dirty_appearance.clear()

    def _write_appearance(self, m):
        I, B = m.instances, m.num_bodies
        slots = self._slots[m.name]
        if m.materials:
            color = srgb_to_linear(np.asarray(m.material_base_color,
                                              np.float32))
            rough = np.asarray(m.material_roughness, np.float32)
            metal = np.asarray(m.material_metallic, np.float32)
        else:
            seg = srgb_to_linear(np.asarray(m.segment_colors, np.float32))
        for i, b, ob in self._objects[m.name]:
            for s in slots[b]:
                if m.materials:
                    ob[f"rf_c{s}"] = color[i, s].tolist()
                    ob[f"rf_r{s}"] = float(rough[i, s])
                    ob[f"rf_m{s}"] = float(metal[i, s])
                else:
                    ob[f"rf_c{s}"] = seg[i * B + b].tolist()
                    ob[f"rf_r{s}"] = 0.5
                    ob[f"rf_m{s}"] = 0.0

    def _write_poses(self, m):
        I, B = m.instances, m.num_bodies
        if m.poses is None:
            poses = np.zeros((I, B, 7), np.float64)
            poses[..., 6] = 1.0
        else:
            p = m.poses
            poses = (p.detach().double().cpu().numpy() if hasattr(p, "detach")
                     else np.asarray(p, np.float64)).reshape(I, B, 7)
        offs = self.env_offsets(I).astype(np.float64)
        pos = poses[..., :3] + offs[:, None]
        quat = poses[..., [6, 3, 4, 5]]                      # wxyz
        vis = np.asarray(m.segment_visible).reshape(I, B)
        hidden = self._hidden.setdefault(m.name, {})
        for i, b, ob in self._objects[m.name]:
            ob.location = pos[i, b]
            ob.rotation_quaternion = quat[i, b]
            h = not bool(vis[i, b])
            if hidden.get(ob.name, False) != h:
                ob.hide_render = h
                hidden[ob.name] = h

    # -- cameras --------------------------------------------------------------

    def _proxy(self, cam):
        key = (cam.width, cam.height, cam.K.tobytes(),
               None if cam.dist is None else cam.dist.tobytes(),
               self.oversample, self.supersample)
        hit = self._proxies.get(cam.name)
        if hit is None or hit[0] != key:
            rays = PixelRays(cam.K, cam.width, cam.height, cam.dist,
                             device=self.dev)
            if rays.max_residual > 0.05:
                warnings.warn(f"camera {cam.name!r}: undistortion residual "
                              f"{rays.max_residual:.3f} px")
            if rays.outside > 0:
                warnings.warn(f"camera {cam.name!r}: {100 * rays.outside:.2f}"
                              "% of the image lies beyond the distortion "
                              "model's fold (drawn with its edge rays)")
            proxy = _Proxy(cam, rays, self.oversample, self.supersample,
                           self.torch, self.dev)
            hit = self._proxies[cam.name] = (key, proxy)
        return hit[1]

    def _place_camera(self, cam, proxy, R, t, i):
        """Point the Blender camera along env ``i``'s camera."""
        import mathutils
        M = np.eye(4)
        M[:3, :3] = R[i] @ np.diag([1.0, -1.0, -1.0])        # OpenCV -> Blender
        M[:3, 3] = t[i]
        ob = self._sc.camera
        ob.matrix_world = mathutils.Matrix(M.tolist())
        c = ob.data
        lens = proxy.f * 36.0 / proxy.width
        if c.lens != lens:
            c.lens = lens
        c.clip_start, c.clip_end = cam.near, cam.far
        self._aperture = float(cam.aperture_radius[i])
        if self._aperture > 0:
            c.dof.focus_distance = float(cam.focus_distance[i])
            c.dof.aperture_fstop = lens * 1e-3 / (2.0 * self._aperture)
            c.dof.aperture_blades = 0
            c.dof.aperture_ratio = 1.0
        r = self._sc.render
        if (r.resolution_x, r.resolution_y) != (proxy.width, proxy.height):
            r.resolution_x, r.resolution_y = proxy.width, proxy.height

    def _render_pass(self, geom, spp):
        """Render the beauty or the geometry pass of the placed camera;
        ``[h, w, 4]`` float32, row 0 = top."""
        bpy, sc = self.bpy, self._sc
        self._set_mode(geom, spp)
        sc.camera.data.dof.use_dof = self._aperture > 0 and not geom
        kind = "geom" if geom else "shade"
        path = os.path.join(self._tmp, f"{kind}.exr")
        sc.render.filepath = path
        self._seed += 1
        sc.cycles.seed = self._seed % (2 ** 31)
        with _quiet(self.quiet):
            bpy.ops.render.render(write_still=True, scene=sc.name)
        img = self._images.get(kind)
        if img is None:
            img = self._images[kind] = bpy.data.images.load(
                path, check_existing=False)
            img.colorspace_settings.name = "Non-Color"
        else:
            img.reload()
        w, h = img.size
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
        return px.reshape(h, w, 4)[::-1]

    # -- render ---------------------------------------------------------------

    def _resample_geometry(self, geom, proxy, I, H, W):
        """Labels of the nearest proxy pixel; z-depth interpolated in 1/z
        (exact on planes) where the 4 neighbors are hits of that label at
        similar depths, else the nearest's."""
        torch = self.torch
        g = torch.as_tensor(geom[..., :3], device=self.dev).reshape(I, -1, 3)
        nb = g[:, proxy.nb_idx]                                 # [I, P, 4, 3]
        z, lab, hit = nb[..., 0], nb[..., 1], nb[..., 2] > 0.5
        near = proxy.nearest.expand(I, -1, -1)[..., None]        # [I, P, 1, 1]
        zn = torch.gather(z, 2, near[..., 0])[..., 0]
        ln = torch.gather(lab, 2, near[..., 0])[..., 0]
        hn = torch.gather(hit, 2, near[..., 0])[..., 0]
        same = (hit.all(-1) & (lab == ln[..., None]).all(-1)
                & (z.amax(-1) - z.amin(-1) < 0.05 * zn))
        zi = 1.0 / (proxy.nb_w / z.clamp(min=1e-9)).sum(-1)
        depth = torch.where(hn, torch.where(same, zi, zn), torch.zeros_like(zn))
        seg = torch.where(hn, ln.round().int(),
                          torch.full_like(zn, -1, dtype=torch.int32))
        return depth.reshape(I, H, W), seg.reshape(I, H, W)

    def render(self, camera, outputs=("rgb",), *, spp=None) -> dict:
        torch = self.torch
        F = torch.nn.functional
        bad = set(outputs) - set(self.OUTPUTS)
        if bad:
            raise ValueError(f"unknown outputs {sorted(bad)}")
        cam = self._camera(camera)
        for m in self.meshes.values():
            if m.instances != cam.instances:
                raise ValueError(f"camera {camera!r} has {cam.instances} "
                                 f"instances, rigid mesh {m.name!r} "
                                 f"{m.instances}")
        t0 = time.perf_counter()
        self._sync()
        t_sync = time.perf_counter() - t0
        I, W, H = cam.instances, cam.width, cam.height
        proxy = self._proxy(cam)
        pose = cam.world_poses()
        pose = (pose.detach().double().cpu().numpy() if hasattr(pose, "detach")
                else np.asarray(pose, np.float64)).reshape(I, 7)
        R = _quat_mat(pose[:, 3:])
        t = pose[:, :3] + self.env_offsets(I)
        want_rgb = "rgb" in outputs or "linear" in outputs
        want_geom = "depth" in outputs or "segmentation" in outputs
        shade = np.empty((I, proxy.height, proxy.width, 4), np.float32) \
            if want_rgb else None
        geom = np.empty((I, proxy.height, proxy.width, 4), np.float32) \
            if want_geom else None
        spp = int(spp or self.spp)
        t0 = time.perf_counter()
        for i in range(I):
            self._place_camera(cam, proxy, R, t, i)
            if want_rgb:
                shade[i] = self._render_pass(False, spp)
            if want_geom:
                geom[i] = self._render_pass(True, spp)
        t_render = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = {}
        if want_rgb:
            img = torch.as_tensor(shade[..., :3], device=self.dev)
            img = img.permute(0, 3, 1, 2)                       # [I, 3, h, w]
            g = proxy.grid_rgb.expand(I, -1, -1, -1)
            lin = F.grid_sample(img, g, mode="bilinear", padding_mode="border",
                                align_corners=False).mean(-1)   # [I, 3, HW]
            lin = lin.reshape(I, 3, H, W).permute(0, 2, 3, 1).clamp(min=0)
            lin = self.sensor_image(cam, lin)
            if "linear" in outputs:
                out["linear"] = lin
            if "rgb" in outputs:
                out["rgb"] = linear_to_srgb(lin)
        if want_geom:
            depth, seg = self._resample_geometry(geom, proxy, I, H, W)
            if "depth" in outputs:
                out["depth"] = depth
            if "segmentation" in outputs:
                out["segmentation"] = seg
        self.last_timing = dict(sync=t_sync, render=t_render,
                                resample=time.perf_counter() - t0,
                                proxy=(proxy.width, proxy.height))
        return out
