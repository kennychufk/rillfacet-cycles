# rillfacet-cycles

A [Blender Cycles](https://www.cycles-renderer.org/) camera renderer backend
for [rillfacet](https://github.com/kennychufk/rillfacet) scenes: RGB, z-depth
and segmentation of every env's camera, as torch tensors, for vision-based RL
and synthetic data.

**License: GPL-3.0-or-later.** It imports Blender's `bpy` module (GPL-3.0-or-later),
which is why it is a separate package from rillfacet (MIT). Keep it out of
anything you distribute under other terms. rillfacet's own Mitsuba 3 backend
(`rillfacet.render.mitsuba`, BSD) has the same interface.

```bash
pip install rillfacet bpy torch pillow
pip install -e .                # this package
python -m pytest tests -q
```

`bpy` wheels on PyPI track a Python version per Blender release (bpy 5.x:
Python 3.13).

## Use

```python
from rillfacet.scene_api import Scene, Material
from rillfacet_cycles import CyclesRenderer

r = CyclesRenderer(spp=16)                 # OptiX on NVIDIA GPUs
scene = Scene(r)
mesh = scene.rigid_mesh("robot", ..., instances=E, materials=[...])
scene.light("sun", "directional", direction=(0.4, 0.3, -1.0), intensity=2.5)
scene.camera_sensor("wrist", 320, 180, K, dist=dist, mount=mesh, body=7,
                    local_pose=cam_in_body, aperture_radius=6e-4,
                    focus_distance=focus)            # per env
out = r.render("wrist", outputs=("rgb", "depth", "segmentation"))
out["rgb"]            # [E, H, W, 3] sRGB 0..1, torch, CUDA
out["depth"]          # [E, H, W] z-depth in m, 0 = nothing
out["segmentation"]   # [E, H, W] int32, -1 = background; r.segment_labels()
```

Per-env domain randomization is a write to the scene, never a rebuild:
`mesh.set_materials(base_color=..., roughness=..., metallic=...)` (textured
materials multiply their texture by the per-env color), `CameraSensor.set_lens`
(aperture, focus distance, gain = exposure and white balance).

## How it works

See the module docstring of `rillfacet_cycles`. In short:

- One Blender mesh per body, one object per (env, body); envs on a grid
  `env_spacing` apart. Per-env appearance is in object custom properties
  that shared materials read through Attribute nodes.
- Cycles renders one camera per call, so a frame is one render per env.
  Persistent data keeps the scene on the GPU between renders.
- OpenCV lenses (the 14-coefficient rational + thin-prism model included):
  each env renders a pinhole proxy covering the field of view, with Cycles'
  thin-lens depth of field and denoiser; torch resamples it through the exact
  distortion.
- Depth and segmentation come from a second, one-sample render of the same
  scene: an AOV every material writes, passed on by the compositor. Only film
  settings change between the two renders, so Cycles' session stays warm.

Measured on an RTX 5070 Ti (the quiddel-rl oven-grasp scene, UR5e visual
meshes, 320 x 180 Pi Camera Module 3 Wide, 16 spp path tracing, RGB + depth +
segmentation): ~15 ms per env per frame at 16 envs. Mitsuba renders all envs in
one launch and is ~6x faster at this size. Cycles has the better noise (light
tree, MIS) and denoiser.

Not modelled yet: motion blur.
