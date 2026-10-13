"""CyclesRenderer on small scenes: geometry (z-depth, labels), appearance
(per-instance materials, textures), lens (distortion, depth of field), and
agreement with rillfacet's Mitsuba backend."""
import numpy as np
import pytest

bpy = pytest.importorskip("bpy")
torch = pytest.importorskip("torch")

from rillfacet.render.camera import look_at              # noqa: E402
from rillfacet.scene_api import Material, Scene          # noqa: E402
from rillfacet_cycles import CyclesRenderer              # noqa: E402

#: Pi Camera Module 3 Wide (OpenCV rational + thin prism) at 4608 x 2592.
PI_K = np.array([[1955.85657, 0.0, 2311.34803], [0.0, 1955.67366, 1305.57668],
                 [0.0, 0.0, 1.0]])
PI_DIST = np.array([-1.36533644, 0.969922653, 1.08871707e-3, -2.14976849e-5,
                    -0.106356684, -1.33222722, 0.887543784, -5.29946910e-2,
                    0, 0, 0, 0, 0, 0])


def _box():
    v = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5)
                  for z in (0.0, 1.0)], np.float32)
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4),
         (1, 5, 7, 3)]
    t = [(a, b, c) for a, b, c, d in q] + [(a, c, d) for a, b, c, d in q]
    return v, np.array(t)


def _scene(renderer, *, textured=False, instances=2, cube_x=(0.0, 2.0)):
    """A unit cube (body 0, red) on a 10 m ground quad (body 1, grey or a
    checker texture), the cube at ``cube_x[i]`` in instance ``i``."""
    bv, bt = _box()
    gv = np.array([[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]], np.float32)
    gt = np.array([[0, 1, 2], [0, 2, 3]]) + len(bv)
    uvs = np.zeros((len(bv) + 4, 2), np.float32)
    uvs[len(bv):] = [[0, 1], [1, 1], [1, 0], [0, 0]]
    tex = None
    if textured:
        k = (np.add.outer(np.arange(64) // 8, np.arange(64) // 8) % 2)
        tex = np.repeat((k * 200 + 40).astype(np.uint8)[..., None], 3, -1)
    s = Scene(renderer)
    rm = s.rigid_mesh(
        "world", np.concatenate([bv, gv]),
        np.r_[np.zeros(len(bv), int), np.ones(4, int)], np.concatenate([bt, gt]),
        np.array([0, len(bt), len(bt) + 2]), instances=instances,
        body_names=["cube", "ground"],
        materials=[Material("red", (0.9, 0.1, 0.1)),
                   Material("ground", (0.6, 0.6, 0.6) if tex is None
                            else (1.0, 1.0, 1.0), base_color_texture=tex)],
        triangle_material=np.r_[np.zeros(len(bt), int), np.ones(2, int)],
        uvs=uvs)
    poses = np.zeros((instances, 2, 7), np.float32)
    poses[..., 6] = 1.0
    poses[:, 0, 0] = cube_x[:instances]
    rm.set_poses(poses)
    return s, rm


def _pinhole(w, h, f):
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])


@pytest.fixture(scope="module")
def renderer():
    r = CyclesRenderer(spp=4)
    yield r
    r.close()


def _fresh(r):
    """Detach ``r`` from its previous scene's objects."""
    for name in list(r.meshes) + list(r.lights) + list(r.cameras):
        r.remove(name)
    r._structure_dirty = True


def test_depth_labels_and_color(renderer):
    _fresh(renderer)
    s, _ = _scene(renderer)
    W, H = 64, 48
    s.camera_sensor("top", W, H, _pinhole(W, H, 60.0), instances=2,
                    local_pose=look_at((0, 0, 3.0), (0, 0, 0), up=(0, 1, 0)),
                    near=0.05, far=10.0)
    out = renderer.render("top", outputs=("rgb", "depth", "segmentation"))
    d, seg, rgb = (out["depth"].cpu().numpy(), out["segmentation"].cpu().numpy(),
                   out["rgb"].cpu().numpy())
    assert out["rgb"].shape == (2, H, W, 3) and seg.dtype == np.int32
    labels = renderer.segment_labels()
    assert labels == ["world/cube", "world/ground"]
    c = (H // 2, W // 2)
    # cube top at z = 1 (instance 0), the ground at z = 0 (instance 1)
    assert d[0][c] == pytest.approx(2.0, abs=1e-3)
    assert d[1][c] == pytest.approx(3.0, abs=1e-3)
    assert seg[0][c] == 0 and seg[1][c] == 1
    # planar (z) depth: the whole ground of instance 1 is 3 m away
    assert np.allclose(d[1][seg[1] == 1], 3.0, atol=1e-3)
    r, g, b = rgb[0][c]
    assert r > 2 * g and r > 2 * b


def test_distorted_plane_depth(renderer):
    """Every pixel of a fronto-parallel plane has the same z-depth, however
    distorted the lens: checks the proxy + resampling geometry."""
    _fresh(renderer)
    s, _ = _scene(renderer, cube_x=(30.0, 30.0))
    W, H = 96, 54
    K = PI_K * np.array([[W / 4608], [H / 2592], [1]])
    s.camera_sensor("pi", W, H, K, dist=PI_DIST, instances=2,
                    local_pose=look_at((0, 0, 1.5), (0, 0, 0), up=(0, 1, 0)),
                    near=0.05, far=10.0)
    with pytest.warns(UserWarning, match="fold"):
        out = renderer.render("pi", outputs=("depth", "segmentation"))
    d = out["depth"].cpu().numpy()
    assert (out["segmentation"].cpu().numpy() == 1).all()
    assert np.allclose(d, 1.5, atol=2e-3)


def test_instance_materials_and_texture_tint(renderer):
    """A per-instance tint multiplies a textured material: with tints t_k,
    each pixel is t_k D + S (D the diffuse, S the specular layer, which
    does not scale), so (I1 - I2) / (I0 - I2) = (t1 - t2) / (t0 - t2)."""
    _fresh(renderer)
    s, rm = _scene(renderer, textured=True, instances=3,
                   cube_x=(30.0, 30.0, 30.0))
    tint = rm.material_base_color.copy()
    srgb = np.array([1.0, 0.6, 0.3])
    tint[:, 1] = srgb[:, None]
    rm.set_materials(base_color=tint)
    W, H = 32, 32
    s.camera_sensor("top", W, H, _pinhole(W, H, 30.0), instances=3,
                    local_pose=look_at((0, 0, 3.0), (0, 0, 0), up=(0, 1, 0)))
    lin = renderer.render("top", outputs=("linear",), spp=64)["linear"]
    lin = lin.mean(-1).cpu().numpy()
    t = np.where(srgb <= 0.04045, srgb / 12.92,
                 ((srgb + 0.055) / 1.055) ** 2.4)
    bright = (lin[0] - lin[2]) > 0.5 * (lin[0] - lin[2]).max()
    ratio = ((lin[1] - lin[2]) / (lin[0] - lin[2]))[bright]
    assert np.median(ratio) == pytest.approx((t[1] - t[2]) / (t[0] - t[2]),
                                             abs=0.03)
    # the checker shows
    assert bright.mean() > 0.2 and (~bright).mean() > 0.2


def test_depth_of_field_blurs(renderer):
    """A wide aperture focused far from the cube widens the band of mixed
    cube / ground pixels along its outline."""
    _fresh(renderer)
    s, _ = _scene(renderer, cube_x=(0.0, 0.0))
    W, H = 48, 48
    s.camera_sensor("cam", W, H, _pinhole(W, H, 40.0), instances=2,
                    local_pose=look_at((0, 0, 3.0), (0, 0, 0), up=(0, 1, 0)),
                    aperture_radius=[0.0, 0.05], focus_distance=[2.0, 0.5])
    lin = renderer.render("cam", outputs=("linear",), spp=64)["linear"]
    lin = lin.cpu().numpy()
    red = lin[..., 0] - lin[..., 1]                     # cube ~1, ground ~0
    frac = red / red[:, H // 2, W // 2][:, None, None]
    band = ((frac > 0.2) & (frac < 0.8)).sum((1, 2))
    assert band[1] > 3 * band[0]


def test_matches_mitsuba():
    """Same scene, same distorted camera: Cycles' resampled proxy and
    Mitsuba's exact rays agree on depth and labels."""
    pytest.importorskip("mitsuba")
    from rillfacet.render.mitsuba import MitsubaRenderer
    W, H = 96, 54
    K = PI_K * np.array([[W / 4608], [H / 2592], [1]])
    pose = look_at((1.2, -1.4, 1.6), (0, 0, 0.4), up=(0, 0, 1))
    outs = []
    for r in (CyclesRenderer(spp=1), MitsubaRenderer(spp=1)):
        s, _ = _scene(r, cube_x=(0.0, 0.4))
        s.camera_sensor("pi", W, H, K, dist=PI_DIST, instances=2,
                        local_pose=pose, near=0.05, far=10.0)
        with pytest.warns(UserWarning, match="fold"):
            outs.append(r.render("pi", outputs=("depth", "segmentation")))
        if hasattr(r, "close"):
            r.close()
    (dc, sc), (dm, sm) = [(o["depth"].cpu().numpy(),
                           o["segmentation"].cpu().numpy()) for o in outs]
    assert (sc == sm).mean() > 0.97
    both = (sc == sm) & (sc >= 0)
    assert np.median(np.abs(dc - dm)[both]) < 2e-3
