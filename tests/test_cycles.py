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


def test_appearance_changes_between_renders(renderer):
    """New per-instance materials reach Cycles' persistent session (object
    attributes written from Python must be tagged)."""
    _fresh(renderer)
    s, rm = _scene(renderer)
    W, H = 16, 16
    s.camera_sensor("top", W, H, _pinhole(W, H, 10.0), instances=2,
                    local_pose=look_at((0, 0, 3.0), (0, 0, 0), up=(0, 1, 0)))
    c = (0, H // 2, W // 2)
    r0, g0, b0 = renderer.render("top")["rgb"][c].tolist()
    col = rm.material_base_color.copy()
    col[:, 0] = (0.1, 0.1, 0.9)
    rm.set_materials(base_color=col)
    r1, g1, b1 = renderer.render("top")["rgb"][c].tolist()
    assert r0 > 2 * b0 and b1 > 2 * r1


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


def test_vignetting_and_noise(renderer):
    """Cycles' radiance goes through rillfacet's shared sensor stage."""
    from rillfacet.render.sensor import radius2
    _fresh(renderer)
    s, _ = _scene(renderer, instances=3, cube_x=(30.0, 30.0, 30.0))
    W, H = 48, 36
    K = _pinhole(W, H, 30.0)
    s.camera_sensor("top", W, H, K, instances=3,
                    local_pose=look_at((0, 0, 3.0), (0, 0, 0), up=(0, 1, 0)),
                    vignetting=[0.0, 0.5, 0.0], read_noise=[0.0, 0.0, 0.05])
    lin = renderer.render("top", outputs=("linear",), spp=16)["linear"]
    lin = lin.mean(-1).cpu()
    r2 = radius2(K, W, H)
    ratio = lin[1] / lin[0]
    corner = r2 > 0.7
    assert float(ratio[r2 < 0.05].mean()) == pytest.approx(1.0, abs=0.03)
    assert float(ratio[corner].mean()) == pytest.approx(
        float((1 - 0.5 * r2[corner]).mean()), abs=0.03)
    extra = float(lin[2].var() - lin[0].var())
    assert extra == pytest.approx(0.05 ** 2 / 3, rel=0.25)


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


# -- solid (3D) textures --------------------------------------------------------

def _osl_eval(r, points, rows):
    """``solid_texture.osl`` evaluated by Cycles at body-local ``points``
    ``[n, T * T, 3]`` (in the plane z = const per n) with rows ``[n, K]``:
    ``(h, rough, linear rgb)``, each ``[n, T * T(, 3)]``. One object per
    row, a T x T pixel tile of an orthographic view, emitting the shader's
    outputs at the pixel centers."""
    import os
    from rillfacet.solid_texture import linearize
    n, TT, _ = points.shape
    T, G = int(round(TT ** 0.5)), int(np.ceil(np.sqrt(n)))
    px = 0.01                                          # m per pixel
    sc = bpy.data.scenes.new("rf_test_osl")
    blocks = []
    try:
        sc.render.engine = "CYCLES"
        c = sc.cycles
        c.device, c.shading_system = r._device, True
        c.samples, c.filter_width, c.max_bounces = 1, 0.01, 0
        c.use_denoising = c.use_adaptive_sampling = False
        sc.render.resolution_x = sc.render.resolution_y = G * T
        sc.render.resolution_percentage = 100
        sc.render.image_settings.file_format = "OPEN_EXR"
        sc.render.image_settings.color_depth = "32"
        sc.view_settings.view_transform = "Standard"
        cam = bpy.data.cameras.new("rf_test_cam")
        cam.type, cam.ortho_scale = "ORTHO", G * T * px
        cob = bpy.data.objects.new("rf_test_cam", cam)
        cob.location = (G * T * px / 2, G * T * px / 2, 5.0)
        sc.collection.objects.link(cob)
        sc.camera = cob
        sc.world = bpy.data.worlds.new("rf_test_world")
        mat = bpy.data.materials.new("rf_test_mat")
        nt = mat.node_tree
        nt.nodes.clear()
        st = r._solid_texture_node(nt, 0, False)
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(em.outputs[0],
                     nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
        blocks += [cam, cob, sc.world, mat]
        lin = linearize(rows).reshape(n, 8, 4)
        lo = points[:, :, :2].min(1)                   # tile corner, body-local
        for k in range(n):
            gx, gy = k % G, k // G
            z = float(points[k, 0, 2])
            x0, y0 = lo[k] - 0.5 * px
            me = bpy.data.meshes.new(f"rf_test_q{k}")
            s = T * px
            me.from_pydata([(x0, y0, z), (x0 + s, y0, z), (x0 + s, y0 + s, z),
                            (x0, y0 + s, z)], [], [(0, 1, 2, 3)])
            me.materials.append(mat)
            ob = bpy.data.objects.new(f"rf_test_q{k}", me)
            ob.location = (gx * s - x0, gy * s - y0, -z)
            for j in range(8):
                ob[f"rf_s0_{j}"] = lin[k, j].tolist()
            sc.collection.objects.link(ob)
            blocks += [ob, me]
        comb = nt.nodes.new("ShaderNodeCombineXYZ")
        nt.links.new(st.outputs["Height"], comb.inputs[0])
        nt.links.new(st.outputs["Roughness"], comb.inputs[1])
        imgs = []
        for out in (st.outputs["Color"], comb.outputs[0]):
            nt.links.new(out, em.inputs["Color"])
            sc.render.filepath = os.path.join(r._tmp, "osl_test.exr")
            bpy.ops.render.render(write_still=True, scene=sc.name)
            img = bpy.data.images.load(sc.render.filepath, check_existing=False)
            img.colorspace_settings.name = "Non-Color"
            a = np.empty(G * T * G * T * 4, np.float32)
            img.pixels.foreach_get(a)
            bpy.data.images.remove(img)
            # row 0 = bottom: tile (gx, gy), pixel (u, v) at y-up order
            a = a.reshape(G, T, G, T, 4).transpose(0, 2, 1, 3, 4)
            imgs.append(a.reshape(G * G, T * T, 4)[:n, :, :3])
        return imgs[1][..., 0], imgs[1][..., 1], imgs[0]
    finally:
        # objects first (removing their data would remove them)
        blocks.sort(key=lambda b: not isinstance(b, bpy.types.Object))
        for b in blocks:
            {bpy.types.Object: bpy.data.objects, bpy.types.Mesh: bpy.data.meshes,
             bpy.types.Camera: bpy.data.cameras, bpy.types.World: bpy.data.worlds,
             bpy.types.Material: bpy.data.materials}[type(b)].remove(b)
        bpy.data.scenes.remove(sc)


def test_solid_texture_osl_matches_the_reference(renderer):
    """The OSL port evaluates the same functions as the torch reference
    (rillfacet.solid_texture.evaluate), every family, random rows and
    points (in body-local planes z = const)."""
    from rillfacet import solid_texture as sx
    rng = np.random.default_rng(0)
    n, T, px = 64, 16, 0.01
    rows = sx.sample_params(rng, n, families=sx.FAMILIES)
    rows[:, sx.PARAM["rough0"]], rows[:, sx.PARAM["rough1"]] = 0.2, 0.9
    corner = rng.uniform(-0.3, 0.1, (n, 1, 2))
    u, v = np.meshgrid(np.arange(T), np.arange(T), indexing="xy")
    uv = (np.stack([u, v], -1).reshape(1, -1, 2) + 0.5) * px
    pts = np.concatenate([corner + uv, np.broadcast_to(
        rng.uniform(-0.3, 0.3, (n, 1, 1)), (n, T * T, 1))], -1)
    h, rough, rgb = _osl_eval(renderer, pts.astype(np.float32), rows)
    got = torch.as_tensor(np.concatenate([h[..., None], rgb], -1)).reshape(-1, 4)
    pt = torch.as_tensor(pts, dtype=torch.float32).reshape(-1, 3)
    rt = torch.as_tensor(np.repeat(rows, T * T, 0))
    hr, cr = sx.evaluate(pt, rt)
    want = torch.cat([hr[:, None], cr], -1)
    err = (got - want).abs().max(-1).values
    fam = rt[:, sx.PARAM["family"]]
    for f, name in enumerate(sx.FAMILIES):
        e = err[fam == f]
        assert len(e), name
        # granite / checker: rare ties at cell boundaries pick another cell
        assert float((e > 1e-3).float().mean()) < 2e-3, name
        assert float(e.median()) < 1e-5, name
    assert np.allclose(rough, 0.2 + 0.7 * h, atol=1e-5)


def _solid_scene(renderer, solid, *, bump=False, instances=2,
                 cube_x=(0.0, 0.0)):
    """``_scene`` with both slots solid-texture slots (or not)."""
    bv, bt = _box()
    gv = np.array([[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]], np.float32)
    gt = np.array([[0, 1, 2], [0, 2, 3]]) + len(bv)
    s = Scene(renderer)
    rm = s.rigid_mesh(
        "world", np.concatenate([bv, gv]),
        np.r_[np.zeros(len(bv), int), np.ones(4, int)], np.concatenate([bt, gt]),
        np.array([0, len(bt), len(bt) + 2]), instances=instances,
        body_names=["cube", "ground"],
        materials=[Material(n, solid_texture=solid, bump=bump and solid)
                   for n in ("a", "b")],
        triangle_material=np.r_[np.zeros(len(bt), int), np.ones(2, int)])
    poses = np.zeros((instances, 2, 7), np.float32)
    poses[..., 6] = 1.0
    poses[:, 0, 0] = cube_x[:instances]
    rm.set_poses(poses)
    return s, rm


def test_plain_solid_rows_render_like_ordinary_materials(renderer):
    """Solid-texture slots with plain rows of each env's color and
    roughness render the image of ordinary materials with those values (the
    (env, slot) -> row mapping, sRGB, roughness, metallic; OSL vs SVM)."""
    from rillfacet import solid_texture as sx
    rng = np.random.default_rng(1)
    col = rng.uniform(0, 1, (2, 2, 3))
    rough = rng.uniform(0.2, 1, (2, 2))
    metal = rng.uniform(0, 1, (2, 2))
    imgs = []
    for solid in (False, True):
        _fresh(renderer)
        s, rm = _solid_scene(renderer, solid)
        rm.set_materials(base_color=col, roughness=rough, metallic=metal)
        if solid:
            rm.set_materials(solid_texture=np.stack([[
                sx.make_params("plain", colors=(col[e, k],),
                               roughness=rough[e, k])
                for k in range(2)] for e in range(2)]))
        W, H = 48, 36
        s.camera_sensor("cam", W, H, _pinhole(W, H, 30.0), instances=2,
                        local_pose=look_at((1.5, -2.0, 2.5), (0, 0, 0.3),
                                           up=(0, 0, 1)))
        renderer._seed = 0                       # the same samples
        imgs.append(renderer.render("cam", outputs=("linear",),
                                    spp=16)["linear"])
    a, b = imgs
    assert float((a - b).abs().mean()) < 2e-3
    assert float((a - b).abs().amax(-1).quantile(0.99)) < 2e-2


def test_solid_texture_bump_follows_the_height(renderer):
    """A bumped white floor, rotated and moved, under the sun: the radiance
    ratio bumped / flat follows the cosine ratio predicted from the
    reference's height gradient at the body-local hit points."""
    from rillfacet import solid_texture as sx
    from rillfacet.render.mitsuba import _qrot
    _fresh(renderer)
    s, rm = _solid_scene(renderer, True, bump=True, instances=1,
                         cube_x=(30.0,))
    ang = 0.7
    q = np.array([0, 0, np.sin(ang / 2), np.cos(ang / 2)], np.float32)
    shift = np.array([0.3, -0.2, 0.0], np.float32)
    poses = rm.poses.copy() if not hasattr(rm.poses, "detach") \
        else rm.poses.cpu().numpy().copy()
    poses = poses.reshape(1, 2, 7)
    poses[0, 1, :3], poses[0, 1, 3:] = shift, q
    rm.set_poses(poses)
    sun = np.array([0.6, 0.3, 0.75])
    s.light("sun", "directional", direction=-sun, intensity=3.0)
    W = H = 96
    f = 120.0
    pose = look_at((0, 0, 1.2), (0, 0, 0), up=(0, 1, 0))
    s.camera_sensor("cam", W, H, _pinhole(W, H, f), instances=1,
                    local_pose=pose)
    amp, scale, eps = 0.02, 3.0, 0.02
    row = sx.make_params("stone", colors=((0.7, 0.7, 0.7),), scale=scale,
                         offset=(11, 23, 5), roughness=1.0, bump=amp, eps=eps)
    out = []
    for a in (amp, 0.0):
        row[sx.PARAM["bump"]] = a
        rm.set_materials(solid_texture=np.stack([row, row])[None])
        out.append(renderer.render("cam", outputs=("linear", "depth"),
                                   spp=64))
    L, L0 = (o["linear"][0].mean(-1).cpu() for o in out)
    z = out[0]["depth"][0].cpu()
    v, u = torch.meshgrid(torch.arange(float(H)), torch.arange(float(W)),
                          indexing="ij")
    xc = torch.stack([(u - (W - 1) / 2) / f * z, (v - (H - 1) / 2) / f * z,
                      z], -1)
    cp = torch.as_tensor(pose, dtype=torch.float32)
    pw = (cp[:3] + _qrot(cp[3:].expand(xc.shape[:-1] + (4,)), xc)).reshape(-1, 3)
    qi = torch.as_tensor([-q[0], -q[1], -q[2], q[3]])
    p = _qrot(qi.expand(len(pw), 4), pw - torch.as_tensor(shift))   # body-local
    rows = torch.as_tensor(row).expand(len(p), sx.K)
    e = eps / scale
    h0, _ = sx.evaluate(p, rows)
    gx = (sx.evaluate(p + torch.tensor([e, 0, 0]), rows)[0] - h0) / e
    gy = (sx.evaluate(p + torch.tensor([0, e, 0]), rows)[0] - h0) / e
    # body-local gradient -> world
    g = _qrot(torch.as_tensor(q).expand(len(p), 4),
              torch.stack([gx, gy, torch.zeros_like(gx)], -1))
    sn = torch.as_tensor(sun / np.linalg.norm(sun), dtype=torch.float32)

    def cos_ratio(sign):
        n = torch.stack([-sign * amp * g[:, 0], -sign * amp * g[:, 1],
                         torch.ones_like(gx)], -1)
        n = n / n.norm(dim=-1, keepdim=True)
        return (n @ sn).clamp(min=0) / sn[2]

    ok = (z > 0).reshape(-1) & (L0.reshape(-1) > 0.05)
    ratio = (L / L0.clamp(min=1e-4)).reshape(-1)[ok]
    corr = lambda x: float(torch.corrcoef(torch.stack([ratio, x[ok]]))[0, 1])  # noqa: E731
    assert float(ratio.std()) > 0.03                     # visibly bumped
    assert corr(cos_ratio(1.0)) > 0.85
    assert corr(cos_ratio(-1.0)) < 0.0
