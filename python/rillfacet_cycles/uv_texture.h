// Procedural UV (2D) textures for Cycles (OSL): a line-by-line port of
// rillfacet's python/rillfacet/uv_texture.py (the spec and the torch
// reference; its docstring has the parameter layout) by way of its WGSL
// port (shaders/common/uv_texture.wgsl), evaluated at the metric UVs.
// Keep them in sync: the tests compare this shader against the reference.
// Needs solid_texture.h (hash, noise, the row accessors); included by
// texture.osl. Unfiltered (footprint 0), like Mitsuba. A UV row's family is
// stored plus UT_FAMILY_OFFSET.

#define UT_FAMILY_OFFSET 64.0
#define UT_OFFSET 2
#define UT_ROT 4
#define UT_SEED 6
#define UT_B 7
#define UT_PI 3.14159265358979
#define UT_SQRT3 1.7320508075688772

float ut_fract(float x) { return x - floor(x); }
int ut_cell(float x) { return int(floor(x)); }
int ut_seed(float S[32]) { return int(floor(S[UT_SEED])); }

void ut_hash(float x, float y, int seed, output int hx, output int hy,
             output int hz)
{
  pcg3d(ut_cell(x), ut_cell(y), seed, hx, hy, hz);
}

float ut_cover(float d) { return d >= 0.0 ? 1.0 : 0.0; }

float ut_pulse(float x, float lo, float width)
{
  return ut_fract(x - lo) < width ? 1.0 : 0.0;
}

float ut_round_box(float px, float py, float hx, float hy, float r)
{
  float qx = abs(px) - hx + r;
  float qy = abs(py) - hy + r;
  float mx = max(qx, 0.0), my = max(qy, 0.0);
  return sqrt(mx * mx + my * my) + min(max(qx, qy), 0.0) - r;
}

void ut_stripes(float S[32], float qx, float qy, output float h,
                output color c)
{
  float a0 = S[ST_A], a1 = S[ST_A + 1], a2 = S[ST_A + 2], b0 = S[UT_B];
  float x = qx + a1 * sin(qy * a2);
  float s1 = ut_pulse(x, 0.0, a0);
  float s2 = ut_pulse(x, a0 + 0.5 * (1.0 - a0) - 0.5 * b0, b0);
  c = st_lerp3(st_lerp3(sp3(S, ST_C0), sp3(S, ST_C1), s1), sp3(S, ST_C2), s2);
  h = clamp(s1 + 0.5 * s2, 0.0, 1.0);
}

void ut_tiles(float S[32], float qx, float qy, output float h, output color c)
{
  float a0 = S[ST_A], a1 = S[ST_A + 1], a2 = S[ST_A + 2], ar = S[UT_B];
  float y = qy / ar;
  float j = floor(y);
  float x = qx + a1 * j;
  float t = ut_pulse(x, a0, 1.0 - 2.0 * a0)
            * ut_pulse(y, a0 / ar, 1.0 - 2.0 * a0 / ar);
  int hx, hy, hz;
  ut_hash(x, y, ut_seed(S), hx, hy, hz);
  color tile = st_lerp3(sp3(S, ST_C0), sp3(S, ST_C2), a2 * nz_unit(hx));
  h = t;
  c = st_lerp3(sp3(S, ST_C1), tile, t);
}

// A disc per cell (rows shifted by stagger): its coverage and hash word.
float ut_disc(float S[32], float qx, float qy, float rad, float jitter,
              float rvar, float stagger, output int hz)
{
  float y = qy;
  float x = qx + stagger * floor(y);
  int hx, hy;
  ut_hash(x, y, ut_seed(S), hx, hy, hz);
  float rr = rad * (1.0 - rvar * nz_unit(hz));
  float room = 0.5 - rr;
  float cx = 0.5 + jitter * room * (2.0 * nz_unit(hx) - 1.0);
  float cy = 0.5 + jitter * room * (2.0 * nz_unit(hy) - 1.0);
  float dx = ut_fract(x) - cx, dy = ut_fract(y) - cy;
  return ut_cover(rr - sqrt(dx * dx + dy * dy));
}

void ut_dots(float S[32], float qx, float qy, output float h, output color c)
{
  float a0 = S[ST_A], a1 = S[ST_A + 1], a2 = S[ST_A + 2], a3 = S[ST_A + 3];
  int hz;
  float cov = ut_disc(S, qx, qy, a0, a1, a2, S[UT_B], hz);
  color dot = nz_unit(hz * 668265261) < a3 ? sp3(S, ST_C2) : sp3(S, ST_C1);
  h = cov;
  c = st_lerp3(sp3(S, ST_C0), dot, cov);
}

void ut_perforated(float S[32], float qx, float qy, output float h,
                   output color c)
{
  int hz;
  float cov = ut_disc(S, qx, qy, S[ST_A], 0.0, 0.0, S[UT_B], hz);
  h = 1.0 - cov;
  c = st_lerp3(sp3(S, ST_C0), sp3(S, ST_C1), cov);
}

void ut_hexagons(float S[32], float qx, float qy, output float h,
                 output color c)
{
  float a0 = S[ST_A], a1 = S[ST_A + 1];
  float hy = 0.5 * UT_SQRT3;
  float ax = qx - floor(qx) - 0.5;
  float ay = qy - UT_SQRT3 * floor(qy * (1.0 / UT_SQRT3)) - hy;
  float bxs = qx - 0.5, bys = qy - hy;
  float bx = bxs - floor(bxs) - 0.5;
  float by = bys - UT_SQRT3 * floor(bys * (1.0 / UT_SQRT3)) - hy;
  int near_a = ax * ax + ay * ay < bx * bx + by * by;
  float gx = near_a ? ax : bx, gy = near_a ? ay : by;
  float cx = qx - gx, cy = qy - gy;
  int h0, h1, h2;
  ut_hash(cx * 2.0 + 0.5, cy * (1.0 / hy) + 0.5, ut_seed(S), h0, h1, h2);
  float d = max(abs(gx), abs(gx) * 0.5 + abs(gy) * hy);
  float t = ut_cover(0.5 - d - a0);
  color cell = st_lerp3(sp3(S, ST_C0), sp3(S, ST_C2), a1 * nz_unit(h0));
  h = t;
  c = st_lerp3(sp3(S, ST_C1), cell, t);
}

void ut_weave(float S[32], float qx, float qy, output float h, output color c)
{
  float a0 = S[ST_A], a1 = S[ST_A + 1], a2 = S[ST_A + 2];
  float i = floor(qx), j = floor(qy);
  float fx = qx - i, fy = qy - j;
  float s = i + j;
  int warp = (s - 2.0 * floor(s * 0.5)) < 0.5;
  float across = (warp ? abs(fx - 0.5) : abs(fy - 0.5)) * 2.0;
  float along = warp ? fy : fx;
  float width = 1.0 - a0;
  float on = ut_cover((width - across) * 0.5);
  float t = clamp(across / width, 0.0, 1.0);
  float prof = 1.0 - t * t;
  float bulge = 0.75 + 0.25 * sin(along * UT_PI);
  float z = S[UT_SEED] * (1.0 / 65536.0);
  float oa = warp ? qx : qy, ob = warp ? qy : qx;
  float fib = perlin(point(oa * 9.0, ob * 0.7, z)) * a1;
  h = clamp((prof * bulge + fib) * on + 0.15 * (1.0 - on), 0.0, 1.0);
  color c0 = sp3(S, ST_C0), c1 = sp3(S, ST_C1);
  color col = st_lerp3(warp ? c1 : c0, warp ? c0 : c1, on);
  c = col * ((1.0 - a2) + a2 * h);
}

void ut_brushed(float S[32], float qx, float qy, output float h,
                output color c)
{
  float a0 = S[ST_A], a1 = S[ST_A + 1], a2 = S[ST_A + 2];
  float z = S[UT_SEED] * (1.0 / 65536.0);
  float n = fbm(point(qx * a1, qy * a2, z), 4, 0);
  h = clamp(0.5 + n, 0.0, 1.0);
  c = st_lerp3(sp3(S, ST_C0), sp3(S, ST_C1), a0 * h);
}

void ut_twill(float S[32], float qx, float qy, output float h, output color c)
{
  float a0 = S[ST_A], a1 = S[ST_A + 1], a2 = S[ST_A + 2];
  float i = floor(qx), j = floor(qy);
  float fx = qx - i, fy = qy - j;
  float s = i - j;
  float m = s - 4.0 * floor(s * 0.25);
  int warp = m < 1.5;
  float across = warp ? fx : fy;
  float fiber = 0.5 + 0.5 * sin(across * (2.0 * UT_PI) * a0);
  float edge = nz_smooth(0.0, a2, across) * nz_smooth(0.0, a2, 1.0 - across);
  float tow = warp ? 1.0 : 1.0 - a1;
  float tone = tow * (0.85 + 0.15 * fiber) * (0.55 + 0.45 * edge);
  h = clamp(tone, 0.0, 1.0);
  c = st_lerp3(sp3(S, ST_C1), sp3(S, ST_C0), tone);
}

void ut_decals(float S[32], float qx, float qy, output float h, output color c)
{
  float a0 = S[ST_A], a1 = S[ST_A + 1], a2 = S[ST_A + 2], a3 = S[ST_A + 3];
  float fx = ut_fract(qx), fy = ut_fract(qy);
  int hx, hy, hz;
  ut_hash(qx, qy, ut_seed(S), hx, hy, hz);
  float r3 = nz_unit(hx * 668265261);
  float r4 = nz_unit((hx ^ lsr(hx, 15)) * 374761393);
  float w = nz_lerp(a1, a2, nz_unit(hx));
  float hh = nz_lerp(a1, a2, nz_unit(hy));
  float cx = 0.5 + (0.5 - w) * (2.0 * r3 - 1.0);
  float cy = 0.5 + (0.5 - hh) * (2.0 * r4 - 1.0);
  int present = nz_unit(hz) < a0;
  float sd = ut_round_box(fx - cx, fy - cy, w, hh, min(a3, min(w, hh)));
  float cov = present ? ut_cover(-sd) : 0.0;
  color ink = (hz & 1) == 0 ? sp3(S, ST_C1) : sp3(S, ST_C2);
  h = cov;
  c = st_lerp3(sp3(S, ST_C0), ink, cov);
}

void ut_stickers(float S[32], float qx, float qy, output float h,
                 output color c)
{
  float a0 = S[ST_A], a1 = S[ST_A + 1], a2 = S[ST_A + 2], a3 = S[ST_A + 3];
  float b1 = S[UT_B + 1];
  float fx = ut_fract(qx), fy = ut_fract(qy);
  int hx, hy, hz;
  ut_hash(qx, qy, ut_seed(S), hx, hy, hz);
  float r3 = nz_unit(hx * 668265261);
  float r4 = nz_unit((hx ^ lsr(hx, 15)) * 374761393);
  float s = nz_lerp(a1, a2, nz_unit(hx));
  float cx = 0.5 + (0.5 - s) * (2.0 * r3 - 1.0);
  float cy = 0.5 + (0.5 - s) * (2.0 * r4 - 1.0);
  int present = nz_unit(hz) < a0;
  float sd = ut_round_box(fx - cx, fy - cy, s, s, b1 * s);
  float shape = present ? ut_cover(-sd) : 0.0;
  float fill = ut_cover(-sd - a3 * s);
  h = shape;
  c = st_lerp3(sp3(S, ST_C0), st_lerp3(sp3(S, ST_C2), sp3(S, ST_C1), fill),
               shape);
}

void ut_glyphs(float S[32], float qx, float qy, output float h, output color c)
{
  float a0 = S[ST_A], a1 = S[ST_A + 1], a2 = S[ST_A + 2];
  float b0 = S[UT_B], b1 = S[UT_B + 1];
  int seed = ut_seed(S);
  float j = floor(qy);
  float fy = qy - j;
  float gx = qx / b0;
  float gi = floor(gx);
  float col = floor(gi / b1);
  float pos = gi - col * b1;
  int hl, hl2, hl3;
  pcg3d(ut_cell(col), ut_cell(j), seed ^ 739982445, hl, hl2, hl3);
  float len = b1 * (0.35 + 0.65 * nz_unit(hl2)) - 2.0;
  int hg, hg2, hs;
  pcg3d(ut_cell(gi), ut_cell(j), seed ^ 1540483477, hg, hg2, hs);
  int show = nz_unit(hl) >= a0 && pos < len && nz_unit(hs) >= a1;
  float W = 0.7 * b0, H = 0.6;
  float px = (gx - gi) * b0 - 0.15 * b0;
  float py = fy - 0.2;
  // segments: center (x, y) and half extents in multiples of (W, H)
  float SX[9] = {0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0};
  float SY[9] = {0.0, 0.5, 1.0, 0.25, 0.75, 0.25, 0.75, 0.25, 0.75};
  float HX[9] = {0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  float HY[9] = {0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25};
  float ink = 0.0;
  for (int k = 0; k < 9; k++) {
    float sd = ut_round_box(px - SX[k] * W, py - SY[k] * H,
                            HX[k] * W + 0.5 * a2, HY[k] * H + 0.5 * a2, 0.0);
    if ((lsr(hg, k) & 1) == 1) ink = max(ink, ut_cover(-sd));
  }
  ink = show ? ink : 0.0;
  h = ink;
  c = st_lerp3(sp3(S, ST_C0), sp3(S, ST_C1), ink);
}

// (h, linear rgb) of a UV row's family at the metric uv (u, v).
void uv_texture(float S[32], float u, float v, output float h, output color c)
{
  float s = S[ST_SCALE], cr = S[UT_ROT], sr = S[UT_ROT + 1];
  float x = u * s, y = v * s;
  float qx = cr * x - sr * y + S[UT_OFFSET];
  float qy = sr * x + cr * y + S[UT_OFFSET + 1];
  int f = min(int(floor(S[ST_FAMILY] - UT_FAMILY_OFFSET + 0.5)), 11);
  if (f == 1) ut_stripes(S, qx, qy, h, c);
  else if (f == 2) ut_tiles(S, qx, qy, h, c);
  else if (f == 3) ut_dots(S, qx, qy, h, c);
  else if (f == 4) ut_hexagons(S, qx, qy, h, c);
  else if (f == 5) ut_weave(S, qx, qy, h, c);
  else if (f == 6) ut_brushed(S, qx, qy, h, c);
  else if (f == 7) ut_perforated(S, qx, qy, h, c);
  else if (f == 8) ut_twill(S, qx, qy, h, c);
  else if (f == 9) ut_decals(S, qx, qy, h, c);
  else if (f == 10) ut_glyphs(S, qx, qy, h, c);
  else if (f == 11) ut_stickers(S, qx, qy, h, c);
  else { h = 0.0; c = sp3(S, ST_C0); }
}
