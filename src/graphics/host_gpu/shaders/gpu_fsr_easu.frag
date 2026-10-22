#version 450

// FidelityFX Super Resolution 1.0 EASU (edge-adaptive spatial upsampling), fragment-shader port.
// Based on AMD FidelityFX FSR 1.0 (ffx_fsr1.h), MIT license, Copyright (c) 2021 Advanced Micro Devices.

layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D source;
layout(push_constant) uniform Params {
	vec2 in_size;
	vec2 out_size;
};

vec3 Fetch(ivec2 p) {
	return texelFetch(source, clamp(p, ivec2(0), ivec2(in_size) - 1), 0).rgb;
}

void TapF(inout vec3 acc, inout float weight, vec2 off, vec2 dir, vec2 len, float lob, float clp,
          vec3 c) {
	vec2 v;
	v.x = off.x * dir.x + off.y * dir.y;
	v.y = off.x * -dir.y + off.y * dir.x;
	v *= len;
	float d2 = min(v.x * v.x + v.y * v.y, clp);
	float wb = 0.4 * d2 - 1.0;
	float wa = lob * d2 - 1.0;
	wb *= wb;
	wa *= wa;
	wb = (25.0 / 16.0) * wb - (25.0 / 16.0 - 1.0);
	float w = wb * wa;
	acc += c * w;
	weight += w;
}

void SetF(inout vec2 dir, inout float len, float w, float lA, float lB, float lC, float lD,
          float lE) {
	float lenX = max(abs(lD - lC), abs(lC - lB));
	float dirX = lD - lB;
	dir.x += dirX * w;
	lenX = lenX > 0.0 ? clamp(abs(dirX) / lenX, 0.0, 1.0) : 0.0;
	lenX *= lenX;
	len += lenX * w;
	float lenY = max(abs(lE - lC), abs(lC - lA));
	float dirY = lE - lA;
	dir.y += dirY * w;
	lenY = lenY > 0.0 ? clamp(abs(dirY) / lenY, 0.0, 1.0) : 0.0;
	lenY *= lenY;
	len += lenY * w;
}

float Luma(vec3 c) {
	return c.b * 0.5 + (c.r * 0.5 + c.g);
}

void main() {
	vec2 ip = floor(gl_FragCoord.xy);
	vec2 pp = ip * (in_size / out_size) + (0.5 * in_size / out_size - 0.5);
	vec2 fp = floor(pp);
	pp -= fp;
	ivec2 f0 = ivec2(fp);

	//      b c
	//    e f g h
	//    i j k l
	//      n o
	vec3 b = Fetch(f0 + ivec2(0, -1));
	vec3 c = Fetch(f0 + ivec2(1, -1));
	vec3 e = Fetch(f0 + ivec2(-1, 0));
	vec3 f = Fetch(f0);
	vec3 g = Fetch(f0 + ivec2(1, 0));
	vec3 h = Fetch(f0 + ivec2(2, 0));
	vec3 i = Fetch(f0 + ivec2(-1, 1));
	vec3 j = Fetch(f0 + ivec2(0, 1));
	vec3 k = Fetch(f0 + ivec2(1, 1));
	vec3 l = Fetch(f0 + ivec2(2, 1));
	vec3 n = Fetch(f0 + ivec2(0, 2));
	vec3 o = Fetch(f0 + ivec2(1, 2));

	float bL = Luma(b), cL = Luma(c), eL = Luma(e), fL = Luma(f), gL = Luma(g), hL = Luma(h);
	float iL = Luma(i), jL = Luma(j), kL = Luma(k), lL = Luma(l), nL = Luma(n), oL = Luma(o);

	vec2  dir = vec2(0.0);
	float len = 0.0;
	SetF(dir, len, (1.0 - pp.x) * (1.0 - pp.y), bL, eL, fL, gL, jL);
	SetF(dir, len, pp.x * (1.0 - pp.y), cL, fL, gL, hL, kL);
	SetF(dir, len, (1.0 - pp.x) * pp.y, fL, iL, jL, kL, nL);
	SetF(dir, len, pp.x * pp.y, gL, jL, kL, lL, oL);

	float dirR = dot(dir, dir);
	bool  zro  = dirR < (1.0 / 32768.0);
	dirR       = zro ? 1.0 : inversesqrt(dirR);
	dir.x      = zro ? 1.0 : dir.x;
	dir *= dirR;
	len           = len * 0.5;
	len *= len;
	float stretch = dot(dir, dir) / max(abs(dir.x), abs(dir.y));
	vec2  len2    = vec2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
	float lob     = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
	float clp     = 1.0 / lob;

	vec3 min4 = min(min(f, g), min(j, k));
	vec3 max4 = max(max(f, g), max(j, k));

	vec3  acc    = vec3(0.0);
	float weight = 0.0;
	TapF(acc, weight, vec2(0, -1) - pp, dir, len2, lob, clp, b);
	TapF(acc, weight, vec2(1, -1) - pp, dir, len2, lob, clp, c);
	TapF(acc, weight, vec2(-1, 1) - pp, dir, len2, lob, clp, i);
	TapF(acc, weight, vec2(0, 1) - pp, dir, len2, lob, clp, j);
	TapF(acc, weight, vec2(0, 0) - pp, dir, len2, lob, clp, f);
	TapF(acc, weight, vec2(-1, 0) - pp, dir, len2, lob, clp, e);
	TapF(acc, weight, vec2(1, 1) - pp, dir, len2, lob, clp, k);
	TapF(acc, weight, vec2(2, 1) - pp, dir, len2, lob, clp, l);
	TapF(acc, weight, vec2(2, 0) - pp, dir, len2, lob, clp, h);
	TapF(acc, weight, vec2(1, 0) - pp, dir, len2, lob, clp, g);
	TapF(acc, weight, vec2(1, 2) - pp, dir, len2, lob, clp, o);
	TapF(acc, weight, vec2(0, 2) - pp, dir, len2, lob, clp, n);

	color = vec4(min(max4, max(min4, acc / weight)), 1.0);
}
