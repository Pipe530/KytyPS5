#version 450

// FXAA 3.11 (Timothy Lottes, NVIDIA) quality-style port using luma computed from RGB.
// Runs on the game image before FSR upscaling.

layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D source;
layout(push_constant) uniform Params {
	vec2  in_size;
	vec2  out_size;
	float unused;
};

const float EDGE_THRESHOLD     = 0.0625;
const float EDGE_THRESHOLD_MIN = 0.0312;
const float SUBPIX_QUALITY     = 1.0;
const int   ITERATIONS         = 12;

float StepSize(int i) {
	if (i < 5) return 1.0;
	if (i == 5) return 1.5;
	if (i < 10) return 2.0;
	if (i == 10) return 4.0;
	return 8.0;
}

float Luma(vec3 c) {
	return dot(c, vec3(0.299, 0.587, 0.114));
}

float LumaAt(vec2 p) {
	return Luma(textureLod(source, p, 0.0).rgb);
}

void main() {
	vec2 rcp   = 1.0 / in_size;
	vec2 pos_m = gl_FragCoord.xy * rcp;

	vec3  rgb_m  = textureLod(source, pos_m, 0.0).rgb;
	float luma_m = Luma(rgb_m);
	float luma_s = LumaAt(pos_m + vec2(0.0, 1.0) * rcp);
	float luma_e = LumaAt(pos_m + vec2(1.0, 0.0) * rcp);
	float luma_n = LumaAt(pos_m + vec2(0.0, -1.0) * rcp);
	float luma_w = LumaAt(pos_m + vec2(-1.0, 0.0) * rcp);

	float range_max = max(max(luma_n, luma_w), max(luma_e, max(luma_s, luma_m)));
	float range_min = min(min(luma_n, luma_w), min(luma_e, min(luma_s, luma_m)));
	float range     = range_max - range_min;
	if (range < max(EDGE_THRESHOLD_MIN, range_max * EDGE_THRESHOLD)) {
		color = vec4(rgb_m, 1.0);
		return;
	}

	float luma_nw = LumaAt(pos_m + vec2(-1.0, -1.0) * rcp);
	float luma_ne = LumaAt(pos_m + vec2(1.0, -1.0) * rcp);
	float luma_sw = LumaAt(pos_m + vec2(-1.0, 1.0) * rcp);
	float luma_se = LumaAt(pos_m + vec2(1.0, 1.0) * rcp);

	float luma_ns      = luma_n + luma_s;
	float luma_we      = luma_w + luma_e;
	float subpix_nswe  = luma_ns + luma_we;
	float edge_horz1   = -2.0 * luma_m + luma_ns;
	float edge_vert1   = -2.0 * luma_m + luma_we;
	float luma_nese    = luma_ne + luma_se;
	float luma_nwne    = luma_nw + luma_ne;
	float edge_horz2   = -2.0 * luma_e + luma_nese;
	float edge_vert2   = -2.0 * luma_n + luma_nwne;
	float luma_nwsw    = luma_nw + luma_sw;
	float luma_swse    = luma_sw + luma_se;
	float edge_horz3   = -2.0 * luma_w + luma_nwsw;
	float edge_vert3   = -2.0 * luma_s + luma_swse;
	float edge_horz    = abs(edge_horz3) + (abs(edge_horz1) * 2.0 + abs(edge_horz2));
	float edge_vert    = abs(edge_vert3) + (abs(edge_vert1) * 2.0 + abs(edge_vert2));
	float length_sign  = rcp.x;
	bool  horz_span    = edge_horz >= edge_vert;
	float subpix_a     = subpix_nswe * 2.0 + (luma_nwsw + luma_nese);

	if (!horz_span) {
		luma_n = luma_w;
		luma_s = luma_e;
	} else {
		length_sign = rcp.y;
	}
	float subpix_b = subpix_a * (1.0 / 12.0) - luma_m;

	float gradient_n = luma_n - luma_m;
	float gradient_s = luma_s - luma_m;
	float luma_nn    = luma_n + luma_m;
	float luma_ss    = luma_s + luma_m;
	bool  pair_n     = abs(gradient_n) >= abs(gradient_s);
	float gradient   = max(abs(gradient_n), abs(gradient_s));
	if (pair_n) length_sign = -length_sign;
	float subpix_c = clamp(abs(subpix_b) / range, 0.0, 1.0);

	vec2 pos_b  = pos_m;
	vec2 off_np = horz_span ? vec2(rcp.x, 0.0) : vec2(0.0, rcp.y);
	if (!horz_span) pos_b.x += length_sign * 0.5;
	else pos_b.y += length_sign * 0.5;

	vec2 pos_n = pos_b - off_np * StepSize(0);
	vec2 pos_p = pos_b + off_np * StepSize(0);
	float subpix_d = -2.0 * subpix_c + 3.0;
	float subpix_e = subpix_c * subpix_c;
	float luma_end_n = LumaAt(pos_n);
	if (!pair_n) luma_nn = luma_ss;
	float gradient_scaled = gradient * 0.25;
	float luma_mm         = luma_m - luma_nn * 0.5;
	bool  luma_mm_lt_zero = luma_mm < 0.0;
	luma_end_n -= luma_nn * 0.5;
	float luma_end_p = LumaAt(pos_p) - luma_nn * 0.5;
	bool  done_n     = abs(luma_end_n) >= gradient_scaled;
	bool  done_p     = abs(luma_end_p) >= gradient_scaled;
	if (!done_n) pos_n -= off_np * StepSize(1);
	if (!done_p) pos_p += off_np * StepSize(1);

	for (int i = 2; i < ITERATIONS && (!done_n || !done_p); i++) {
		if (!done_n) luma_end_n = LumaAt(pos_n) - luma_nn * 0.5;
		if (!done_p) luma_end_p = LumaAt(pos_p) - luma_nn * 0.5;
		done_n = abs(luma_end_n) >= gradient_scaled;
		done_p = abs(luma_end_p) >= gradient_scaled;
		if (!done_n) pos_n -= off_np * StepSize(i);
		if (!done_p) pos_p += off_np * StepSize(i);
	}

	float dst_n = pos_m.x - pos_n.x;
	float dst_p = pos_p.x - pos_m.x;
	if (!horz_span) {
		dst_n = pos_m.y - pos_n.y;
		dst_p = pos_p.y - pos_m.y;
	}
	bool  good_span_n = (luma_end_n < 0.0) != luma_mm_lt_zero;
	bool  good_span_p = (luma_end_p < 0.0) != luma_mm_lt_zero;
	float span_length_rcp = 1.0 / (dst_p + dst_n);
	bool  direction_n     = dst_n < dst_p;
	float dst             = min(dst_n, dst_p);
	bool  good_span       = direction_n ? good_span_n : good_span_p;
	float subpix_f        = subpix_d * subpix_e;
	float subpix_g        = subpix_f * subpix_f;
	float pixel_offset    = dst * (-span_length_rcp) + 0.5;
	float subpix_h        = subpix_g * SUBPIX_QUALITY;
	float pixel_offset_good = good_span ? pixel_offset : 0.0;
	float pixel_offset_final = max(pixel_offset_good, subpix_h);
	if (!horz_span) pos_m.x += pixel_offset_final * length_sign;
	else pos_m.y += pixel_offset_final * length_sign;

	color = vec4(textureLod(source, pos_m, 0.0).rgb, 1.0);
}
