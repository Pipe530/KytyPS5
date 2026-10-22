#version 450

// FidelityFX Super Resolution 1.0 RCAS (robust contrast adaptive sharpening), fragment-shader port.
// Based on AMD FidelityFX FSR 1.0 (ffx_fsr1.h), MIT license, Copyright (c) 2021 Advanced Micro Devices.

layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D source;
layout(push_constant) uniform Params {
	vec2  in_size;
	vec2  out_size;
	float sharpness; // 2^-stops; 1.0 is maximum sharpening
};

const float RCAS_LIMIT = 0.25 - (1.0 / 16.0);

vec3 Fetch(ivec2 p) {
	return texelFetch(source, clamp(p, ivec2(0), ivec2(in_size) - 1), 0).rgb;
}

void main() {
	ivec2 sp = ivec2(gl_FragCoord.xy);
	vec3  b  = Fetch(sp + ivec2(0, -1));
	vec3  d  = Fetch(sp + ivec2(-1, 0));
	vec3  e  = Fetch(sp);
	vec3  f  = Fetch(sp + ivec2(1, 0));
	vec3  h  = Fetch(sp + ivec2(0, 1));

	vec3 mn4 = min(min(b, d), min(f, h));
	vec3 mx4 = max(max(b, d), max(f, h));
	const float eps = 1.0 / 65536.0;

	vec3 hit_min = min(mn4, e) / (4.0 * max(mx4, eps));
	vec3 hit_max = (1.0 - max(mx4, e)) / (4.0 * mn4 - 4.0 - eps);
	vec3 lobes   = max(-hit_min, hit_max);
	float lobe   = max(-RCAS_LIMIT, min(max(lobes.r, max(lobes.g, lobes.b)), 0.0)) * sharpness;

	float rcp_l = 1.0 / (4.0 * lobe + 1.0);
	color       = vec4((lobe * (b + d + f + h) + e) * rcp_l, 1.0);
}
