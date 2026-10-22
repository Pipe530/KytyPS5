#version 450

// SMAA 1x luma edge detection (Jimenez et al., MIT license), HIGH preset.

layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D source;
layout(push_constant) uniform Params {
	vec2  in_size;
	vec2  out_size;
	float unused;
};

const float THRESHOLD = 0.1;
const float LOCAL_CONTRAST_ADAPTATION_FACTOR = 2.0;

float Luma(vec2 uv) {
	return dot(textureLod(source, uv, 0.0).rgb, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
	vec2 rcp = 1.0 / in_size;
	vec2 uv  = gl_FragCoord.xy * rcp;

	float l       = Luma(uv);
	float l_left  = Luma(uv + vec2(-1.0, 0.0) * rcp);
	float l_top   = Luma(uv + vec2(0.0, -1.0) * rcp);
	vec2  delta_a = abs(l - vec2(l_left, l_top));
	vec2  edges   = step(vec2(THRESHOLD), delta_a);
	if (dot(edges, vec2(1.0)) == 0.0) {
		color = vec4(0.0);
		return;
	}

	float l_right  = Luma(uv + vec2(1.0, 0.0) * rcp);
	float l_bottom = Luma(uv + vec2(0.0, 1.0) * rcp);
	vec2  delta_b  = abs(l - vec2(l_right, l_bottom));
	vec2  max_delta = max(delta_a, delta_b);

	float l_leftleft = Luma(uv + vec2(-2.0, 0.0) * rcp);
	float l_toptop   = Luma(uv + vec2(0.0, -2.0) * rcp);
	vec2  delta_c    = abs(vec2(l_left, l_top) - vec2(l_leftleft, l_toptop));
	max_delta        = max(max_delta, delta_c);
	float final_delta = max(max_delta.x, max_delta.y);

	edges *= step(vec2(final_delta), LOCAL_CONTRAST_ADAPTATION_FACTOR * delta_a);
	color = vec4(edges, 0.0, 1.0);
}
