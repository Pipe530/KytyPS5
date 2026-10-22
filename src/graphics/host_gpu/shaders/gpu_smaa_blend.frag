#version 450

// SMAA 1x neighborhood blending (Jimenez et al., MIT license).

layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D source;
layout(binding = 1) uniform sampler2D blend_tex;
layout(push_constant) uniform Params {
	vec2  in_size;
	vec2  out_size;
	float unused;
};

void main() {
	vec2 rcp      = 1.0 / in_size;
	vec2 texcoord = gl_FragCoord.xy * rcp;
	vec4 offset   = rcp.xyxy * vec4(1.0, 0.0, 0.0, 1.0) + texcoord.xyxy;

	vec4 a;
	a.x  = textureLod(blend_tex, offset.xy, 0.0).a;
	a.y  = textureLod(blend_tex, offset.zw, 0.0).g;
	a.wz = textureLod(blend_tex, texcoord, 0.0).xz;

	if (dot(a, vec4(1.0)) < 1e-5) {
		color = vec4(textureLod(source, texcoord, 0.0).rgb, 1.0);
		return;
	}

	bool h = max(a.x, a.z) > max(a.y, a.w);

	vec4 blending_offset = vec4(0.0, a.y, 0.0, a.w);
	vec2 blending_weight = a.yw;
	if (h) {
		blending_offset = vec4(a.x, 0.0, a.z, 0.0);
		blending_weight = a.xz;
	}
	blending_weight /= dot(blending_weight, vec2(1.0));

	vec4 blending_coord = blending_offset * vec4(rcp.xy, -rcp.xy) + texcoord.xyxy;

	vec3 rgb = blending_weight.x * textureLod(source, blending_coord.xy, 0.0).rgb;
	rgb += blending_weight.y * textureLod(source, blending_coord.zw, 0.0).rgb;
	color = vec4(rgb, 1.0);
}
