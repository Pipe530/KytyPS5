#version 450

// SMAA 1x blending weight calculation (Jimenez et al., MIT license), HIGH preset without
// diagonal detection.

layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D edges_tex;
layout(binding = 1) uniform sampler2D area_tex;
layout(binding = 2) uniform sampler2D search_tex;
layout(push_constant) uniform Params {
	vec2  in_size;
	vec2  out_size;
	float unused;
};

const float MAX_SEARCH_STEPS      = 16.0;
const float CORNER_ROUNDING_NORM  = 0.25;
const float AREATEX_MAX_DISTANCE  = 16.0;
const vec2  AREATEX_PIXEL_SIZE    = vec2(1.0 / 160.0, 1.0 / 560.0);
const float AREATEX_SUBTEX_SIZE   = 1.0 / 7.0;
const vec2  SEARCHTEX_SIZE        = vec2(66.0, 33.0);
const vec2  SEARCHTEX_PACKED_SIZE = vec2(64.0, 16.0);

vec4 rt_metrics;

vec2 Edges(vec2 uv) {
	return textureLod(edges_tex, uv, 0.0).rg;
}

vec2 EdgesOffset(vec2 uv, ivec2 offset) {
	return textureLod(edges_tex, uv + vec2(offset) * rt_metrics.xy, 0.0).rg;
}

float SearchLength(vec2 e, float offset) {
	vec2 scale = SEARCHTEX_SIZE * vec2(0.5, -1.0);
	vec2 bias  = SEARCHTEX_SIZE * vec2(offset, 1.0);
	scale += vec2(-1.0, 1.0);
	bias += vec2(0.5, -0.5);
	scale *= 1.0 / SEARCHTEX_PACKED_SIZE;
	bias *= 1.0 / SEARCHTEX_PACKED_SIZE;
	return textureLod(search_tex, scale * e + bias, 0.0).r;
}

float SearchXLeft(vec2 uv, float end) {
	vec2 e = vec2(0.0, 1.0);
	while (uv.x > end && e.g > 0.8281 && e.r == 0.0) {
		e  = Edges(uv);
		uv = uv - vec2(2.0, 0.0) * rt_metrics.xy;
	}
	float offset = -(255.0 / 127.0) * SearchLength(e, 0.0) + 3.25;
	return rt_metrics.x * offset + uv.x;
}

float SearchXRight(vec2 uv, float end) {
	vec2 e = vec2(0.0, 1.0);
	while (uv.x < end && e.g > 0.8281 && e.r == 0.0) {
		e  = Edges(uv);
		uv = uv + vec2(2.0, 0.0) * rt_metrics.xy;
	}
	float offset = -(255.0 / 127.0) * SearchLength(e, 0.5) + 3.25;
	return -rt_metrics.x * offset + uv.x;
}

float SearchYUp(vec2 uv, float end) {
	vec2 e = vec2(1.0, 0.0);
	while (uv.y > end && e.r > 0.8281 && e.g == 0.0) {
		e  = Edges(uv);
		uv = uv - vec2(0.0, 2.0) * rt_metrics.xy;
	}
	float offset = -(255.0 / 127.0) * SearchLength(e.gr, 0.0) + 3.25;
	return rt_metrics.y * offset + uv.y;
}

float SearchYDown(vec2 uv, float end) {
	vec2 e = vec2(1.0, 0.0);
	while (uv.y < end && e.r > 0.8281 && e.g == 0.0) {
		e  = Edges(uv);
		uv = uv + vec2(0.0, 2.0) * rt_metrics.xy;
	}
	float offset = -(255.0 / 127.0) * SearchLength(e.gr, 0.5) + 3.25;
	return -rt_metrics.y * offset + uv.y;
}

vec2 Area(vec2 dist, float e1, float e2, float offset) {
	vec2 uv = AREATEX_MAX_DISTANCE * round(4.0 * vec2(e1, e2)) + dist;
	uv      = AREATEX_PIXEL_SIZE * uv + 0.5 * AREATEX_PIXEL_SIZE;
	uv.y    = AREATEX_SUBTEX_SIZE * offset + uv.y;
	return textureLod(area_tex, uv, 0.0).rg;
}

void DetectHorizontalCornerPattern(inout vec2 weights, vec4 coords, vec2 d) {
	vec2 left_right = step(d.xy, d.yx);
	vec2 rounding   = (1.0 - CORNER_ROUNDING_NORM) * left_right;
	rounding /= left_right.x + left_right.y;

	vec2 factor = vec2(1.0);
	factor.x -= rounding.x * EdgesOffset(coords.xy, ivec2(0, 1)).r;
	factor.x -= rounding.y * EdgesOffset(coords.zw, ivec2(1, 1)).r;
	factor.y -= rounding.x * EdgesOffset(coords.xy, ivec2(0, -2)).r;
	factor.y -= rounding.y * EdgesOffset(coords.zw, ivec2(1, -2)).r;
	weights *= clamp(factor, 0.0, 1.0);
}

void DetectVerticalCornerPattern(inout vec2 weights, vec4 coords, vec2 d) {
	vec2 left_right = step(d.xy, d.yx);
	vec2 rounding   = (1.0 - CORNER_ROUNDING_NORM) * left_right;
	rounding /= left_right.x + left_right.y;

	vec2 factor = vec2(1.0);
	factor.x -= rounding.x * EdgesOffset(coords.xy, ivec2(1, 0)).g;
	factor.x -= rounding.y * EdgesOffset(coords.zw, ivec2(1, 1)).g;
	factor.y -= rounding.x * EdgesOffset(coords.xy, ivec2(-2, 0)).g;
	factor.y -= rounding.y * EdgesOffset(coords.zw, ivec2(-2, 1)).g;
	weights *= clamp(factor, 0.0, 1.0);
}

void main() {
	rt_metrics = vec4(1.0 / in_size, in_size);

	vec2 texcoord = gl_FragCoord.xy * rt_metrics.xy;
	vec2 pixcoord = texcoord * rt_metrics.zw;

	vec4 offset0 = rt_metrics.xyxy * vec4(-0.25, -0.125, 1.25, -0.125) + texcoord.xyxy;
	vec4 offset1 = rt_metrics.xyxy * vec4(-0.125, -0.25, -0.125, 1.25) + texcoord.xyxy;
	vec4 offset2 = rt_metrics.xxyy * (vec4(-2.0, 2.0, -2.0, 2.0) * MAX_SEARCH_STEPS) +
	               vec4(offset0.xz, offset1.yw);

	vec4 weights = vec4(0.0);
	vec2 e       = Edges(texcoord);

	if (e.g > 0.0) {
		vec2  d;
		vec3  coords;
		coords.x = SearchXLeft(offset0.xy, offset2.x);
		coords.y = offset1.y;
		d.x      = coords.x;

		float e1 = textureLod(edges_tex, coords.xy, 0.0).r;

		coords.z = SearchXRight(offset0.zw, offset2.y);
		d.y      = coords.z;

		d = abs(round(rt_metrics.z * d - pixcoord.x));
		vec2 sqrt_d = sqrt(d);

		float e2 = EdgesOffset(coords.zy, ivec2(1, 0)).r;

		weights.rg = Area(sqrt_d, e1, e2, 0.0);

		coords.y = texcoord.y;
		DetectHorizontalCornerPattern(weights.rg, coords.xyzy, d);
	}

	if (e.r > 0.0) {
		vec2 d;
		vec3 coords;
		coords.y = SearchYUp(offset1.xy, offset2.z);
		coords.x = offset0.x;
		d.x      = coords.y;

		float e1 = textureLod(edges_tex, coords.xy, 0.0).g;

		coords.z = SearchYDown(offset1.zw, offset2.w);
		d.y      = coords.z;

		d = abs(round(rt_metrics.w * d - pixcoord.y));
		vec2 sqrt_d = sqrt(d);

		float e2 = EdgesOffset(coords.xz, ivec2(0, 1)).g;

		weights.ba = Area(sqrt_d, e1, e2, 0.0);

		coords.x = texcoord.x;
		DetectVerticalCornerPattern(weights.ba, coords.xyxz, d);
	}

	color = weights;
}
