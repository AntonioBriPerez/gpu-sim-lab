// Rendering: additive glow billboards into an HDR accumulation texture, trail fade,
// tonemap to the swapchain, and line overlays (cube wireframe, brush ring).
// Byte layout must match uniforms.pack_render_params(): 384 bytes.

struct RenderParams {
    view_proj: mat4x4<f32>,       // view-cell coords (target at origin) -> clip
    center: vec3<f32>,            // camera target in world [0,1)^3 (`target` is a WGSL reserved word)
    point_size: f32,              // glow radius in world units
    fwd: vec3<f32>,               // unit view direction
    slice_half: f32,              // only draw |dot(rel, fwd)| <= slice_half
    viewport: vec2<f32>,          // framebuffer size in pixels
    proj_scale: f32,              // pixels per world unit at clip.w == 1: f * height / 2
    fog: f32,                     // depth cue strength (0 = off)
    exposure: f32,
    min_px: f32,                  // minimum glow radius in pixels (avoids sub-pixel flicker)
    _pad0: f32,
    _pad1: f32,
    colors: array<vec4<f32>, 16>, // per type: rgb * highlight gain
};

@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var accum: texture_2d<f32>;

struct ParticleOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
};

// Instanced quad (triangle strip, 4 vertices per particle), offset in clip space so the
// glow has an exact pixel size and never drops below min_px.
@vertex
fn vs_particle(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> ParticleOut {
    var out: ParticleOut;
    let corner = vec2<f32>(f32(vi & 1u), f32(vi >> 1u)) * 2.0 - 1.0;
    out.uv = corner;
    let p = pos[ii];
    var rel = p.xyz - R.center;
    rel -= round(rel);  // periodic: draw the copy nearest to the target
    let depth = dot(rel, R.fwd);  // signed distance to the focal plane (+ = farther)
    let c = R.view_proj * vec4<f32>(rel, 1.0);
    if (abs(depth) > R.slice_half || c.z < 0.0) {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);  // outside the clip volume: culled
        out.color = vec3<f32>(0.0);
        return out;
    }
    let px = R.point_size * R.proj_scale / c.w;
    let s = max(px, R.min_px);
    let energy = (px / s) * (px / s);
    let fade = exp(-R.fog * clamp(depth + 0.5, 0.0, 1.5));
    out.clip = vec4<f32>(c.xy + corner * (2.0 * s / R.viewport) * c.w, c.z, c.w);
    out.color = R.colors[u32(p.w)].rgb * (energy * fade);
    return out;
}

@fragment
fn fs_particle(in: ParticleOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) {
        discard;
    }
    return vec4<f32>(in.color * exp(-4.0 * r2), 0.0);
}

// Fullscreen triangle.
@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// Trails: blend state is (src * 0 + dst * constant), constant = trail decay.
@fragment
fn fs_fade() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}

@fragment
fn fs_tonemap(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(accum, vec2<i32>(frag.xy), 0).rgb * R.exposure;
    return vec4<f32>(1.0 - exp(-c), 1.0);
}

struct LineOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_line(@location(0) p: vec3<f32>, @location(1) color: vec3<f32>) -> LineOut {
    var out: LineOut;
    out.clip = R.view_proj * vec4<f32>(p, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
