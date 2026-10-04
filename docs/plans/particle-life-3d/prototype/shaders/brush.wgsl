// Mouse brush: pull (strength > 0) or push (< 0) particles around a world point.
// Byte layout must match uniforms.pack_brush_params(): 32 bytes.

struct BrushParams {
    center: vec3<f32>,  // world position in [0,1)^3
    radius: f32,
    strength: f32,
    dt: f32,
    n: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> B: BrushParams;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4<f32>>;

@compute @workgroup_size(256)
fn brush(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= B.n) {
        return;
    }
    var d = B.center - pos[i].xyz;
    d -= round(d);
    let dist = length(d);
    if (dist > 0.0 && dist < B.radius) {
        let v = vel[i];
        let dv = d / dist * B.strength * (1.0 - dist / B.radius) * B.dt;
        vel[i] = vec4<f32>(v.xyz + dv, v.w);
    }
}
