// Shared by every simulation kernel (prepended by the Python loader; WGSL has no #include).
// Byte layout must match uniforms.pack_sim_params(): 48 bytes.
struct SimParams {
    n: u32,            // particle count
    nc: u32,           // grid cells per side (>= 3)
    n_types: u32,
    _pad0: u32,
    r_max: f32,
    beta: f32,
    force_scale: f32,
    dt: f32,
    friction: f32,     // per-step velocity multiplier, 0.5^(dt / half_life)
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

const WG: u32 = 256u;

// Cell of a position in [0,1)^3; clamped so p == 1.0 (float rounding) stays in range.
fn cell_coord(p: vec3<f32>, nc: u32) -> vec3<u32> {
    let hi = f32(nc - 1u);
    return vec3<u32>(clamp(p * f32(nc), vec3<f32>(0.0), vec3<f32>(hi)));
}

fn cell_index(c: vec3<u32>, nc: u32) -> u32 {
    return (c.x * nc + c.y) * nc + c.z;
}
