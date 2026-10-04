// Forces + integration, one thread per particle of the *sorted* buffer B.
// Reads B (positions/velocities in cell order), writes the new state to A
// (same sorted order). Never reads what it writes, so no race.

const MAX_TYPES: u32 = 16u;

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> pos_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vel_in: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> cell_start: array<u32>;
@group(0) @binding(4) var<storage, read> cell_count: array<u32>;
@group(0) @binding(5) var<storage, read> matrix: array<f32>;
@group(0) @binding(6) var<storage, read_write> pos_out: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> vel_out: array<vec4<f32>>;

var<workgroup> m: array<f32, 256>;  // MAX_TYPES * MAX_TYPES

fn force(r: f32, a: f32, beta: f32) -> f32 {
    if (r < beta) {
        return r / beta - 1.0;
    }
    if (r < 1.0) {
        return a * (1.0 - abs(2.0 * r - 1.0 - beta) / (1.0 - beta));
    }
    return 0.0;
}

@compute @workgroup_size(WG)
fn step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    // Matrix to workgroup memory; the barrier must run before any early return.
    if (li < P.n_types * P.n_types) {
        m[li] = matrix[li];
    }
    workgroupBarrier();

    let i = gid.x;
    if (i >= P.n) {
        return;
    }
    let me = pos_in[i];
    let p = me.xyz;
    let row = u32(me.w) * P.n_types;
    let nc = P.nc;
    let inc = i32(nc);
    let c = vec3<i32>(cell_coord(p, nc));
    let r2max = P.r_max * P.r_max;
    var acc = vec3<f32>(0.0);
    for (var dx = -1; dx <= 1; dx++) {
        let cx = u32((c.x + dx + inc) % inc);
        for (var dy = -1; dy <= 1; dy++) {
            let cy = u32((c.y + dy + inc) % inc);
            for (var dz = -1; dz <= 1; dz++) {
                let cz = u32((c.z + dz + inc) % inc);
                let cell = (cx * nc + cy) * nc + cz;
                let start = cell_start[cell];
                let end = start + cell_count[cell];
                for (var j = start; j < end; j++) {
                    let other = pos_in[j];
                    var d = other.xyz - p;
                    d -= round(d);  // minimum image on the periodic unit cube
                    let d2 = dot(d, d);
                    if (d2 < r2max && d2 > 0.0) {
                        let dist = sqrt(d2);
                        let f = force(dist / P.r_max, m[row + u32(other.w)], P.beta);
                        acc += d * (f / dist);
                    }
                }
            }
        }
    }
    acc *= P.r_max * P.force_scale;
    let v = vel_in[i].xyz * P.friction + acc * P.dt;
    var q = fract(p + v * P.dt);
    q = select(q, vec3<f32>(0.0), q >= vec3<f32>(1.0));  // fract(-tiny) can round to 1.0
    pos_out[i] = vec4<f32>(q, me.w);
    vel_out[i] = vec4<f32>(v, 0.0);
}
