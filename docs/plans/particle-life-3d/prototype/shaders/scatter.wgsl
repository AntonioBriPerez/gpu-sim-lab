// Pass 3 of the counting sort: copy particle i of buffer A to its sorted slot in buffer B.

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> pos_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vel_in: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> cell_of: array<u32>;
@group(0) @binding(4) var<storage, read> rank: array<u32>;
@group(0) @binding(5) var<storage, read> cell_start: array<u32>;
@group(0) @binding(6) var<storage, read_write> pos_out: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> vel_out: array<vec4<f32>>;

@compute @workgroup_size(WG)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.n) {
        return;
    }
    let dst = cell_start[cell_of[i]] + rank[i];
    pos_out[dst] = pos_in[i];
    vel_out[dst] = vel_in[i];
}
