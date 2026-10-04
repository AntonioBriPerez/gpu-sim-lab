// Counting sort of particles into grid cells. Pass 1: count + per-particle rank.
// Pass 3 (after the exclusive scan of cell_count into cell_start): scatter A -> B.

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> pos_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cell_count: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cell_of: array<u32>;
@group(0) @binding(4) var<storage, read_write> rank: array<u32>;

@compute @workgroup_size(WG)
fn count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.n) {
        return;
    }
    let c = cell_index(cell_coord(pos_in[i].xyz, P.nc), P.nc);
    cell_of[i] = c;
    rank[i] = atomicAdd(&cell_count[c], 1u);
}
