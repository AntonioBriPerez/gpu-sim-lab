// Exclusive prefix sum of u32, two levels (up to BLOCK * BLOCK = 1,048,576 elements).
//   scan_blocks: each workgroup scans BLOCK = 1024 elements (256 threads x 4) and writes
//                its total to block_sums[workgroup].
//   add_offsets: dst[i] += block_sums[i / BLOCK], where block_sums now holds the
//                *scanned* block totals (the same scan_blocks run on the totals).
// Standalone module (no SimParams).

struct ScanParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

const WG: u32 = 256u;
const PER_THREAD: u32 = 4u;
const BLOCK: u32 = 1024u;  // WG * PER_THREAD

@group(0) @binding(0) var<uniform> S: ScanParams;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read_write> block_sums: array<u32>;

var<workgroup> sh: array<u32, WG>;

@compute @workgroup_size(WG)
fn scan_blocks(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let t = lid.x;
    let base = wid.x * BLOCK + t * PER_THREAD;
    var local_excl: array<u32, PER_THREAD>;
    var sum = 0u;
    for (var k = 0u; k < PER_THREAD; k++) {
        let idx = base + k;
        var x = 0u;
        if (idx < S.n) {
            x = src[idx];
        }
        local_excl[k] = sum;
        sum += x;
    }
    // Hillis-Steele inclusive scan of the per-thread sums (barriers in uniform control flow).
    sh[t] = sum;
    workgroupBarrier();
    for (var off = 1u; off < WG; off *= 2u) {
        var add = 0u;
        if (t >= off) {
            add = sh[t - off];
        }
        workgroupBarrier();
        sh[t] += add;
        workgroupBarrier();
    }
    let thread_excl = sh[t] - sum;
    for (var k = 0u; k < PER_THREAD; k++) {
        let idx = base + k;
        if (idx < S.n) {
            dst[idx] = local_excl[k] + thread_excl;
        }
    }
    if (t == WG - 1u) {
        block_sums[wid.x] = sh[t];
    }
}

@compute @workgroup_size(WG)
fn add_offsets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= S.n) {
        return;
    }
    dst[i] += block_sums[i / BLOCK];
}
