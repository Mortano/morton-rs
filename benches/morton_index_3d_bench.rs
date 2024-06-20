use criterion::{criterion_group, criterion_main, Criterion};
use morton_index::{dimensions::OctantOrdering, FixedDepthMortonIndex3D64};
use nalgebra::Vector3;
use rand::{thread_rng, Rng};

fn get_dummy_grid_indices_64(count: usize) -> Vec<Vector3<usize>> {
    let mut rng = thread_rng();
    let max_idx = 1 << 21;
    (0..count)
        .map(|_| {
            Vector3::new(
                rng.gen_range(0..max_idx),
                rng.gen_range(0..max_idx),
                rng.gen_range(0..max_idx),
            )
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let grid_indices = get_dummy_grid_indices_64(1024);

    c.bench_function("fixed_depth_morton_index_64_from_grid_index", |b| {
        b.iter(|| {
            grid_indices
                .iter()
                .copied()
                .map(|idx| {
                    FixedDepthMortonIndex3D64::from_grid_index_fast(idx, OctantOrdering::XYZ)
                })
                .for_each(|idx| {
                    criterion::black_box(idx);
                });
        })
    });
}

criterion_group! {
    name = point_buffer_iterators;
    config = Criterion::default().sample_size(40);
    targets = bench
}
criterion_main!(point_buffer_iterators);
