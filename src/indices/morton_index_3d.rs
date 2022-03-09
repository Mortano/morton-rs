// TODO This whole file will be VERY similar to `morton_index_2d.rs`. Is there some way to combine the two?
// The biggest hurdle I see is that many of the types are different, and the implementations will be different
// for some of the functions that have to care about dimensionality...

use nalgebra::Vector3;

use crate::{
    dimensions::{Dim3D, Dimension, Octant, OctantOrdering},
    number::{add_two_zeroes_before_every_bit_u8, Bits},
    CellIter, FixedDepthStorage, FixedStorageType, MortonIndex, MortonIndexNaming, Storage,
    StorageType, VariableDepthStorage,
};

pub type FixedDepthMortonIndex3D8 = MortonIndex3D<FixedDepthStorage3D<u8>>;
pub type FixedDepthMortonIndex3D16 = MortonIndex3D<FixedDepthStorage3D<u16>>;
pub type FixedDepthMortonIndex3D32 = MortonIndex3D<FixedDepthStorage3D<u32>>;
pub type FixedDepthMortonIndex3D64 = MortonIndex3D<FixedDepthStorage3D<u64>>;
pub type FixedDepthMortonIndex3D128 = MortonIndex3D<FixedDepthStorage3D<u128>>;

/// A §D Morton index. This represents a single node inside an octree. The depth of the node and the maximum storage
/// capacity of this type depend on the generic `Storage` type
#[derive(Default, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct MortonIndex3D<S: Storage<Dim3D>> {
    storage: S,
}

impl<S: Storage<Dim3D> + Clone> Clone for MortonIndex3D<S> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
        }
    }
}

impl<S: Storage<Dim3D> + Copy> Copy for MortonIndex3D<S> {}

impl<'a, S: Storage<Dim3D> + 'a> MortonIndex3D<S>
where
    &'a S: IntoIterator<Item = Octant>,
{
    pub fn cells(&'a self) -> <&'a S as IntoIterator>::IntoIter {
        self.storage.into_iter()
    }
}

impl<S: VariableDepthStorage<Dim3D>> MortonIndex3D<S> {
    /// Returns a Morton index for the child `octant` of this Morton index. If this index is already at
    /// the maximum depth, `None` is returned instead
    pub fn child(&self, octant: Octant) -> Option<Self> {
        self.storage.child(octant).map(|storage| Self { storage })
    }

    /// Returns a Morton index for the parent node of this Morton index. If this index represents the root
    /// node, `None` is returned instead
    pub fn parent(&self) -> Option<Self> {
        self.storage.parent().map(|storage| Self { storage })
    }
}

impl<'a, S: Storage<Dim3D> + TryFrom<&'a [Octant], Error = crate::Error>> TryFrom<&'a [Octant]>
    for MortonIndex3D<S>
{
    type Error = crate::Error;

    fn try_from(value: &'a [Octant]) -> Result<Self, Self::Error> {
        S::try_from(value).map(|storage| Self { storage })
    }
}

impl<S: Storage<Dim3D>> MortonIndex for MortonIndex3D<S> {
    type Dimension = Dim3D;

    fn get_cell_at_level(&self, level: usize) -> Octant {
        if level >= self.depth() {
            panic!("level must not be >= self.depth()");
        }
        unsafe { self.storage.get_cell_at_level_unchecked(level) }
    }

    unsafe fn get_cell_at_level_unchecked(&self, level: usize) -> Octant {
        self.storage.get_cell_at_level_unchecked(level)
    }

    fn set_cell_at_level(&mut self, level: usize, cell: Octant) {
        if level >= self.depth() {
            panic!("level must not be >= self.depth()");
        }
        unsafe { self.storage.set_cell_at_level_unchecked(level, cell) }
    }

    unsafe fn set_cell_at_level_unchecked(&mut self, level: usize, cell: Octant) {
        self.storage.set_cell_at_level_unchecked(level, cell)
    }

    fn depth(&self) -> usize {
        self.storage.depth()
    }

    fn to_string(&self, naming: crate::MortonIndexNaming) -> String {
        match naming {
            crate::MortonIndexNaming::CellConcatenation => (0..self.depth())
                .map(|level| {
                    // Safe because we know the depth
                    let cell = unsafe { self.get_cell_at_level_unchecked(level) };
                    std::char::from_digit(cell.index() as u32, 10).unwrap()
                })
                .collect::<String>(),
            crate::MortonIndexNaming::CellConcatenationWithRoot => {
                format!("r{}", self.to_string(MortonIndexNaming::CellConcatenation))
            }
            crate::MortonIndexNaming::GridIndex => {
                let grid_index = self.to_grid_index(OctantOrdering::default());
                format!(
                    "{}-{}-{}-{}",
                    self.depth(),
                    grid_index.x,
                    grid_index.y,
                    grid_index.z
                )
            }
        }
    }

    fn to_grid_index(&self, ordering: OctantOrdering) -> Vector3<usize> {
        let mut index = Vector3::new(0, 0, 0);
        for level in 0..self.depth() {
            let octant = unsafe { self.get_cell_at_level_unchecked(level) };
            let octant_index = ordering.to_index(octant);
            let shift = self.depth() - level - 1;
            index.x |= octant_index.x << shift;
            index.y |= octant_index.y << shift;
            index.z |= octant_index.z << shift;
        }
        index
    }
}

impl<B: FixedStorageType> MortonIndex3D<FixedDepthStorage3D<B>> {
    /// Creates a new MortonIndex3D with fixed-depth storage from the given 3D grid index. The `grid_index` is assumed to
    /// represent a grid with a depth equal to `FixedDepthStorage3D<B>::MAX_LEVELS`.
    ///
    /// # Panics
    ///
    /// There is an edge-case in which the fixed depth of the `FixedDepthStorage3D<B>` is greater than what a single `usize`
    /// value in the `grid_index` can represent. In this case the code will panic.
    pub fn from_grid_index(
        grid_index: <Dim3D as Dimension>::GridIndex,
        ordering: OctantOrdering,
    ) -> Self {
        let fixed_depth = <FixedDepthStorage<Dim3D, B> as StorageType>::MAX_LEVELS;
        if fixed_depth > (std::mem::size_of::<usize>() * 8) {
            panic!(
                "Size of usize is too small for a fixed depth of {}",
                fixed_depth
            );
        }
        // Similar construction as compared to static storage, but we have a fixed depth

        let x_bits = unsafe { grid_index.x.get_bits(0..fixed_depth) };
        let y_bits = unsafe { grid_index.y.get_bits(0..fixed_depth) };
        let z_bits = unsafe { grid_index.z.get_bits(0..fixed_depth) };

        let (x_shift, y_shift, z_shift) = ordering.get_bit_shifts_for_xyz();

        let mut bits = B::default();
        // Data is stored starting at the MSB, so there might be some leftover (i.e. unused) bits at the end
        // We have to take this into consideration when we set the bits!
        let leftover_bits_at_end: usize = B::BITS % Dim3D::DIMENSIONALITY;
        // As opposed to the 2D case, in the 3D case we can't simply process 8-bit chunks at a time, because
        // a single cell takes 3 bits and thus doesn't evenly divide 8. So instead we read one byte at a time
        // from the index and set 10 bits at a time
        let num_chunks = (fixed_depth + 7) / 8;
        for chunk_index in 0..num_chunks {
            let start_bit = chunk_index * 8;
            let end_bit = start_bit + 8;
            let x_chunk = unsafe { x_bits.get_bits(start_bit..end_bit) as u8 };
            let y_chunk = unsafe { y_bits.get_bits(start_bit..end_bit) as u8 };
            let z_chunk = unsafe { z_bits.get_bits(start_bit..end_bit) as u8 };
            let chunk = (add_two_zeroes_before_every_bit_u8(x_chunk) << x_shift)
                | (add_two_zeroes_before_every_bit_u8(y_chunk) << y_shift)
                | (add_two_zeroes_before_every_bit_u8(z_chunk) << z_shift);

            // chunk contains 24 valid bits at most
            let start_bit = (chunk_index * 24) + leftover_bits_at_end;
            let end_bit = (start_bit + 24).min(B::BITS);
            unsafe {
                bits.set_bits(start_bit..end_bit, B::from_u32(chunk));
            }
        }

        Self {
            storage: FixedDepthStorage3D { bits },
        }
    }
}

// Storage types:

/// Storage for a 3D Morton index that always stores a Morton index with a fixed depth (fixed number of levels)
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FixedDepthStorage3D<B: FixedStorageType> {
    bits: B,
}

impl<B: FixedStorageType> Storage<Dim3D> for FixedDepthStorage3D<B> {
    type StorageType = FixedDepthStorage<Dim3D, B>;
    type Bits = B;

    fn bits(&self) -> &Self::Bits {
        &self.bits
    }

    fn bits_mut(&mut self) -> &mut Self::Bits {
        &mut self.bits
    }

    fn depth(&self) -> usize {
        FixedDepthStorage::<Dim3D, B>::MAX_LEVELS
    }
}

impl<'a, B: FixedStorageType> TryFrom<&'a [Octant]> for FixedDepthStorage3D<B> {
    type Error = crate::Error;

    fn try_from(octants: &'a [Octant]) -> Result<Self, Self::Error> {
        if octants.len() > <FixedDepthStorage<Dim3D, B> as StorageType>::MAX_LEVELS {
            return Err(crate::Error::DepthLimitedExceeded {
                max_depth: <FixedDepthStorage<Dim3D, B> as StorageType>::MAX_LEVELS,
            });
        }
        let mut ret: Self = Default::default();
        for (level, cell) in octants.iter().enumerate() {
            unsafe {
                ret.set_cell_at_level_unchecked(level, *cell);
            }
        }
        Ok(ret)
    }
}

impl<'a, B: FixedStorageType> IntoIterator for &'a FixedDepthStorage3D<B> {
    type Item = Octant;
    type IntoIter = CellIter<'a, Dim3D, FixedDepthStorage3D<B>>;

    fn into_iter(self) -> Self::IntoIter {
        CellIter {
            index: 0,
            storage: &self,
            _phantom: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    fn get_test_octants(count: usize) -> Vec<Octant> {
        let mut rng = thread_rng();
        (0..count)
            .map(|_| {
                let num: usize = rng.gen_range(0..4);
                Octant::try_from(num).unwrap()
            })
            .collect()
    }

    macro_rules! test_fixed_depth {
        ($typename:ident, $modname:ident, $max_levels:literal) => {
            mod $modname {
                use super::*;

                const MAX_LEVELS: usize = $max_levels;

                #[test]
                fn default() {
                    let idx = $typename::default();
                    assert_eq!(MAX_LEVELS, idx.depth());
                    for level in 0..MAX_LEVELS {
                        assert_eq!(Octant::Zero, idx.get_cell_at_level(level));
                    }
                }

                #[test]
                fn from_octants() {
                    let octants = get_test_octants(MAX_LEVELS);
                    let idx = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    assert_eq!(octants.len(), idx.depth());
                    for (level, expected_octant) in octants.iter().enumerate() {
                        assert_eq!(*expected_octant, idx.get_cell_at_level(level));
                    }
                }

                #[test]
                fn from_too_many_octants_fails() {
                    let octants = get_test_octants(MAX_LEVELS + 1);
                    let res = $typename::try_from(octants.as_slice());
                    assert!(res.is_err());
                }

                #[test]
                fn from_fewer_octants_than_max_depth() {
                    let octants = get_test_octants(MAX_LEVELS - 2);
                    let idx = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    assert_eq!(MAX_LEVELS, idx.depth());
                    for (level, expected_octant) in octants.iter().enumerate() {
                        assert_eq!(*expected_octant, idx.get_cell_at_level(level));
                    }
                    // Remaining octants should be zero-initialized!
                    assert_eq!(Octant::Zero, idx.get_cell_at_level(MAX_LEVELS - 2));
                    assert_eq!(Octant::Zero, idx.get_cell_at_level(MAX_LEVELS - 1));
                }

                #[test]
                fn set_cell() {
                    let octants = get_test_octants(MAX_LEVELS);
                    let mut idx = $typename::default();
                    for level in 0..MAX_LEVELS {
                        idx.set_cell_at_level(level, octants[level]);
                    }

                    for level in 0..MAX_LEVELS {
                        assert_eq!(
                            octants[level],
                            idx.get_cell_at_level(level),
                            "Wrong octants at level {} in index {}",
                            level,
                            idx.to_string(MortonIndexNaming::CellConcatenation)
                        );
                    }
                }

                #[test]
                #[should_panic]
                fn set_cell_oob_panics() {
                    let mut idx = $typename::default();
                    idx.set_cell_at_level(MAX_LEVELS, Octant::One);
                }

                #[test]
                fn cells_iter() {
                    let octants = get_test_octants(MAX_LEVELS);
                    let idx = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    let collected_octants = idx.cells().collect::<Vec<_>>();
                    assert_eq!(octants.as_slice(), collected_octants.as_slice());
                }

                #[test]
                fn ordering() {
                    let count = 16;
                    let mut indices = (0..count)
                        .map(|_| {
                            let octants = get_test_octants(MAX_LEVELS);
                            $typename::try_from(octants.as_slice())
                                .expect("Could not create Morton index from octants")
                        })
                        .collect::<Vec<_>>();

                    indices.sort();

                    // Indices have to be sorted in ascending order, meaning that for indices i and j with i < j,
                    // each octant of index i must be <= the octant at the same level of index j, up until and including
                    // the first octant of i that is less than j
                    for idx in 0..(count - 1) {
                        let i = indices[idx];
                        let j = indices[idx + 1];
                        for (cell_low, cell_high) in i.cells().zip(j.cells()) {
                            if cell_low < cell_high {
                                break;
                            }
                            assert!(
                                cell_low <= cell_high,
                                "Index {} is not <= index {}",
                                i.to_string(MortonIndexNaming::CellConcatenation),
                                j.to_string(MortonIndexNaming::CellConcatenation)
                            );
                        }
                    }
                }

                #[test]
                fn roundtrip_grid_index() {
                    let octants = get_test_octants(MAX_LEVELS);
                    let idx = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");

                    for octant_ordering in [
                        OctantOrdering::XYZ,
                        OctantOrdering::XZY,
                        OctantOrdering::YXZ,
                        OctantOrdering::YZX,
                        OctantOrdering::ZXY,
                        OctantOrdering::ZYX,
                    ] {
                        let grid_index = idx.to_grid_index(octant_ordering);
                        let rountrip_idx = $typename::from_grid_index(grid_index, octant_ordering);
                        assert_eq!(idx, rountrip_idx);
                    }
                }
            }
        };
    }

    //
    test_fixed_depth!(FixedDepthMortonIndex3D8, fixed_depth_morton_index_3d8, 2);
    test_fixed_depth!(FixedDepthMortonIndex3D16, fixed_depth_morton_index_3d16, 5);
    test_fixed_depth!(FixedDepthMortonIndex3D32, fixed_depth_morton_index_3d32, 10);
    test_fixed_depth!(FixedDepthMortonIndex3D64, fixed_depth_morton_index_3d64, 21);
    test_fixed_depth!(
        FixedDepthMortonIndex3D128,
        fixed_depth_morton_index_3d128,
        42
    );
}
