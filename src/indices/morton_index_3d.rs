// TODO This whole file will be VERY similar to `morton_index_2d.rs`. Is there some way to combine the two?
// The biggest hurdle I see is that many of the types are different, and the implementations will be different
// for some of the functions that have to care about dimensionality...

use std::cmp::Ordering;

use nalgebra::Vector3;

use crate::{
    dimensions::{Dim3D, Dimension, Octant, OctantOrdering},
    number::{add_two_zeroes_before_every_bit_u8, Bits},
    CellIter, DynamicStorage, FixedDepthStorage, FixedStorageType, MortonIndex, MortonIndexNaming,
    StaticStorage, Storage, StorageType, VariableDepthMortonIndex, VariableDepthStorage,
};

pub type FixedDepthMortonIndex3D8 = MortonIndex3D<FixedDepthStorage3D<u8>>;
pub type FixedDepthMortonIndex3D16 = MortonIndex3D<FixedDepthStorage3D<u16>>;
pub type FixedDepthMortonIndex3D32 = MortonIndex3D<FixedDepthStorage3D<u32>>;
pub type FixedDepthMortonIndex3D64 = MortonIndex3D<FixedDepthStorage3D<u64>>;
pub type FixedDepthMortonIndex3D128 = MortonIndex3D<FixedDepthStorage3D<u128>>;

pub type StaticMortonIndex3D8 = MortonIndex3D<StaticStorage3D<u8>>;
pub type StaticMortonIndex3D16 = MortonIndex3D<StaticStorage3D<u16>>;
pub type StaticMortonIndex3D32 = MortonIndex3D<StaticStorage3D<u32>>;
pub type StaticMortonIndex3D64 = MortonIndex3D<StaticStorage3D<u64>>;
pub type StaticMortonIndex3D128 = MortonIndex3D<StaticStorage3D<u128>>;

pub type DynamicMortonIndex3D = MortonIndex3D<DynamicStorage3D>;

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

impl<S: VariableDepthStorage<Dim3D>> VariableDepthMortonIndex for MortonIndex3D<S> {
    fn ancestor(&self, generations: std::num::NonZeroUsize) -> Option<Self> {
        self.storage
            .ancestor(generations)
            .map(|storage| Self { storage })
    }

    fn descendant(
        &self,
        cells: &[<Self::Dimension as crate::dimensions::Dimension>::Cell],
    ) -> Option<Self> {
        self.storage
            .descendant(cells)
            .map(|storage| Self { storage })
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
    /// Maximum levels that this Morton index type can represent
    pub const MAX_LEVELS: usize = B::BITS / Dim3D::DIMENSIONALITY;
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

impl<B: FixedStorageType> MortonIndex3D<StaticStorage3D<B>> {
    /// Maximum levels that this Morton index type can represent
    pub const MAX_LEVELS: usize = B::BITS / Dim3D::DIMENSIONALITY;
    /// Returns a MortonIndex3D with static storage with the given `depth` where all cells are zeroed (i.e. representing `Octant::Zero`). If
    /// the `depth` is larger than the maximum depth of the `StaticStorage3D<B>`, `None` is returned
    pub fn zeroed(depth: u8) -> Option<Self> {
        if depth as usize > <StaticStorage<Dim3D, B> as StorageType>::MAX_LEVELS {
            return None;
        }
        Some(Self {
            storage: StaticStorage3D::<B> {
                bits: Default::default(),
                depth,
            },
        })
    }

    /// Creates a new MortonIndex3D with static storage from the given 3D grid index. The `grid_depth` parameter
    /// describes the 'depth' of the grid as per the equation: `N = 2 ^ grid_depth`, where `N` is the number of cells
    /// per axis in the grid. For example, a `32*32*32` grid has a `grid_depth` of `log2(32) = 5`. If `3 * grid_depth`
    /// exceeds the capacity of the static storage type `B`, this returns an error.
    pub fn from_grid_index(
        grid_index: <Dim3D as Dimension>::GridIndex,
        grid_depth: usize,
        ordering: OctantOrdering,
    ) -> Result<Self, crate::Error> {
        if grid_depth > <FixedDepthStorage<Dim3D, B> as StorageType>::MAX_LEVELS {
            return Err(crate::Error::DepthLimitedExceeded {
                max_depth: <FixedDepthStorage<Dim3D, B> as StorageType>::MAX_LEVELS,
            });
        }

        // Since we always start at the MSB for level 0, but grid indices have different numbers of bits depending
        // on the `grid_depth`, we shift the x, y, and z parts of the `grid_index` so that they match the maximum depth
        // Example:
        // Suppose we have a `u8` storage for a StaticMortonIndex3D. Then, the bits (in default XYZ octant order) look like this:
        //   Bit index:   7  6  5  4  3  2  1  0
        //               z1 y1 x1 z2 y2 x2  0  0
        // The first octant, defined by (x1, y1, z1) sits in the most-significant bits. This is true regardless of the depth of the
        // Morton index, so for an index with depth 1, the bits look like this:
        //   Bit index:   7  6  5  4  3  2  1  0
        //               z1 y1 x1  0  0  0  0  0
        //
        // Now look at the grid indices that correspond to these Morton indices! In the first case, the grid index looks like this:
        //   Bit index:   7  6  5  4  3  2  1  0
        //           X:   0  0  0  0  0  0 x1 x2
        //           Y:   0  0  0  0  0  0 y1 y2
        //           Z:   0  0  0  0  0  0 z1 z2
        //
        // In the second case, this is the grid index:
        //   Bit index:   7  6  5  4  3  2  1  0
        //           X:   0  0  0  0  0  0  0 x1
        //           Y:   0  0  0  0  0  0  0 y1
        //           Z:   0  0  0  0  0  0  0 z1
        //
        // So for depth=2, the `x1` bit has index 1, whereas for depth=1, the `x1` bit has index 0. This might look weird, but this is
        // how numbers work, and a grid index is nothing but a number that identifies a cell position in X, Y, Z. In order to rectify this,
        // we thus have to shift the bits of the grid indices that have a depth less than the MAX_LEVELS up, so that we get this:
        //   Bit index:   7  6  5  4  3  2  1  0
        //           X:   0  0  0  0  0  0 x1  0
        //           Y:   0  0  0  0  0  0 y1  0
        //           Z:   0  0  0  0  0  0 z1  0
        //
        // We then start setting bits from the grid index bits over to the Morton index from the most-significant bit down, which is why
        // we have this rev() call in the for-loop!

        let shift_to_max_bits =
            <FixedDepthStorage<Dim3D, B> as StorageType>::MAX_LEVELS - grid_depth;
        let x_bits = unsafe { grid_index.x.get_bits(0..grid_depth) << shift_to_max_bits };
        let y_bits = unsafe { grid_index.y.get_bits(0..grid_depth) << shift_to_max_bits };
        let z_bits = unsafe { grid_index.z.get_bits(0..grid_depth) << shift_to_max_bits };

        let (x_shift, y_shift, z_shift) = ordering.get_bit_shifts_for_xyz();

        let mut bits = B::default();
        // Data is stored starting at the MSB, so there might be some leftover (i.e. unused) bits at the end
        // We have to take this into consideration when we set the bits!
        let leftover_bits_at_end: usize = B::BITS % Dim3D::DIMENSIONALITY;
        // As opposed to the 2D case, in the 3D case we can't simply process 8-bit chunks at a time, because
        // a single cell takes 3 bits and thus doesn't evenly divide 8. So instead we read one byte at a time
        // from the grid index and set 24 bits at a time in the Morton index
        // We also start setting the bits from the most significant chunk towards the least significant chunk!
        // This way, we make sure that we set the high bits first in cases where the grid_depth is smaller than
        // the MAX_LEVELS
        let num_chunks = (grid_depth + 7) / 8;
        let max_chunks = (<FixedDepthStorage<Dim3D, B> as StorageType>::MAX_LEVELS + 7) / 8;
        let chunk_start = max_chunks - num_chunks;
        for chunk_index in (chunk_start..max_chunks).rev() {
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

        Ok(Self {
            storage: StaticStorage3D {
                bits,
                depth: grid_depth as u8,
            },
        })
    }
}

impl MortonIndex3D<DynamicStorage3D> {
    pub fn zeroed(depth: usize) -> Self {
        Self {
            storage: DynamicStorage3D::zeroed(depth),
        }
    }

    /// Creates a new MortonIndex3D with dynamic storage from the given 3D grid index. The `grid_depth` parameter
    /// describes the 'depth' of the grid as per the equation: `N = 2 ^ grid_depth`, where `N` is the number of cells
    /// per axis in the grid. For example, a `32*32*32` grid has a `grid_depth` of `log2(32) = 5`.
    pub fn from_grid_index(
        grid_index: <Dim3D as Dimension>::GridIndex,
        grid_depth: usize,
        ordering: OctantOrdering,
    ) -> Self {
        let max_bits = std::mem::size_of_val(&grid_index.x) * 8;
        if grid_depth > (max_bits / 2) {
            panic!(
                "2 * grid_depth ({}) must not exceed maximum number of bits ({}) in the grid index",
                grid_depth, max_bits
            );
        }

        let x_bits = unsafe { grid_index.x.get_bits(0..grid_depth) };
        let y_bits = unsafe { grid_index.y.get_bits(0..grid_depth) };
        let z_bits = unsafe { grid_index.z.get_bits(0..grid_depth) };

        let (x_shift, y_shift, z_shift) = ordering.get_bit_shifts_for_xyz();

        let mut ret = DynamicStorage3D::zeroed(grid_depth);
        for level in 0..grid_depth {
            let shift = grid_depth - level - 1;
            let x_bit = ((x_bits >> shift) & 1) << x_shift;
            let y_bit = ((y_bits >> shift) & 1) << y_shift;
            let z_bit = ((z_bits >> shift) & 1) << z_shift;
            let octant = Octant::try_from(x_bit | y_bit | z_bit).unwrap();
            unsafe {
                ret.set_cell_at_level_unchecked(level, octant);
            }
        }

        Self { storage: ret }
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

#[derive(Default, Debug, Clone, Copy, Hash)]
pub struct StaticStorage3D<B: FixedStorageType> {
    bits: B,
    depth: u8,
}

impl<B: FixedStorageType> Storage<Dim3D> for StaticStorage3D<B> {
    type StorageType = StaticStorage<Dim3D, B>;
    type Bits = B;

    fn bits(&self) -> &Self::Bits {
        &self.bits
    }

    fn bits_mut(&mut self) -> &mut Self::Bits {
        &mut self.bits
    }

    fn depth(&self) -> usize {
        self.depth as usize
    }
}

impl<B: FixedStorageType> VariableDepthStorage<Dim3D> for StaticStorage3D<B> {
    fn ancestor(&self, generations: std::num::NonZeroUsize) -> Option<Self> {
        if generations.get() > self.depth as usize {
            return None;
        }

        let mut ret = *self;
        // Zero out all quadrants below the new depth
        let new_depth = self.depth - generations.get() as u8;
        for level in new_depth..self.depth {
            unsafe {
                ret.set_cell_at_level_unchecked(level as usize, Octant::Zero);
            }
        }
        ret.depth = new_depth;
        Some(ret)
    }

    fn descendant(&self, cells: &[<Dim3D as Dimension>::Cell]) -> Option<Self> {
        let new_depth = self.depth as usize + cells.len();
        if new_depth > <StaticStorage<Dim3D, B> as StorageType>::MAX_LEVELS {
            return None;
        }

        let mut ret = *self;
        for (offset, cell) in cells.iter().enumerate() {
            unsafe {
                ret.set_cell_at_level_unchecked(self.depth as usize + offset, *cell);
            }
        }
        ret.depth = new_depth as u8;
        Some(ret)
    }
}

impl<B: FixedStorageType> PartialOrd for StaticStorage3D<B> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<B: FixedStorageType> Ord for StaticStorage3D<B> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let end_bit = B::BITS;
        let self_start_bit = end_bit - (self.depth() * Dim3D::DIMENSIONALITY);
        let other_start_bit = end_bit - (other.depth() * Dim3D::DIMENSIONALITY);
        match self.depth().cmp(&other.depth()) {
            Ordering::Less => {
                // Only compare bits up to 2*self.depth(). If less or equal, self is by definition less
                let self_bits = unsafe { self.bits.get_bits(self_start_bit..end_bit) };
                let other_bits = unsafe { other.bits.get_bits(self_start_bit..end_bit) };
                if self_bits > other_bits {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
            Ordering::Equal => {
                let self_bits = unsafe { self.bits.get_bits(self_start_bit..end_bit) };
                let other_bits = unsafe { other.bits.get_bits(other_start_bit..end_bit) };
                self_bits.cmp(&other_bits)
            }
            Ordering::Greater => {
                // This is the opposite of the Ordering::Less case, we compare bits up to 2*other.depth() and
                // if self is greater or equal, other is less, otherwise other is greater
                let self_bits = unsafe { self.bits.get_bits(other_start_bit..end_bit) };
                let other_bits = unsafe { other.bits.get_bits(other_start_bit..end_bit) };
                if self_bits >= other_bits {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
        }
    }
}

impl<B: FixedStorageType> PartialEq for StaticStorage3D<B> {
    fn eq(&self, other: &Self) -> bool {
        // Technically we should compare self.bits.bits(B::BITS..(B::BITS - 3*self.depth)) instead of all bits,
        // but all the operations that create a StaticStorage3D make sure that the 'waste' bits are always zero
        self.bits == other.bits && self.depth == other.depth
    }
}

impl<B: FixedStorageType> Eq for StaticStorage3D<B> {}

impl<'a, B: FixedStorageType> TryFrom<&'a [Octant]> for StaticStorage3D<B> {
    type Error = crate::Error;

    fn try_from(octants: &'a [Octant]) -> Result<Self, Self::Error> {
        if octants.len() > <StaticStorage<Dim3D, B> as StorageType>::MAX_LEVELS {
            return Err(crate::Error::DepthLimitedExceeded {
                max_depth: <StaticStorage<Dim3D, B> as StorageType>::MAX_LEVELS,
            });
        }
        let mut ret: Self = Default::default();
        for (level, cell) in octants.iter().enumerate() {
            unsafe {
                ret.set_cell_at_level_unchecked(level, *cell);
            }
        }
        ret.depth = octants.len() as u8;
        Ok(ret)
    }
}

impl<'a, B: FixedStorageType> IntoIterator for &'a StaticStorage3D<B> {
    type Item = Octant;
    type IntoIter = CellIter<'a, Dim3D, StaticStorage3D<B>>;

    fn into_iter(self) -> Self::IntoIter {
        CellIter {
            index: 0,
            storage: self,
            _phantom: Default::default(),
        }
    }
}

/// Morton index storage which can store an arbitrary number of levels without an upper limit. This uses a
/// `Vec<u8>` as the internal storage, which can grow dynamically, at the expense of not being `Copy` and
/// potentially worse performance than the `StaticStorage3D` and `FixedStorage3D` variants.
///
/// Note: There is no guarantee about endianness on any of the storage types, the `DynamicStorage3D` might
/// not even have a consistent endianness, so it's not valid to assume any order of the internal bits.
#[derive(Default, Debug, Clone, Hash, PartialEq, Eq)]
pub struct DynamicStorage3D {
    bits: Vec<u8>,
    depth: usize,
}

impl DynamicStorage3D {
    /// Returns a new `DynamicStorage3D` with the given `depth` and all bits set to zero, i.e. all cells equal
    /// to `Octant::Zero?`
    pub fn zeroed(depth: usize) -> Self {
        let num_bytes = ((depth * 3) + 7) / 8;
        Self {
            bits: vec![0; num_bytes],
            depth,
        }
    }

    fn get_cell_at_level_or_none(&self, level: usize) -> Option<Octant> {
        if level >= self.depth.into() {
            None
        } else {
            // Is safe because of level check above
            unsafe { Some(self.get_cell_at_level_unchecked(level)) }
        }
    }
}

impl PartialOrd for DynamicStorage3D {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DynamicStorage3D {
    fn cmp(&self, other: &Self) -> Ordering {
        let max_level = self.depth().max(other.depth());
        for level in 0..max_level {
            let self_cell = self.get_cell_at_level_or_none(level);
            let other_cell = other.get_cell_at_level_or_none(level);
            let cmp = self_cell.cmp(&other_cell);
            if cmp == Ordering::Equal {
                continue;
            }
            return cmp;
        }
        Ordering::Equal
    }
}

impl Storage<Dim3D> for DynamicStorage3D {
    type StorageType = DynamicStorage<Dim3D>;
    type Bits = Vec<u8>;

    fn bits(&self) -> &Self::Bits {
        &self.bits
    }

    fn bits_mut(&mut self) -> &mut Self::Bits {
        &mut self.bits
    }

    fn depth(&self) -> usize {
        self.depth
    }
}

impl VariableDepthStorage<Dim3D> for DynamicStorage3D {
    fn ancestor(&self, generations: std::num::NonZeroUsize) -> Option<Self> {
        // ancestor is probably something like: Take the last N bits, where N = (depth_depth + 3) / 4, and then trim off the
        // remaining bits within the last byte (like the _ case)
        if generations.get() > self.depth {
            return None;
        }
        let new_depth = self.depth - generations.get();
        let current_num_bytes = ((self.depth * 3) + 7) / 8;
        let new_num_bytes = ((new_depth * 3) + 7) / 8;
        let bytes_to_skip = current_num_bytes - new_num_bytes;
        let mut new_bits = self
            .bits
            .iter()
            .copied()
            .skip(bytes_to_skip)
            .collect::<Vec<_>>();
        let bits_in_first_byte = (new_depth * 3) % 8;
        if new_num_bytes > 0 && bits_in_first_byte > 0 {
            // Mask off the lowest bits in the first byte. These bits contain cells for levels > new_depth
            let leftover_bits_in_last_byte = 8 - bits_in_first_byte;
            // This is a mask with the lowest 'leftover_bits_in_last_byte' set to 0 and the rest to 1
            let mask = !((1 << leftover_bits_in_last_byte) - 1);
            new_bits[0] &= mask;
        }

        Some(Self {
            bits: new_bits,
            depth: new_depth,
        })
    }

    fn descendant(&self, cells: &[<Dim3D as Dimension>::Cell]) -> Option<Self> {
        let new_depth = self.depth + cells.len();
        let current_num_bytes = ((self.depth * 3) + 7) / 8;
        let new_num_bytes = ((new_depth * 3) + 7) / 8;
        let bytes_to_add = new_num_bytes - current_num_bytes;
        let new_bits = std::iter::repeat(0)
            .take(bytes_to_add)
            .chain(self.bits.iter().copied())
            .collect::<Vec<_>>();
        let mut ret = Self {
            bits: new_bits,
            depth: new_depth,
        };
        for (level_offset, cell) in cells.iter().enumerate() {
            unsafe {
                ret.set_cell_at_level_unchecked(self.depth as usize + level_offset, *cell);
            }
        }
        Some(ret)
    }
}

impl<'a> TryFrom<&'a [Octant]> for DynamicStorage3D {
    type Error = crate::Error;

    fn try_from(octants: &'a [Octant]) -> Result<Self, Self::Error> {
        let num_bytes = ((octants.len() * 3) + 7) / 8;
        let mut storage = Self {
            bits: vec![0; num_bytes],
            depth: octants.len(),
        };
        for (level, octant) in octants.iter().enumerate() {
            unsafe {
                storage.set_cell_at_level_unchecked(level, *octant);
            }
        }
        Ok(storage)
    }
}

impl<'a> IntoIterator for &'a DynamicStorage3D {
    type Item = Octant;
    type IntoIter = CellIter<'a, Dim3D, DynamicStorage3D>;

    fn into_iter(self) -> Self::IntoIter {
        CellIter {
            index: 0,
            storage: self,
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

    macro_rules! test_static {
        ($typename:ident, $modname:ident, $max_levels:literal) => {
            mod $modname {
                use super::*;
                use std::num::NonZeroUsize;

                const MAX_LEVELS: usize = $max_levels;

                #[test]
                fn default() {
                    let idx = $typename::default();
                    assert_eq!(0, idx.depth());
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
                    assert_eq!(MAX_LEVELS - 2, idx.depth());
                    for (level, expected_octant) in octants.iter().enumerate() {
                        assert_eq!(*expected_octant, idx.get_cell_at_level(level));
                    }
                }

                #[test]
                fn from_just_one_octant() {
                    let octants = vec![Octant::Seven];
                    let idx = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    assert_eq!(1, idx.depth());
                    assert_eq!(Octant::Seven, idx.get_cell_at_level(0));
                }

                #[test]
                fn set_cell() {
                    let octants = get_test_octants(MAX_LEVELS);
                    let mut idx = $typename::zeroed(MAX_LEVELS as u8).unwrap();
                    for level in 0..MAX_LEVELS {
                        idx.set_cell_at_level(level, octants[level]);
                    }

                    for level in 0..MAX_LEVELS {
                        assert_eq!(octants[level], idx.get_cell_at_level(level));
                    }
                }

                #[test]
                #[should_panic]
                fn set_cell_oob_panics() {
                    let mut idx = $typename::default();
                    // Tests both that OOB panics AND that a default-constructed Morton index with static storage has initial
                    // depth of zero
                    idx.set_cell_at_level(0, Octant::One);
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
                fn child() {
                    let idx = $typename::default();
                    let child_idx = idx
                        .child(Octant::Three)
                        .expect("child() should not return None");
                    assert_eq!(1, child_idx.depth());
                    assert_eq!(Octant::Three, child_idx.get_cell_at_level(0));
                }

                #[test]
                fn child_oob_yields_none() {
                    let octants = get_test_octants(MAX_LEVELS);
                    let idx = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    assert_eq!(None, idx.child(Octant::Zero));
                }

                #[test]
                fn descendant_multi_levels() {
                    let octants = get_test_octants(MAX_LEVELS);
                    let idx = $typename::try_from(&octants[0..MAX_LEVELS - 1])
                        .expect("Could not create Morton index from octants");
                    let child_idx = idx.descendant(&octants[(MAX_LEVELS - 1)..MAX_LEVELS]);
                    let expected_idx = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    assert_eq!(Some(expected_idx), child_idx);
                }

                #[test]
                fn descendant_too_far_yields_none() {
                    let octants = get_test_octants(MAX_LEVELS);
                    let idx = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    assert_eq!(None, idx.descendant(&[Octant::Zero]));
                }

                #[test]
                fn parent() {
                    let octants = get_test_octants(MAX_LEVELS);
                    let parent_octants = &octants[0..MAX_LEVELS - 1];
                    let child = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    let expected_parent = $typename::try_from(parent_octants)
                        .expect("Could not create Morton index from octants");
                    assert_eq!(Some(expected_parent), child.parent());
                }

                #[test]
                fn parent_of_root_yields_none() {
                    let root = $typename::default();
                    assert_eq!(None, root.parent());
                }

                #[test]
                fn ancestor_multi() {
                    let octants = get_test_octants(MAX_LEVELS);
                    let ancestor_generations = 1;
                    let parent_octants = &octants[0..(MAX_LEVELS - ancestor_generations)];
                    let child = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    let expected_parent = $typename::try_from(parent_octants)
                        .expect("Could not create Morton index from octants");
                    assert_eq!(
                        Some(expected_parent),
                        child.ancestor(NonZeroUsize::new(ancestor_generations).unwrap())
                    );
                }

                #[test]
                fn ancestor_to_root() {
                    let octants = get_test_octants(MAX_LEVELS);
                    let child = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    let expected_parent = $typename::default();
                    assert_eq!(
                        Some(expected_parent),
                        child.ancestor(NonZeroUsize::new(MAX_LEVELS).unwrap())
                    );
                }

                #[test]
                fn ancestor_too_far_yields_none() {
                    let octants = get_test_octants(MAX_LEVELS);
                    let child = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    assert_eq!(
                        None,
                        child.ancestor(NonZeroUsize::new(MAX_LEVELS + 1).unwrap())
                    );
                }

                #[test]
                fn ordering() {
                    let count = 128;
                    let mut rng = thread_rng();
                    let mut indices = (0..count)
                        .map(|_| {
                            let rnd_levels = rng.gen_range(0..MAX_LEVELS);
                            let octants = get_test_octants(rnd_levels);
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
                        let rountrip_idx =
                            $typename::from_grid_index(grid_index, MAX_LEVELS, octant_ordering)
                                .expect("Can't get Morton index from grid index");
                        assert_eq!(idx, rountrip_idx);
                    }
                }

                #[test]
                fn roundtrip_grid_index_with_odd_levels() {
                    let octants = get_test_octants(MAX_LEVELS - 1);
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
                        let rountrip_idx =
                            $typename::from_grid_index(grid_index, MAX_LEVELS - 1, octant_ordering)
                                .expect("Can't get Morton index from grid index");
                        assert_eq!(idx, rountrip_idx);
                    }
                }

                #[test]
                fn roundtrip_grid_index_with_one_level() {
                    let octants = get_test_octants(1);
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
                        let rountrip_idx =
                            $typename::from_grid_index(grid_index, 1, octant_ordering)
                                .expect("Can't get Morton index from grid index");
                        assert_eq!(idx, rountrip_idx);
                    }
                }
            }
        };
    }

    macro_rules! test_dynamic {
        ($typename:ident, $modname:ident) => {
            mod $modname {
                use super::*;
                use std::num::NonZeroUsize;

                #[test]
                fn default() {
                    let idx = $typename::default();
                    assert_eq!(0, idx.depth());
                }

                #[test]
                fn from_even_number_of_octants() {
                    let octants = get_test_octants(8);
                    let idx = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    assert_eq!(octants.len(), idx.depth());
                    for (level, expected_octant) in octants.iter().enumerate() {
                        assert_eq!(*expected_octant, idx.get_cell_at_level(level));
                    }
                }

                #[test]
                fn from_odd_number_of_octants() {
                    let octants = get_test_octants(9);
                    let idx = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    assert_eq!(octants.len(), idx.depth());
                    for (level, expected_octant) in octants.iter().enumerate() {
                        assert_eq!(*expected_octant, idx.get_cell_at_level(level));
                    }
                }

                #[test]
                fn set_cell() {
                    let depth = 8;
                    let octants = get_test_octants(depth);
                    let mut idx = $typename::zeroed(depth);
                    for level in 0..depth {
                        idx.set_cell_at_level(level, octants[level]);
                    }

                    for level in 0..depth {
                        assert_eq!(octants[level], idx.get_cell_at_level(level));
                    }
                }

                #[test]
                #[should_panic]
                fn set_cell_oob_panics() {
                    let mut idx = $typename::default();
                    // Tests both that OOB panics AND that a default-constructed Morton index with dynamic storage has initial
                    // depth of zero
                    idx.set_cell_at_level(0, Octant::One);
                }

                #[test]
                fn cells_iter() {
                    let octants = get_test_octants(8);
                    let idx = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    let collected_octants = idx.cells().collect::<Vec<_>>();
                    assert_eq!(octants.as_slice(), collected_octants.as_slice());
                }

                #[test]
                fn child() {
                    let idx = $typename::default();
                    let child_idx = idx
                        .child(Octant::Three)
                        .expect("child() should not return None");
                    assert_eq!(1, child_idx.depth());
                    assert_eq!(Octant::Three, child_idx.get_cell_at_level(0));
                }

                #[test]
                fn child_at_four_levels() {
                    let octants = get_test_octants(4);
                    let idx = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    let child_idx = idx
                        .child(Octant::Three)
                        .expect("child() should not return None");
                    assert_eq!(octants.len() + 1, child_idx.depth());
                    assert_eq!(Octant::Three, child_idx.get_cell_at_level(octants.len()));
                }

                #[test]
                fn descendant_multi_levels() {
                    let octants = get_test_octants(6);
                    let idx = $typename::try_from(&octants[0..4])
                        .expect("Could not create Morton index from octants");
                    let child_idx = idx.descendant(&octants[4..6]);
                    let expected_idx = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    assert_eq!(Some(expected_idx), child_idx);
                }

                #[test]
                fn parent_4() {
                    let depth = 4;
                    let octants = get_test_octants(depth);
                    let parent_octants = &octants[0..depth - 1];
                    let child = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    let expected_parent = $typename::try_from(parent_octants)
                        .expect("Could not create Morton index from octants");
                    assert_eq!(Some(expected_parent), child.parent());
                }

                #[test]
                fn parent_5() {
                    let depth = 5;
                    let octants = get_test_octants(depth);
                    let parent_octants = &octants[0..depth - 1];
                    let child = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    let expected_parent = $typename::try_from(parent_octants)
                        .expect("Could not create Morton index from octants");
                    assert_eq!(Some(expected_parent), child.parent());
                }

                #[test]
                fn parent_of_root_yields_none() {
                    let root = $typename::default();
                    assert_eq!(None, root.parent());
                }

                #[test]
                fn ancestor_multi() {
                    let octants = get_test_octants(8);
                    let ancestor_generations = 3;
                    let parent_octants = &octants[0..(8 - ancestor_generations)];
                    let child = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    let expected_parent = $typename::try_from(parent_octants)
                        .expect("Could not create Morton index from octants");
                    assert_eq!(
                        Some(expected_parent),
                        child.ancestor(NonZeroUsize::new(ancestor_generations).unwrap())
                    );
                }

                #[test]
                fn ancestor_to_root() {
                    let octants = get_test_octants(5);
                    let child = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    let expected_parent = $typename::default();
                    assert_eq!(
                        Some(expected_parent),
                        child.ancestor(NonZeroUsize::new(5).unwrap())
                    );
                }

                #[test]
                fn ancestor_too_far_yields_none() {
                    let octants = get_test_octants(4);
                    let child = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");
                    assert_eq!(None, child.ancestor(NonZeroUsize::new(5).unwrap()));
                }

                #[test]
                fn ordering() {
                    let count = 16;
                    let mut indices = (0..count)
                        .map(|_| {
                            let octants = get_test_octants(9);
                            $typename::try_from(octants.as_slice())
                                .expect("Could not create Morton index from octants")
                        })
                        .collect::<Vec<_>>();

                    indices.sort();

                    // Indices have to be sorted in ascending order, meaning that for indices i and j with i < j,
                    // each octant of index i must be <= the octant at the same level of index j, up until and including
                    // the first octant of i that is less than j
                    for idx in 0..(count - 1) {
                        let i = &indices[idx];
                        let j = &indices[idx + 1];
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
                    let octants = get_test_octants(32);
                    let idx = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");

                    for ordering in [
                        OctantOrdering::XYZ,
                        OctantOrdering::XZY,
                        OctantOrdering::YXZ,
                        OctantOrdering::YZX,
                        OctantOrdering::ZXY,
                        OctantOrdering::ZYX,
                    ] {
                        let grid_index = idx.to_grid_index(ordering);
                        let roundtrip_idx = $typename::from_grid_index(grid_index, 32, ordering);
                        assert_eq!(idx, roundtrip_idx);
                    }
                }

                #[test]
                fn roundtrip_grid_index_with_odd_levels() {
                    let octants = get_test_octants(29);
                    let idx = $typename::try_from(octants.as_slice())
                        .expect("Could not create Morton index from octants");

                    for ordering in [
                        OctantOrdering::XYZ,
                        OctantOrdering::XZY,
                        OctantOrdering::YXZ,
                        OctantOrdering::YZX,
                        OctantOrdering::ZXY,
                        OctantOrdering::ZYX,
                    ] {
                        let grid_index = idx.to_grid_index(ordering);
                        let roundtrip_idx = $typename::from_grid_index(grid_index, 29, ordering);
                        assert_eq!(idx, roundtrip_idx);
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

    test_static!(StaticMortonIndex3D8, static_morton_index_3d8, 2);
    test_static!(StaticMortonIndex3D16, static_morton_index_3d16, 5);
    test_static!(StaticMortonIndex3D32, static_morton_index_3d32, 10);
    test_static!(StaticMortonIndex3D64, static_morton_index_3d64, 21);
    test_static!(StaticMortonIndex3D128, static_morton_index_3d128, 42);

    test_dynamic!(DynamicMortonIndex3D, dynamic_morton_index_3d);
}
