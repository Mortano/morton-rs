use std::cmp::Ordering;
use std::fmt::Debug;
use std::hash::Hash;

use nalgebra::Vector2;

use crate::dimensions::{Dim2D, Dimension, Quadrant, QuadrantOrdering};
use crate::number::{add_zero_before_every_bit_u8, add_zero_behind_every_bit_u8, Bits, Endianness};
use crate::{
    CellIter, DynamicStorage, FixedDepthStorage, FixedStorageType, MortonIndex, MortonIndexNaming,
    StaticStorage, Storage, StorageType, VariableDepthMortonIndex, VariableDepthStorage,
};

/// A 2D Morton index with a fixed depth of 4 levels (using a single `u8` value as storage)
pub type FixedDepthMortonIndex2D8 = MortonIndex2D<FixedDepthStorage2D<u8>>;
/// A 2D Morton index with a fixed depth of 8 levels (using a single `u16` value as storage)
pub type FixedDepthMortonIndex2D16 = MortonIndex2D<FixedDepthStorage2D<u16>>;
/// A 2D Morton index with a fixed depth of 16 levels (using a single `u32` value as storage)
pub type FixedDepthMortonIndex2D32 = MortonIndex2D<FixedDepthStorage2D<u32>>;
/// A 2D Morton index with a fixed depth of 32 levels (using a single `u64` value as storage)
pub type FixedDepthMortonIndex2D64 = MortonIndex2D<FixedDepthStorage2D<u64>>;
/// A 2D Morton index with a fixed depth of 64 levels (using a single `u128` value as storage)
pub type FixedDepthMortonIndex2D128 = MortonIndex2D<FixedDepthStorage2D<u128>>;
/// A 2D Morton index with variable depth, but a statically-determined maximum depth of 4 levels (using a single `u8` value as storage)
pub type StaticMortonIndex2D8 = MortonIndex2D<StaticStorage2D<u8>>;
/// A 2D Morton index with variable depth, but a statically-determined maximum depth of 8 levels (using a single `u16` value as storage)
pub type StaticMortonIndex2D16 = MortonIndex2D<StaticStorage2D<u16>>;
/// A 2D Morton index with variable depth, but a statically-determined maximum depth of 16 levels (using a single `u32` value as storage)
pub type StaticMortonIndex2D32 = MortonIndex2D<StaticStorage2D<u32>>;
/// A 2D Morton index with variable depth, but a statically-determined maximum depth of 32 levels (using a single `u64` value as storage)
pub type StaticMortonIndex2D64 = MortonIndex2D<StaticStorage2D<u64>>;
/// A 2D Morton index with variable depth, but a statically-determined maximum depth of 64 levels (using a single `u128` value as storage)
pub type StaticMortonIndex2D128 = MortonIndex2D<StaticStorage2D<u128>>;
/// A 2D Morton index with variable depth. It uses a `Vec` as storage and can represent an unlimited number of cells
pub type DynamicMortonIndex2D = MortonIndex2D<DynamicStorage2D>;

/// A 2D Morton index. This represents a single node inside a quadtree. The depth of the node and the maximum storage
/// capacity of this type depend on the generic `Storage` type
#[derive(Default, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct MortonIndex2D<S: Storage<Dim2D>> {
    storage: S,
}

impl<S: Storage<Dim2D> + Clone> Clone for MortonIndex2D<S> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
        }
    }
}

impl<S: Storage<Dim2D> + Copy> Copy for MortonIndex2D<S> {}

impl<'a, S: Storage<Dim2D> + 'a> MortonIndex2D<S>
where
    &'a S: IntoIterator<Item = Quadrant>,
{
    pub fn cells(&'a self) -> <&'a S as IntoIterator>::IntoIter {
        self.storage.into_iter()
    }
}

impl<S: VariableDepthStorage<Dim2D>> VariableDepthMortonIndex for MortonIndex2D<S> {
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

impl<B: FixedStorageType> MortonIndex2D<FixedDepthStorage2D<B>> {
    /// Creates a new MortonIndex2D with fixed-depth storage from the given 2D grid index. The `grid_index` is assumed to
    /// represent a grid with a depth equal to `FixedDepthStorage2D<B>::MAX_LEVELS`.
    ///
    /// # Panics
    ///
    /// There is an edge-case in which the fixed depth of the `FixedDepthStorage2D<B>` is greater than what a single `usize`
    /// value in the `grid_index` can represent. In this case the code will panic.
    pub fn from_grid_index(
        grid_index: <Dim2D as Dimension>::GridIndex,
        ordering: QuadrantOrdering,
    ) -> Self {
        let fixed_depth = <FixedDepthStorage<Dim2D, B> as StorageType>::MAX_LEVELS;
        if fixed_depth > (std::mem::size_of::<usize>() * 8) {
            panic!(
                "Size of usize is too small for a fixed depth of {}",
                fixed_depth
            );
        }
        // Similar construction as compared to static storage, but we have a fixed depth

        let x_bits = unsafe { grid_index.x.get_bits(0..fixed_depth) };
        let y_bits = unsafe { grid_index.y.get_bits(0..fixed_depth) };

        let (lower_bits, higher_bits) = match ordering {
            QuadrantOrdering::XY => (x_bits, y_bits),
            QuadrantOrdering::YX => (y_bits, x_bits),
        };

        let mut bits = B::default();
        // Set bits in 8-bit chunks at a time
        let num_chunks = (fixed_depth + 3) / 4;
        for chunk_index in 0..num_chunks {
            let start_bit = chunk_index * 4;
            let end_bit = start_bit + 4;
            let lower_chunk = unsafe { lower_bits.get_bits(start_bit..end_bit) as u8 };
            let higher_chunk = unsafe { higher_bits.get_bits(start_bit..end_bit) as u8 };
            let chunk = add_zero_before_every_bit_u8(lower_chunk)
                | add_zero_behind_every_bit_u8(higher_chunk);
            unsafe {
                bits.set_bits(
                    (chunk_index * 8)..((chunk_index + 1) * 8),
                    B::from_u8(chunk),
                );
            }
        }

        Self {
            storage: FixedDepthStorage2D { bits },
        }
    }
}

impl<B: FixedStorageType> MortonIndex2D<StaticStorage2D<B>> {
    /// Returns a Morton index with the given `depth` where all cells are zeroed (i.e. representing `Quadrant::Zero`). If
    /// the `depth` is larger than the maximum depth of the `StaticStorage2D<B>`, `None` is returned
    pub fn zeroed(depth: u8) -> Option<Self> {
        if depth as usize > <StaticStorage<Dim2D, B> as StorageType>::MAX_LEVELS {
            return None;
        }
        Some(Self {
            storage: StaticStorage2D::<B> {
                bits: Default::default(),
                depth,
            },
        })
    }

    /// Creates a new MortonIndex2D with static storage from the given 2D grid index. The `grid_depth` parameter
    /// describes the 'depth' of the grid as per the equation: `N = 2 ^ grid_depth`, where `N` is the number of cells
    /// per axis in the grid. For example, a `32*32` grid has a `grid_depth` of `log2(32) = 5`. If `2 * grid_depth`
    /// exceeds the capacity of the static storage type `B`, this returns an error.
    pub fn from_grid_index(
        grid_index: <Dim2D as Dimension>::GridIndex,
        grid_depth: usize,
        ordering: QuadrantOrdering,
    ) -> Result<Self, crate::Error> {
        if grid_depth > (B::BITS / 2) {
            return Err(crate::Error::DepthLimitedExceeded {
                max_depth: B::BITS / 2,
            });
        }

        // the grid_depth is equal to the number of valid bits in the X and Y fields of grid_index
        // There is an explanation for the shifting in the corresponding implementation for the 3D case
        // in `morton_index_3d.rs`
        let shift_to_max_bits =
            <FixedDepthStorage<Dim2D, B> as StorageType>::MAX_LEVELS - grid_depth;
        let x_bits = unsafe { grid_index.x.get_bits(0..grid_depth) << shift_to_max_bits };
        let y_bits = unsafe { grid_index.y.get_bits(0..grid_depth) << shift_to_max_bits };

        let (lower_bits, higher_bits) = match ordering {
            QuadrantOrdering::XY => (x_bits, y_bits),
            QuadrantOrdering::YX => (y_bits, x_bits),
        };

        let mut bits = B::default();
        // Set bits in 8-bit chunks at a time
        let num_chunks = (grid_depth + 3) / 4;
        let max_chunks = (<FixedDepthStorage<Dim2D, B> as StorageType>::MAX_LEVELS + 3) / 4;
        let chunk_start = max_chunks - num_chunks;
        for chunk_index in (chunk_start..max_chunks).rev() {
            let start_bit = chunk_index * 4;
            let end_bit = start_bit + 4;
            let lower_chunk = unsafe { lower_bits.get_bits(start_bit..end_bit) as u8 };
            let higher_chunk = unsafe { higher_bits.get_bits(start_bit..end_bit) as u8 };
            let chunk = add_zero_before_every_bit_u8(lower_chunk)
                | add_zero_behind_every_bit_u8(higher_chunk);

            // chunk contains 8 valid bits at most
            let start_bit = chunk_index * 8;
            let end_bit = start_bit + 8;
            unsafe {
                bits.set_bits(start_bit..end_bit, B::from_u8(chunk));
            }
        }

        Ok(Self {
            storage: StaticStorage2D {
                bits,
                depth: grid_depth as u8,
            },
        })
    }
}

impl MortonIndex2D<DynamicStorage2D> {
    pub fn zeroed(depth: usize) -> Self {
        Self {
            storage: DynamicStorage2D::zeroed(depth),
        }
    }

    /// Creates a new MortonIndex2D with dynamic storage from the given 2D grid index. The `grid_depth` parameter
    /// describes the 'depth' of the grid as per the equation: `N = 2 ^ grid_depth`, where `N` is the number of cells
    /// per axis in the grid. For example, a `32*32` grid has a `grid_depth` of `log2(32) = 5`.
    pub fn from_grid_index(
        grid_index: <Dim2D as Dimension>::GridIndex,
        grid_depth: usize,
        ordering: QuadrantOrdering,
    ) -> Self {
        let max_bits = std::mem::size_of_val(&grid_index.x) * 8;
        if grid_depth > (max_bits / 2) {
            panic!(
                "2 * grid_depth ({}) must not exceed maximum number of bits ({}) in the grid index",
                grid_depth, max_bits
            );
        }
        // the grid_depth is equal to the number of valid bits in the X and Y fields of grid_index
        let x_bits = unsafe { grid_index.x.get_bits(0..grid_depth) };
        let y_bits = unsafe { grid_index.y.get_bits(0..grid_depth) };

        let (lower_bits, higher_bits) = match ordering {
            QuadrantOrdering::XY => (x_bits, y_bits),
            QuadrantOrdering::YX => (y_bits, x_bits),
        };

        // Expand bits by 2, then interleave. We do this in 4-bit chunks to prevent overflow
        let num_chunks = (grid_depth + 3) / 4;
        Self {
            storage: DynamicStorage2D {
                bits: (0..num_chunks)
                    .map(|chunk_index| {
                        // Same weird calculation as with StaticStorage2D, see lengthy comment there for explanation
                        let end_bit = grid_depth - (chunk_index * 4);
                        let start_bit = end_bit.saturating_sub(4);
                        let lower_chunk = unsafe { lower_bits.get_bits(start_bit..end_bit) as u8 };
                        let higher_chunk =
                            unsafe { higher_bits.get_bits(start_bit..end_bit) as u8 };
                        let chunk = add_zero_before_every_bit_u8(lower_chunk)
                            | add_zero_behind_every_bit_u8(higher_chunk);
                        // To get correct chunks with bit counts not divisible by 4, we have to set the right bits
                        // in the u8 value (similar to what we do in StaticStorage2D)
                        let bits_in_current_chunk = end_bit - start_bit;
                        let end_bit_in_index = 8;
                        let start_bit_in_index = end_bit_in_index - 2 * bits_in_current_chunk;
                        let mut bits: u8 = 0;
                        unsafe {
                            // Maybe set the bits in BigEndian instead ?! Might require mirroring the bits of 'chunk'...
                            // Now everything is weird, the 'less-depth' thing works, but not the one with full depth?!?!
                            // Maybe use a simple implementation with 'set_cell_at_level_unchecked' as a reference?
                            bits.set_bits(start_bit_in_index..end_bit_in_index, chunk);
                        }
                        bits
                    })
                    .rev()
                    .collect(),
                depth: grid_depth,
            },
        }
    }
}

impl<'a, S: Storage<Dim2D> + TryFrom<&'a [Quadrant], Error = crate::Error>> TryFrom<&'a [Quadrant]>
    for MortonIndex2D<S>
{
    type Error = crate::Error;

    fn try_from(value: &'a [Quadrant]) -> Result<Self, Self::Error> {
        S::try_from(value).map(|storage| Self { storage })
    }
}

// TODO Don't know how to write this implementation in terms of TryFrom<&'a [Quadrant]> ...
// impl<'a, S: Storage<Dim2D> + TryFrom<&'a [Quadrant], Error = crate::Error>, const N: usize>
//     TryFrom<[Quadrant; N]> for MortonIndex2D<S>
// {
//     type Error = crate::Error;

//     fn try_from(value: [Quadrant; N]) -> Result<Self, Self::Error> {
//         let slice = value.as_slice();
//         S::try_from(slice).map(|storage| Self { storage })
//     }
// }

impl<S: Storage<Dim2D>> MortonIndex for MortonIndex2D<S> {
    type Dimension = Dim2D;

    fn get_cell_at_level(&self, level: usize) -> Quadrant {
        if level >= self.depth() {
            panic!("level must not be >= self.depth()");
        }
        unsafe { self.storage.get_cell_at_level_unchecked(level) }
    }

    unsafe fn get_cell_at_level_unchecked(&self, level: usize) -> Quadrant {
        self.storage.get_cell_at_level_unchecked(level)
    }

    fn set_cell_at_level(&mut self, level: usize, cell: Quadrant) {
        if level >= self.depth() {
            panic!("level must not be >= self.depth()");
        }
        unsafe { self.storage.set_cell_at_level_unchecked(level, cell) }
    }

    unsafe fn set_cell_at_level_unchecked(&mut self, level: usize, cell: Quadrant) {
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
                let grid_index = self.to_grid_index(QuadrantOrdering::default());
                format!("{}-{}-{}", self.depth(), grid_index.x, grid_index.y)
            }
        }
    }

    fn to_grid_index(&self, ordering: QuadrantOrdering) -> <Dim2D as Dimension>::GridIndex {
        let mut index = Vector2::new(0, 0);
        for level in 0..self.depth() {
            let quadrant = unsafe { self.get_cell_at_level_unchecked(level) };
            let quadrant_index = ordering.to_index(quadrant);
            let shift = self.depth() - level - 1;
            index.x |= quadrant_index.x << shift;
            index.y |= quadrant_index.y << shift;
        }
        index
    }
}

/// Storage for a 2D Morton index that always stores a Morton index with a fixed depth (fixed number of levels)
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FixedDepthStorage2D<B: FixedStorageType> {
    bits: B,
}

impl<B: FixedStorageType> Storage<Dim2D> for FixedDepthStorage2D<B> {
    type StorageType = FixedDepthStorage<Dim2D, B>;
    type Bits = B;

    fn bits(&self) -> &Self::Bits {
        &self.bits
    }

    fn bits_mut(&mut self) -> &mut Self::Bits {
        &mut self.bits
    }

    fn depth(&self) -> usize {
        FixedDepthStorage::<Dim2D, B>::MAX_LEVELS
    }
}

impl<'a, B: FixedStorageType> TryFrom<&'a [Quadrant]> for FixedDepthStorage2D<B> {
    type Error = crate::Error;

    fn try_from(quadrants: &'a [Quadrant]) -> Result<Self, Self::Error> {
        if quadrants.len() > <FixedDepthStorage<Dim2D, B> as StorageType>::MAX_LEVELS {
            return Err(crate::Error::DepthLimitedExceeded {
                max_depth: <FixedDepthStorage<Dim2D, B> as StorageType>::MAX_LEVELS,
            });
        }
        let mut ret: Self = Default::default();
        for (level, cell) in quadrants.iter().enumerate() {
            unsafe {
                ret.set_cell_at_level_unchecked(level, *cell);
            }
        }
        Ok(ret)
    }
}

impl<'a, B: FixedStorageType> IntoIterator for &'a FixedDepthStorage2D<B> {
    type Item = Quadrant;
    type IntoIter = CellIter<'a, Dim2D, FixedDepthStorage2D<B>>;

    fn into_iter(self) -> Self::IntoIter {
        CellIter {
            index: 0,
            storage: &self,
            _phantom: Default::default(),
        }
    }
}

/// Morton index storage which can store an arbitrary number of levels up to a static maximum number of levels
/// as defined by the `FixedStorageType` `B`. As opposed to the `FixedDepthStorage`, two Morton indices with this
/// storage type can represent two nodes with different depth levels. Since the internal bit representation is
/// fixed at compile-time by the `FixedStorageType` parameter, the maximum depth is limited at compile-time as
/// well, so some of the methods that construct or extend Morton indices with this storage might return `None` if
/// the internal storage capacity is not sufficient.
#[derive(Default, Debug, Clone, Copy, Hash)]
pub struct StaticStorage2D<B: FixedStorageType> {
    bits: B,
    depth: u8,
}

impl<B: FixedStorageType> Storage<Dim2D> for StaticStorage2D<B> {
    type StorageType = StaticStorage<Dim2D, B>;
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

impl<B: FixedStorageType> PartialOrd for StaticStorage2D<B> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<B: FixedStorageType> Ord for StaticStorage2D<B> {
    fn cmp(&self, other: &Self) -> Ordering {
        let end_bit = B::BITS;
        let self_start_bit = end_bit - (self.depth() * 2);
        let other_start_bit = end_bit - (other.depth() * 2);
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

impl<B: FixedStorageType> PartialEq for StaticStorage2D<B> {
    fn eq(&self, other: &Self) -> bool {
        // TODO Should we maybe only compare self.bits.bits(0..2*self.depth)?
        self.bits == other.bits && self.depth == other.depth
    }
}

impl<B: FixedStorageType> Eq for StaticStorage2D<B> {}

impl<B: FixedStorageType> VariableDepthStorage<Dim2D> for StaticStorage2D<B> {
    fn ancestor(&self, generations: std::num::NonZeroUsize) -> Option<Self> {
        if generations.get() > self.depth as usize {
            return None;
        }

        let mut ret = *self;
        // Zero out all quadrants below the new depth
        let new_depth = self.depth - generations.get() as u8;
        for level in new_depth..self.depth {
            unsafe {
                ret.set_cell_at_level_unchecked(level as usize, Quadrant::Zero);
            }
        }
        ret.depth = new_depth;
        Some(ret)
    }

    fn descendant(&self, cells: &[<Dim2D as Dimension>::Cell]) -> Option<Self> {
        let new_depth = self.depth as usize + cells.len();
        if new_depth >= <StaticStorage<Dim2D, B> as StorageType>::MAX_LEVELS {
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

impl<'a, B: FixedStorageType> TryFrom<&'a [Quadrant]> for StaticStorage2D<B> {
    type Error = crate::Error;

    fn try_from(quadrants: &'a [Quadrant]) -> Result<Self, Self::Error> {
        if quadrants.len() > <StaticStorage<Dim2D, B> as StorageType>::MAX_LEVELS {
            return Err(crate::Error::DepthLimitedExceeded {
                max_depth: <StaticStorage<Dim2D, B> as StorageType>::MAX_LEVELS,
            });
        }
        let mut ret: Self = Default::default();
        for (level, cell) in quadrants.iter().enumerate() {
            unsafe {
                ret.set_cell_at_level_unchecked(level, *cell);
            }
        }
        ret.depth = quadrants.len() as u8;
        Ok(ret)
    }
}

impl<'a, B: FixedStorageType> IntoIterator for &'a StaticStorage2D<B> {
    type Item = Quadrant;
    type IntoIter = CellIter<'a, Dim2D, StaticStorage2D<B>>;

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
/// potentially worse performance than the `StaticStorage2D` and `FixedStorage2D` variants.
///
/// Note: There is no guarantee about endianness on any of the storage types, the `DynamicStorage2D` might
/// not even have a consistent endianness, so it's not valid to assume any order of the internal bits.
#[derive(Default, Debug, Clone, Hash, PartialEq, Eq)]
pub struct DynamicStorage2D {
    bits: Vec<u8>,
    depth: usize,
}

impl DynamicStorage2D {
    pub fn zeroed(depth: usize) -> Self {
        Self {
            bits: (0..=(depth / 4)).map(|_| 0_u8).collect(),
            depth,
        }
    }

    fn get_cell_at_level_or_none(&self, level: usize) -> Option<Quadrant> {
        if level >= self.depth.into() {
            None
        } else {
            // Is safe because of level check above
            unsafe { Some(self.get_cell_at_level_unchecked(level)) }
        }
    }
}

impl PartialOrd for DynamicStorage2D {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DynamicStorage2D {
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

impl Storage<Dim2D> for DynamicStorage2D {
    type StorageType = DynamicStorage<Dim2D>;
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

impl VariableDepthStorage<Dim2D> for DynamicStorage2D {
    fn ancestor(&self, generations: std::num::NonZeroUsize) -> Option<Self> {
        // ancestor is probably something like: Take the last N bits, where N = (depth_depth + 3) / 4, and then trim off the
        // remaining bits within the last byte (like the _ case)
        if generations.get() > self.depth {
            return None;
        }
        let new_depth = self.depth - generations.get();
        let current_num_bytes = (self.depth + 3) / 4;
        let new_num_bytes = (new_depth + 3) / 4;
        let bytes_to_skip = current_num_bytes - new_num_bytes;
        let mut new_bits = self
            .bits
            .iter()
            .copied()
            .skip(bytes_to_skip)
            .collect::<Vec<_>>();
        if new_num_bytes > 0 {
            // Mask off the lowest 2*K bits (K=new_depth % 2)
            let leftover_bits_in_last_byte = ((4 - new_depth) % 4) * 2;
            // This is a mask with the lowest 'leftover_bits_in_last_byte' set to 0 and the rest to 1
            let mask = !((1 << leftover_bits_in_last_byte) - 1);
            new_bits[0] &= mask;
        }

        Some(Self {
            bits: new_bits,
            depth: new_depth,
        })
    }

    fn descendant(&self, cells: &[<Dim2D as Dimension>::Cell]) -> Option<Self> {
        // descendant is harder: figure out how many cells fit into the current bit, and push the rest as new bytes
        // maybe implement it as a simple 'grow bits so that bits.len() * 4 can fit all levels', and then call 'set_cell_at_level_unchecked'
        // to set all the cells
        let new_depth = self.depth + cells.len();
        let current_num_bytes = (self.depth + 3) / 4;
        let new_num_bytes = (new_depth + 3) / 4;
        let bytes_to_add = new_num_bytes - current_num_bytes;
        let new_bits = self
            .bits
            .iter()
            .copied()
            .chain(std::iter::repeat(0).take(bytes_to_add))
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

impl<'a> TryFrom<&'a [Quadrant]> for DynamicStorage2D {
    type Error = crate::Error;

    fn try_from(quadrants: &'a [Quadrant]) -> Result<Self, Self::Error> {
        let num_bytes = (quadrants.len() + 3) / 4;
        let mut storage = Self {
            bits: vec![0; num_bytes],
            depth: quadrants.len(),
        };
        for (level, quadrant) in quadrants.iter().enumerate() {
            unsafe {
                storage.set_cell_at_level_unchecked(level, *quadrant);
            }
        }
        Ok(storage)
    }
}

impl<'a> IntoIterator for &'a DynamicStorage2D {
    type Item = Quadrant;
    type IntoIter = CellIter<'a, Dim2D, DynamicStorage2D>;

    fn into_iter(self) -> Self::IntoIter {
        CellIter {
            index: 0,
            storage: self,
            _phantom: Default::default(),
        }
    }
}

// 6 possible conversions:
// FixedDepth -> Static   (safe if Static Bits >= FixedDepth Bits)
// FixedDepth -> Dynamic  (always safe)
// Static -> Dynamic      (always safe)
// Static -> FixedDepth   (safe if FixedDepth Bits >= Static Bits, has to zero-fill. Maybe doesn't make sense?)
// Dynamic -> FixedDepth  (safe if FixedDepth Bits >= Dynamic depth, has to zero-fill. Maybe doesn't make sense?)
// Dynamic -> Static      (safe if Static Bits >= Dynamic depth)

impl<B: FixedStorageType> From<MortonIndex2D<FixedDepthStorage2D<B>>>
    for MortonIndex2D<StaticStorage2D<B>>
{
    fn from(fixed_index: MortonIndex2D<FixedDepthStorage2D<B>>) -> Self {
        Self {
            storage: StaticStorage2D {
                bits: fixed_index.storage.bits,
                depth: StaticStorage::<Dim2D, B>::MAX_LEVELS as u8,
            },
        }
    }
}

impl<B: FixedStorageType> From<MortonIndex2D<FixedDepthStorage2D<B>>>
    for MortonIndex2D<DynamicStorage2D>
{
    fn from(fixed_index: MortonIndex2D<FixedDepthStorage2D<B>>) -> Self {
        let native_bits = unsafe { fixed_index.storage.bits.as_u8_slice() };
        #[cfg(target_endian = "little")]
        let bits = native_bits.to_owned();
        #[cfg(target_endian = "big")]
        let bits = native_bits.iter().copied().rev().collect::<Vec<_>>();
        Self {
            storage: DynamicStorage2D {
                bits,
                depth: FixedDepthStorage::<Dim2D, B>::MAX_LEVELS,
            },
        }
    }
}

impl<B: FixedStorageType> From<MortonIndex2D<StaticStorage2D<B>>>
    for MortonIndex2D<FixedDepthStorage2D<B>>
{
    fn from(static_index: MortonIndex2D<StaticStorage2D<B>>) -> Self {
        Self {
            storage: FixedDepthStorage2D {
                bits: static_index.storage.bits,
            },
        }
    }
}

impl<B: FixedStorageType> From<MortonIndex2D<StaticStorage2D<B>>>
    for MortonIndex2D<DynamicStorage2D>
{
    fn from(fixed_index: MortonIndex2D<StaticStorage2D<B>>) -> Self {
        let native_bits = unsafe { fixed_index.storage.bits.as_u8_slice() };
        #[cfg(target_endian = "little")]
        let bits = native_bits.to_owned();
        #[cfg(target_endian = "big")]
        let bits = native_bits.iter().copied().rev().collect::<Vec<_>>();
        Self {
            storage: DynamicStorage2D {
                bits,
                depth: StaticStorage::<Dim2D, B>::MAX_LEVELS,
            },
        }
    }
}

// The conversion from a dynamic index to other index types are dependent of a runtime parameter (the depth() of the dynamic
// index) and as such these conversions can fail at runtime. Which is why dynamic conversions only implement `TryFrom`

impl<B: FixedStorageType> TryFrom<MortonIndex2D<DynamicStorage2D>>
    for MortonIndex2D<FixedDepthStorage2D<B>>
{
    type Error = crate::Error;

    fn try_from(value: MortonIndex2D<DynamicStorage2D>) -> Result<Self, Self::Error> {
        if value.depth() > FixedDepthStorage::<Dim2D, B>::MAX_LEVELS {
            Err(crate::Error::DepthLimitedExceeded {
                max_depth: FixedDepthStorage::<Dim2D, B>::MAX_LEVELS,
            })
        } else {
            // DynamicStorage2D stores its cells as BigEndian, FixedDepthStorage2D uses the native endianness of the current
            // machine
            #[cfg(target_endian = "little")]
            let endianness = Endianness::LittleEndian;
            #[cfg(target_endian = "big")]
            let endianness = Endianness::BigEndian;
            let bytes = value.storage.bits.as_slice();
            Ok(Self {
                storage: FixedDepthStorage2D {
                    bits: unsafe { B::from_u8_slice(bytes, endianness) },
                },
            })
        }
    }
}

impl<B: FixedStorageType> TryFrom<MortonIndex2D<DynamicStorage2D>>
    for MortonIndex2D<StaticStorage2D<B>>
{
    type Error = crate::Error;

    fn try_from(value: MortonIndex2D<DynamicStorage2D>) -> Result<Self, Self::Error> {
        if value.depth() > FixedDepthStorage::<Dim2D, B>::MAX_LEVELS {
            Err(crate::Error::DepthLimitedExceeded {
                max_depth: FixedDepthStorage::<Dim2D, B>::MAX_LEVELS,
            })
        } else {
            // TODO There might be a way to optimize this code by directly setting the bits of the StaticStorage2D, however
            // it involves potentially extracting less than a byte worth of data from the DynamicStorage
            let mut ret: MortonIndex2D<StaticStorage2D<B>> = Default::default();
            for level in 0..value.depth() {
                // Safe because of depth check above
                unsafe {
                    ret.set_cell_at_level_unchecked(
                        level,
                        value.get_cell_at_level_unchecked(level),
                    );
                }
            }
            ret.storage.depth = value.depth() as u8;
            Ok(ret)
        }
    }
}

// TODO Bit truncation / extension conversions, e.g.:
//   FixedDepth<u8> to FixedDepth<u64> (extension)
//   FixedDepth<u64> to FixedDepth<u8> (truncation)

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    /// Returns a bunch of test quadrants
    fn get_test_quadrants(count: usize) -> Vec<Quadrant> {
        let mut rng = thread_rng();
        (0..count)
            .map(|_| {
                let num: usize = rng.gen_range(0..4);
                Quadrant::try_from(num).unwrap()
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
                        assert_eq!(Quadrant::Zero, idx.get_cell_at_level(level));
                    }
                }

                #[test]
                fn from_quadrants() {
                    let quadrants = get_test_quadrants(MAX_LEVELS);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");
                    assert_eq!(quadrants.len(), idx.depth());
                    for (level, expected_quadrant) in quadrants.iter().enumerate() {
                        assert_eq!(*expected_quadrant, idx.get_cell_at_level(level));
                    }
                }

                #[test]
                fn from_too_many_quadrants_fails() {
                    let quadrants = get_test_quadrants(MAX_LEVELS + 1);
                    let res = $typename::try_from(quadrants.as_slice());
                    assert!(res.is_err());
                }

                #[test]
                fn from_fewer_quadrants_than_max_depth() {
                    let quadrants = get_test_quadrants(MAX_LEVELS - 2);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");
                    assert_eq!(MAX_LEVELS, idx.depth());
                    for (level, expected_quadrant) in quadrants.iter().enumerate() {
                        assert_eq!(*expected_quadrant, idx.get_cell_at_level(level));
                    }
                    // Remaining quadrants should be zero-initialized!
                    assert_eq!(Quadrant::Zero, idx.get_cell_at_level(MAX_LEVELS - 2));
                    assert_eq!(Quadrant::Zero, idx.get_cell_at_level(MAX_LEVELS - 1));
                }

                #[test]
                fn set_cell() {
                    let quadrants = get_test_quadrants(MAX_LEVELS);
                    let mut idx = $typename::default();
                    for level in 0..MAX_LEVELS {
                        idx.set_cell_at_level(level, quadrants[level]);
                    }

                    for level in 0..MAX_LEVELS {
                        assert_eq!(
                            quadrants[level],
                            idx.get_cell_at_level(level),
                            "Wrong quadrants at level {} in index {}",
                            level,
                            idx.to_string(MortonIndexNaming::CellConcatenation)
                        );
                    }
                }

                #[test]
                #[should_panic]
                fn set_cell_oob_panics() {
                    let mut idx = $typename::default();
                    idx.set_cell_at_level(MAX_LEVELS, Quadrant::One);
                }

                #[test]
                fn cells_iter() {
                    let quadrants = get_test_quadrants(MAX_LEVELS);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");
                    let collected_quadrants = idx.cells().collect::<Vec<_>>();
                    assert_eq!(quadrants.as_slice(), collected_quadrants.as_slice());
                }

                #[test]
                fn ordering() {
                    let count = 16;
                    let mut indices = (0..count)
                        .map(|_| {
                            let quadrants = get_test_quadrants(MAX_LEVELS);
                            $typename::try_from(quadrants.as_slice())
                                .expect("Could not create Morton index from quadrants")
                        })
                        .collect::<Vec<_>>();

                    indices.sort();

                    // Indices have to be sorted in ascending order, meaning that for indices i and j with i < j,
                    // each quadrant of index i must be <= the quadrant at the same level of index j, up until and including
                    // the first quadrant of i that is less than j
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
                    let quadrants = get_test_quadrants(MAX_LEVELS);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");

                    let grid_index_xy = idx.to_grid_index(QuadrantOrdering::XY);
                    let roundtrip_idx_xy =
                        $typename::from_grid_index(grid_index_xy, QuadrantOrdering::XY);
                    assert_eq!(idx, roundtrip_idx_xy);

                    let grid_index_yx = idx.to_grid_index(QuadrantOrdering::YX);
                    let roundtrip_idx_yx =
                        $typename::from_grid_index(grid_index_yx, QuadrantOrdering::YX);
                    assert_eq!(idx, roundtrip_idx_yx);
                }
            }
        };
    }

    macro_rules! test_static {
        ($typename:ident, $modname:ident, $max_levels:literal) => {
            mod $modname {
                use super::*;

                const MAX_LEVELS: usize = $max_levels;

                #[test]
                fn default() {
                    let idx = $typename::default();
                    assert_eq!(0, idx.depth());
                }

                #[test]
                fn from_quadrants() {
                    let quadrants = get_test_quadrants(MAX_LEVELS);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");
                    assert_eq!(quadrants.len(), idx.depth());
                    for (level, expected_quadrant) in quadrants.iter().enumerate() {
                        assert_eq!(*expected_quadrant, idx.get_cell_at_level(level));
                    }
                }

                #[test]
                fn from_too_many_quadrants_fails() {
                    let quadrants = get_test_quadrants(MAX_LEVELS + 1);
                    let res = $typename::try_from(quadrants.as_slice());
                    assert!(res.is_err());
                }

                #[test]
                fn from_fewer_quadrants_than_max_depth() {
                    let quadrants = get_test_quadrants(MAX_LEVELS - 2);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");
                    assert_eq!(MAX_LEVELS - 2, idx.depth());
                    for (level, expected_quadrant) in quadrants.iter().enumerate() {
                        assert_eq!(*expected_quadrant, idx.get_cell_at_level(level));
                    }
                }

                #[test]
                fn set_cell() {
                    let quadrants = get_test_quadrants(MAX_LEVELS);
                    let mut idx = $typename::zeroed(MAX_LEVELS as u8).unwrap();
                    for level in 0..MAX_LEVELS {
                        idx.set_cell_at_level(level, quadrants[level]);
                    }

                    for level in 0..MAX_LEVELS {
                        assert_eq!(quadrants[level], idx.get_cell_at_level(level));
                    }
                }

                #[test]
                #[should_panic]
                fn set_cell_oob_panics() {
                    let mut idx = $typename::default();
                    // Tests both that OOB panics AND that a default-constructed Morton index with static storage has initial
                    // depth of zero
                    idx.set_cell_at_level(0, Quadrant::One);
                }

                #[test]
                fn cells_iter() {
                    let quadrants = get_test_quadrants(MAX_LEVELS);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");
                    let collected_quadrants = idx.cells().collect::<Vec<_>>();
                    assert_eq!(quadrants.as_slice(), collected_quadrants.as_slice());
                }

                #[test]
                fn child() {
                    let idx = $typename::default();
                    let child_idx = idx
                        .child(Quadrant::Three)
                        .expect("child() should not return None");
                    assert_eq!(1, child_idx.depth());
                    assert_eq!(Quadrant::Three, child_idx.get_cell_at_level(0));
                }

                #[test]
                fn child_oob_yields_none() {
                    let quadrants = get_test_quadrants(MAX_LEVELS);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");
                    assert_eq!(None, idx.child(Quadrant::Zero));
                }

                #[test]
                fn parent() {
                    let quadrants = get_test_quadrants(MAX_LEVELS);
                    let parent_quadrants = &quadrants[0..MAX_LEVELS - 1];
                    let child = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");
                    let expected_parent = $typename::try_from(parent_quadrants)
                        .expect("Could not create Morton index from quadrants");
                    assert_eq!(Some(expected_parent), child.parent());
                }

                #[test]
                fn parent_of_root_yields_none() {
                    let root = $typename::default();
                    assert_eq!(None, root.parent());
                }

                #[test]
                fn ordering() {
                    let count = 128;
                    let mut rng = thread_rng();
                    let mut indices = (0..count)
                        .map(|_| {
                            let rnd_levels = rng.gen_range(0..MAX_LEVELS);
                            let quadrants = get_test_quadrants(rnd_levels);
                            $typename::try_from(quadrants.as_slice())
                                .expect("Could not create Morton index from quadrants")
                        })
                        .collect::<Vec<_>>();

                    indices.sort();

                    // Indices have to be sorted in ascending order, meaning that for indices i and j with i < j,
                    // each quadrant of index i must be <= the quadrant at the same level of index j, up until and including
                    // the first quadrant of i that is less than j
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
                    let quadrants = get_test_quadrants(MAX_LEVELS);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");

                    let grid_index_xy = idx.to_grid_index(QuadrantOrdering::XY);
                    let roundtrip_idx_xy =
                        $typename::from_grid_index(grid_index_xy, MAX_LEVELS, QuadrantOrdering::XY)
                            .expect("Can't get Morton index from grid index");
                    assert_eq!(idx, roundtrip_idx_xy);

                    let grid_index_yx = idx.to_grid_index(QuadrantOrdering::YX);
                    let roundtrip_idx_yx =
                        $typename::from_grid_index(grid_index_yx, MAX_LEVELS, QuadrantOrdering::YX)
                            .expect("Can't get Morton index from grid index");
                    assert_eq!(idx, roundtrip_idx_yx);
                }

                #[test]
                fn roundtrip_grid_index_with_odd_levels() {
                    let quadrants = get_test_quadrants(MAX_LEVELS - 3);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");

                    let grid_index_xy = idx.to_grid_index(QuadrantOrdering::XY);
                    let roundtrip_idx_xy = $typename::from_grid_index(
                        grid_index_xy,
                        MAX_LEVELS - 3,
                        QuadrantOrdering::XY,
                    )
                    .expect("Can't get Morton index from grid index");
                    assert_eq!(idx, roundtrip_idx_xy);

                    let grid_index_yx = idx.to_grid_index(QuadrantOrdering::YX);
                    let roundtrip_idx_yx = $typename::from_grid_index(
                        grid_index_yx,
                        MAX_LEVELS - 3,
                        QuadrantOrdering::YX,
                    )
                    .expect("Can't get Morton index from grid index");
                    assert_eq!(idx, roundtrip_idx_yx);
                }

                #[test]
                fn roundtrip_grid_index_with_one_level() {
                    let quadrants = get_test_quadrants(1);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");

                    let grid_index_xy = idx.to_grid_index(QuadrantOrdering::XY);
                    let roundtrip_idx_xy =
                        $typename::from_grid_index(grid_index_xy, 1, QuadrantOrdering::XY)
                            .expect("Can't get Morton index from grid index");
                    assert_eq!(idx, roundtrip_idx_xy);

                    let grid_index_yx = idx.to_grid_index(QuadrantOrdering::YX);
                    let roundtrip_idx_yx =
                        $typename::from_grid_index(grid_index_yx, 1, QuadrantOrdering::YX)
                            .expect("Can't get Morton index from grid index");
                    assert_eq!(idx, roundtrip_idx_yx);
                }
            }
        };
    }

    macro_rules! test_dynamic {
        ($typename:ident, $modname:ident) => {
            mod $modname {
                use super::*;

                #[test]
                fn default() {
                    let idx = $typename::default();
                    assert_eq!(0, idx.depth());
                }

                #[test]
                fn from_even_number_of_quadrants() {
                    let quadrants = get_test_quadrants(8);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");
                    assert_eq!(quadrants.len(), idx.depth());
                    for (level, expected_quadrant) in quadrants.iter().enumerate() {
                        assert_eq!(*expected_quadrant, idx.get_cell_at_level(level));
                    }
                }

                #[test]
                fn from_odd_number_of_quadrants() {
                    let quadrants = get_test_quadrants(9);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");
                    assert_eq!(quadrants.len(), idx.depth());
                    for (level, expected_quadrant) in quadrants.iter().enumerate() {
                        assert_eq!(*expected_quadrant, idx.get_cell_at_level(level));
                    }
                }

                #[test]
                fn set_cell() {
                    let depth = 8;
                    let quadrants = get_test_quadrants(depth);
                    let mut idx = $typename::zeroed(depth);
                    for level in 0..depth {
                        idx.set_cell_at_level(level, quadrants[level]);
                    }

                    for level in 0..depth {
                        assert_eq!(quadrants[level], idx.get_cell_at_level(level));
                    }
                }

                #[test]
                #[should_panic]
                fn set_cell_oob_panics() {
                    let mut idx = $typename::default();
                    // Tests both that OOB panics AND that a default-constructed Morton index with static storage has initial
                    // depth of zero
                    idx.set_cell_at_level(0, Quadrant::One);
                }

                #[test]
                fn cells_iter() {
                    let quadrants = get_test_quadrants(8);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");
                    let collected_quadrants = idx.cells().collect::<Vec<_>>();
                    assert_eq!(quadrants.as_slice(), collected_quadrants.as_slice());
                }

                #[test]
                fn child() {
                    let idx = $typename::default();
                    let child_idx = idx
                        .child(Quadrant::Three)
                        .expect("child() should not return None");
                    assert_eq!(1, child_idx.depth());
                    assert_eq!(Quadrant::Three, child_idx.get_cell_at_level(0));
                }

                #[test]
                fn child_at_four_levels() {
                    let quadrants = get_test_quadrants(4);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");
                    let child_idx = idx
                        .child(Quadrant::Three)
                        .expect("child() should not return None");
                    assert_eq!(quadrants.len() + 1, child_idx.depth());
                    assert_eq!(
                        Quadrant::Three,
                        child_idx.get_cell_at_level(quadrants.len())
                    );
                }

                #[test]
                fn parent_4() {
                    let depth = 4;
                    let quadrants = get_test_quadrants(depth);
                    let parent_quadrants = &quadrants[0..depth - 1];
                    let child = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");
                    let expected_parent = $typename::try_from(parent_quadrants)
                        .expect("Could not create Morton index from quadrants");
                    assert_eq!(Some(expected_parent), child.parent());
                }

                #[test]
                fn parent_5() {
                    let depth = 5;
                    let quadrants = get_test_quadrants(depth);
                    let parent_quadrants = &quadrants[0..depth - 1];
                    let child = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");
                    let expected_parent = $typename::try_from(parent_quadrants)
                        .expect("Could not create Morton index from quadrants");
                    assert_eq!(Some(expected_parent), child.parent());
                }

                #[test]
                fn parent_of_root_yields_none() {
                    let root = $typename::default();
                    assert_eq!(None, root.parent());
                }

                #[test]
                fn ordering() {
                    let count = 16;
                    let mut indices = (0..count)
                        .map(|_| {
                            let quadrants = get_test_quadrants(9);
                            $typename::try_from(quadrants.as_slice())
                                .expect("Could not create Morton index from quadrants")
                        })
                        .collect::<Vec<_>>();

                    indices.sort();

                    // Indices have to be sorted in ascending order, meaning that for indices i and j with i < j,
                    // each quadrant of index i must be <= the quadrant at the same level of index j, up until and including
                    // the first quadrant of i that is less than j
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
                    let quadrants = get_test_quadrants(32);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");

                    let grid_index_xy = idx.to_grid_index(QuadrantOrdering::XY);
                    let roundtrip_idx_xy =
                        $typename::from_grid_index(grid_index_xy, 32, QuadrantOrdering::XY);
                    assert_eq!(idx, roundtrip_idx_xy);

                    let grid_index_yx = idx.to_grid_index(QuadrantOrdering::YX);
                    let roundtrip_idx_yx =
                        $typename::from_grid_index(grid_index_yx, 32, QuadrantOrdering::YX);
                    assert_eq!(idx, roundtrip_idx_yx);
                }

                #[test]
                fn roundtrip_grid_index_with_odd_levels() {
                    let quadrants = get_test_quadrants(29);
                    let idx = $typename::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");

                    let grid_index_xy = idx.to_grid_index(QuadrantOrdering::XY);
                    let roundtrip_idx_xy =
                        $typename::from_grid_index(grid_index_xy, 29, QuadrantOrdering::XY);
                    assert_eq!(idx, roundtrip_idx_xy);

                    let grid_index_yx = idx.to_grid_index(QuadrantOrdering::YX);
                    let roundtrip_idx_yx =
                        $typename::from_grid_index(grid_index_yx, 29, QuadrantOrdering::YX);
                    assert_eq!(idx, roundtrip_idx_yx);
                }
            }
        };
    }

    macro_rules! test_conversions {
        ($max_levels:literal, $datatype:ident, $modname:ident) => {
            mod $modname {
                use super::*;

                type FixedType = MortonIndex2D<FixedDepthStorage2D<$datatype>>;
                type StaticType = MortonIndex2D<StaticStorage2D<$datatype>>;
                type DynamicType = MortonIndex2D<DynamicStorage2D>;

                #[test]
                fn convert_fixed_to_static() {
                    let quadrants = get_test_quadrants($max_levels);
                    let fixed_index = FixedType::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");

                    let static_index: StaticType = fixed_index.into();

                    assert_eq!($max_levels, static_index.depth());
                    let expected_cells = fixed_index.cells().collect::<Vec<_>>();
                    let actual_cells = static_index.cells().collect::<Vec<_>>();
                    assert_eq!(expected_cells, actual_cells);
                }

                #[test]
                fn convert_fixed_to_dynamic() {
                    let quadrants = get_test_quadrants($max_levels);
                    let fixed_index = FixedType::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");

                    let dynamic_index: DynamicType = fixed_index.into();

                    assert_eq!($max_levels, dynamic_index.depth());
                    let expected_cells = fixed_index.cells().collect::<Vec<_>>();
                    let actual_cells = dynamic_index.cells().collect::<Vec<_>>();
                    assert_eq!(expected_cells, actual_cells);
                }

                #[test]
                fn convert_static_to_fixed() {
                    let quadrants = get_test_quadrants($max_levels);
                    let static_index = StaticType::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");

                    let fixed_index: FixedType = static_index.into();

                    let expected_cells = static_index.cells().collect::<Vec<_>>();
                    let actual_cells = fixed_index.cells().collect::<Vec<_>>();
                    assert_eq!(expected_cells, actual_cells);
                }

                #[test]
                fn convert_static_to_fixed_with_fewer_levels() {
                    let quadrants = get_test_quadrants($max_levels / 2);
                    let static_index = StaticType::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");

                    let fixed_index: FixedType = static_index.into();

                    // The remaining levels should be zero-filled in a conversion from static to fixed-depth index
                    let append_cells = std::iter::once(Quadrant::Zero)
                        .cycle()
                        .take($max_levels / 2);
                    let expected_cells =
                        static_index.cells().chain(append_cells).collect::<Vec<_>>();
                    let actual_cells = fixed_index.cells().collect::<Vec<_>>();
                    assert_eq!(expected_cells, actual_cells);
                }

                #[test]
                fn convert_static_to_dynamic() {
                    let quadrants = get_test_quadrants($max_levels);
                    let static_index = StaticType::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");

                    let dynamic_index: DynamicType = static_index.into();

                    assert_eq!(static_index.depth(), dynamic_index.depth());
                    let expected_cells = static_index.cells().collect::<Vec<_>>();
                    let actual_cells = dynamic_index.cells().collect::<Vec<_>>();
                    assert_eq!(expected_cells, actual_cells);
                }

                #[test]
                fn convert_dynamic_to_fixed() {
                    let quadrants = get_test_quadrants($max_levels);
                    let dynamic_index = DynamicType::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");

                    let fixed_index: FixedType = dynamic_index
                        .clone()
                        .try_into()
                        .expect("Can't convert dynamic index to fixed depth index");

                    let expected_cells = dynamic_index.cells().collect::<Vec<_>>();
                    let actual_cells = fixed_index.cells().collect::<Vec<_>>();
                    assert_eq!(expected_cells, actual_cells);
                }

                #[test]
                fn convert_dynamic_to_fixed_with_fewer_levels() {
                    // Subtracting an odd number from the max cells to test edge cases
                    let quadrants = get_test_quadrants($max_levels - 3);
                    let dynamic_index = DynamicType::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");

                    let fixed_index: FixedType = dynamic_index
                        .clone()
                        .try_into()
                        .expect("Can't convert dynamic index to fixed depth index");

                    let padding_cells = std::iter::repeat(Quadrant::Zero).take(3);
                    let expected_cells = dynamic_index
                        .cells()
                        .chain(padding_cells)
                        .collect::<Vec<_>>();
                    let actual_cells = fixed_index.cells().collect::<Vec<_>>();
                    assert_eq!(expected_cells, actual_cells);
                }

                #[test]
                fn convert_dynamic_to_static() {
                    let quadrants = get_test_quadrants($max_levels);
                    let dynamic_index = DynamicType::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");

                    let static_index: StaticType = dynamic_index
                        .clone()
                        .try_into()
                        .expect("Can't convert dynamic index to fixed depth index");

                    assert_eq!(static_index.depth(), dynamic_index.depth());
                    let expected_cells = dynamic_index.cells().collect::<Vec<_>>();
                    let actual_cells = static_index.cells().collect::<Vec<_>>();
                    assert_eq!(expected_cells, actual_cells);
                }

                #[test]
                fn convert_dynamic_to_static_with_fewer_levels() {
                    // Subtracting an odd number from the max cells to test edge cases
                    let quadrants = get_test_quadrants($max_levels - 3);
                    let dynamic_index = DynamicType::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");

                    let static_index: StaticType = dynamic_index
                        .clone()
                        .try_into()
                        .expect("Can't convert dynamic index to fixed depth index");

                    assert_eq!(static_index.depth(), dynamic_index.depth());
                    let expected_cells = dynamic_index.cells().collect::<Vec<_>>();
                    let actual_cells = static_index.cells().collect::<Vec<_>>();
                    assert_eq!(expected_cells, actual_cells);
                }
            }
        };
    }

    test_fixed_depth!(FixedDepthMortonIndex2D8, fixed_depth_morton_index_2d8, 4);
    test_fixed_depth!(FixedDepthMortonIndex2D16, fixed_depth_morton_index_2d16, 8);
    test_fixed_depth!(FixedDepthMortonIndex2D32, fixed_depth_morton_index_2d32, 16);
    test_fixed_depth!(FixedDepthMortonIndex2D64, fixed_depth_morton_index_2d64, 32);
    test_fixed_depth!(
        FixedDepthMortonIndex2D128,
        fixed_depth_morton_index_2d128,
        64
    );

    test_static!(StaticMortonIndex2D8, static_morton_index_2d8, 4);
    test_static!(StaticMortonIndex2D16, static_morton_index_2d16, 8);
    test_static!(StaticMortonIndex2D32, static_morton_index_2d32, 16);
    test_static!(StaticMortonIndex2D64, static_morton_index_2d64, 32);
    test_static!(StaticMortonIndex2D128, static_morton_index_2d128, 64);

    test_dynamic!(DynamicMortonIndex2D, dynamic_morton_index_2d);

    test_conversions!(4, u8, conversions_2d8);
    test_conversions!(8, u16, conversions_2d16);
    test_conversions!(16, u32, conversions_2d32);
    test_conversions!(32, u64, conversions_2d64);
    test_conversions!(64, u128, conversions_2d128);

    // Test the from_grid_index functions separately with predefined values
    #[test]
    fn from_grid_index_fixed_u16() {
        // u16 means we have a 2^8 * 2^8 grid
        // We specify the grid_index as a bit pattern here, because it makes it easier to see what value
        // we are expecting!
        let grid_index = Vector2::new(0b11000101, 0b01101100);
        let morton_index_xy =
            FixedDepthMortonIndex2D16::from_grid_index(grid_index, QuadrantOrdering::XY);
        let morton_index_yx =
            FixedDepthMortonIndex2D16::from_grid_index(grid_index, QuadrantOrdering::YX);

        // These two values are simply the bits of the X and Y index interleaved, one time with X in the lower bits,
        // one time with Y in the lower bits
        let _expected_xy = 0b01111000_10110001;
        let _expected_yx = 0b10110100_01110010;
        // From there, we take every two bits and convert them into an index, either little endian or big endian, to get
        // the expected quadrants. The first quadrant in XY order is `01` and hence `One`, in YX order it is `10` and hence
        // `Two`
        let expected_quadrants_xy = vec![
            Quadrant::One,
            Quadrant::Three,
            Quadrant::Two,
            Quadrant::Zero,
            Quadrant::Two,
            Quadrant::Three,
            Quadrant::Zero,
            Quadrant::One,
        ];
        let expected_quadrants_yx = vec![
            Quadrant::Two,
            Quadrant::Three,
            Quadrant::One,
            Quadrant::Zero,
            Quadrant::One,
            Quadrant::Three,
            Quadrant::Zero,
            Quadrant::Two,
        ];

        let xy_quadrants = morton_index_xy.cells().collect::<Vec<_>>();
        let yx_quadrants = morton_index_yx.cells().collect::<Vec<_>>();

        assert_eq!(expected_quadrants_xy, xy_quadrants);
        assert_eq!(expected_quadrants_yx, yx_quadrants);
    }

    #[test]
    fn from_grid_index_static_u16() {
        // TODO Test fails, I think because we set the quadrants in the wrong order. Level 5 means we want `0b11000` instead
        // of `0b00101`
        // Then again, my intuition might be wrong, 5 bits of `0b11000101` IS `0b00101` on little endian. This is counter-intuitive
        // with how it looks in the code, we might have to specify this in the documentation!
        let grid_index = Vector2::new(0b11000101, 0b01101100);
        let index_full_depth_xy =
            StaticMortonIndex2D16::from_grid_index(grid_index, 8, QuadrantOrdering::XY)
                .expect("Could not create Morton index from grid index");
        let index_full_depth_yx =
            StaticMortonIndex2D16::from_grid_index(grid_index, 8, QuadrantOrdering::YX)
                .expect("Could not create Morton index from grid index");

        let index_less_depth_xy =
            StaticMortonIndex2D16::from_grid_index(grid_index, 5, QuadrantOrdering::XY)
                .expect("Could not create Morton index from grid index");
        let index_less_depth_yx =
            StaticMortonIndex2D16::from_grid_index(grid_index, 5, QuadrantOrdering::YX)
                .expect("Could not create Morton index from grid index");

        // For quadrant values, see the from_grid_index_fixed_u16 test
        let expected_quadrants_xy = vec![
            Quadrant::One,
            Quadrant::Three,
            Quadrant::Two,
            Quadrant::Zero,
            Quadrant::Two,
            Quadrant::Three,
            Quadrant::Zero,
            Quadrant::One,
        ];
        let expected_quadrants_yx = vec![
            Quadrant::Two,
            Quadrant::Three,
            Quadrant::One,
            Quadrant::Zero,
            Quadrant::One,
            Quadrant::Three,
            Quadrant::Zero,
            Quadrant::Two,
        ];

        assert_eq!(
            expected_quadrants_xy,
            index_full_depth_xy.cells().collect::<Vec<_>>()
        );
        assert_eq!(
            expected_quadrants_yx,
            index_full_depth_yx.cells().collect::<Vec<_>>()
        );

        // !!! We don't expect quadrants 0..5 because the from_grid_index function only looks at the `grid_depth` LOWEST
        // bits of the grid index, which is equal to the deeper quadrants of a larger index!
        assert_eq!(
            expected_quadrants_xy[3..8].to_owned(),
            index_less_depth_xy.cells().collect::<Vec<_>>()
        );
        assert_eq!(
            expected_quadrants_yx[3..8].to_owned(),
            index_less_depth_yx.cells().collect::<Vec<_>>()
        );
    }

    #[test]
    fn from_grid_index_dynamic() {
        // Let's use a grid_depth of 8 as with the other tests
        // We specify the grid_index as a bit pattern here, because it makes it easier to see what value
        // we are expecting!
        let grid_depth = 8;
        let grid_index = Vector2::new(0b11000101, 0b01101100);
        let morton_index_xy =
            DynamicMortonIndex2D::from_grid_index(grid_index, grid_depth, QuadrantOrdering::XY);
        let morton_index_yx =
            DynamicMortonIndex2D::from_grid_index(grid_index, grid_depth, QuadrantOrdering::YX);

        let index_less_depth_xy =
            DynamicMortonIndex2D::from_grid_index(grid_index, 5, QuadrantOrdering::XY);
        let index_less_depth_yx =
            DynamicMortonIndex2D::from_grid_index(grid_index, 5, QuadrantOrdering::YX);

        // These two values are simply the bits of the X and Y index interleaved, one time with X in the lower bits,
        // one time with Y in the lower bits
        let _expected_xy = 0b01111000_10110001;
        let _expected_yx = 0b10110100_01110010;
        // From there, we take every two bits and convert them into an index, either little endian or big endian, to get
        // the expected quadrants. The first quadrant in XY order is `01` and hence `One`, in YX order it is `10` and hence
        // `Two`
        let expected_quadrants_xy = vec![
            Quadrant::One,
            Quadrant::Three,
            Quadrant::Two,
            Quadrant::Zero,
            Quadrant::Two,
            Quadrant::Three,
            Quadrant::Zero,
            Quadrant::One,
        ];
        let expected_quadrants_yx = vec![
            Quadrant::Two,
            Quadrant::Three,
            Quadrant::One,
            Quadrant::Zero,
            Quadrant::One,
            Quadrant::Three,
            Quadrant::Zero,
            Quadrant::Two,
        ];

        let xy_quadrants = morton_index_xy.cells().collect::<Vec<_>>();
        let yx_quadrants = morton_index_yx.cells().collect::<Vec<_>>();

        assert_eq!(expected_quadrants_xy, xy_quadrants);
        assert_eq!(expected_quadrants_yx, yx_quadrants);

        assert_eq!(
            expected_quadrants_xy[3..8].to_owned(),
            index_less_depth_xy.cells().collect::<Vec<_>>()
        );
        assert_eq!(
            expected_quadrants_yx[3..8].to_owned(),
            index_less_depth_yx.cells().collect::<Vec<_>>()
        );
    }
}
