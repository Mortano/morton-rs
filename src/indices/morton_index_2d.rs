use std::cmp::Ordering;
use std::fmt::Debug;
use std::hash::Hash;

use nalgebra::Vector2;

use crate::dimensions::{Dim2D, Dimension, Quadrant, QuadrantOrdering};
use crate::number::Bits;
use crate::{MortonIndex, MortonIndexNaming};

pub type FixedDepthMortonIndex2D8 = MortonIndex2D<FixedDepthStorage2D<u8>>;
pub type FixedDepthMortonIndex2D16 = MortonIndex2D<FixedDepthStorage2D<u16>>;
pub type FixedDepthMortonIndex2D32 = MortonIndex2D<FixedDepthStorage2D<u32>>;
pub type FixedDepthMortonIndex2D64 = MortonIndex2D<FixedDepthStorage2D<u64>>;
pub type FixedDepthMortonIndex2D128 = MortonIndex2D<FixedDepthStorage2D<u128>>;

pub type StaticMortonIndex2D8 = MortonIndex2D<StaticStorage2D<u8>>;
pub type StaticMortonIndex2D16 = MortonIndex2D<StaticStorage2D<u16>>;
pub type StaticMortonIndex2D32 = MortonIndex2D<StaticStorage2D<u32>>;
pub type StaticMortonIndex2D64 = MortonIndex2D<StaticStorage2D<u64>>;
pub type StaticMortonIndex2D128 = MortonIndex2D<StaticStorage2D<u128>>;

pub type DynamicMortonIndex = MortonIndex2D<DynamicStorage2D>;

pub trait Storage2D: Default + PartialOrd + Ord + PartialEq + Eq + Debug + Hash {
    fn max_depth() -> Option<usize>;
    fn try_from_quadrants(quadrants: &[Quadrant]) -> Result<Self, crate::Error>;

    fn depth(&self) -> usize;
    unsafe fn get_cell_at_level_unchecked(&self, level: usize) -> Quadrant;
    unsafe fn set_cell_at_level_unchecked(&mut self, level: usize, cell: Quadrant);
}

pub trait VariableDepthStorage2D: Storage2D {
    fn max_depth() -> Option<usize> {
        None
    }

    fn parent(&self) -> Option<Self>;
    fn child(&self, quadrant: Quadrant) -> Option<Self>;
}

#[derive(Default, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct MortonIndex2D<S: Storage2D> {
    storage: S,
}

impl<S: Storage2D + Clone> Clone for MortonIndex2D<S> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
        }
    }
}

impl<S: Storage2D + Copy> Copy for MortonIndex2D<S> {}

impl<'a, S: Storage2D + 'a> MortonIndex2D<S>
where
    &'a S: IntoIterator<Item = Quadrant>,
{
    pub fn cells(&'a self) -> <&'a S as IntoIterator>::IntoIter {
        self.storage.into_iter()
    }
}

impl<S: VariableDepthStorage2D> MortonIndex2D<S> {
    /// Returns a Morton index for the child `quadrant` of this Morton index. If this index is already at
    /// the maximum depth, `None` is returned instead
    pub fn child(&self, quadrant: Quadrant) -> Option<Self> {
        self.storage.child(quadrant).map(|storage| Self { storage })
    }

    /// Returns a Morton index for the parent node of this Morton index. If this index represents the root
    /// node, `None` is returned instead
    pub fn parent(&self) -> Option<Self> {
        self.storage.parent().map(|storage| Self { storage })
    }
}

impl<B: FixedStorageType> MortonIndex2D<StaticStorage2D<B>> {
    /// Returns a Morton index with the given `depth` where all cells are zeroed (i.e. representing `Quadrant::Zero`). If
    /// the `depth` is larger than the maximum depth of the `StaticStorage2D<B>`, `None` is returned
    pub fn zeroed(depth: u8) -> Option<Self> {
        if depth as usize > StaticStorage2D::<B>::MAX_LEVELS {
            return None;
        }
        Some(Self {
            storage: StaticStorage2D::<B> {
                bits: Default::default(),
                depth,
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
}

impl<S: Storage2D> TryFrom<&[Quadrant]> for MortonIndex2D<S> {
    type Error = crate::Error;

    fn try_from(value: &[Quadrant]) -> Result<Self, Self::Error> {
        S::try_from_quadrants(value).map(|storage| Self { storage })
    }
}

impl<S: Storage2D, const N: usize> TryFrom<[Quadrant; N]> for MortonIndex2D<S> {
    type Error = crate::Error;

    fn try_from(value: [Quadrant; N]) -> Result<Self, Self::Error> {
        S::try_from_quadrants(value.as_slice()).map(|storage| Self { storage })
    }
}

impl<S: Storage2D> MortonIndex for MortonIndex2D<S> {
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

pub trait FixedStorageType:
    Bits + Copy + Clone + PartialEq + Eq + PartialOrd + Ord + Default + Debug + Hash
{
}

impl<B: Bits + Copy + Clone + PartialEq + Eq + PartialOrd + Ord + Default + Debug + Hash>
    FixedStorageType for B
{
}

/// Storage for a 2D Morton index that always stores a Morton index with a fixed depth (fixed number of levels)
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FixedDepthStorage2D<B: FixedStorageType> {
    bits: B,
}

impl<B: FixedStorageType> FixedDepthStorage2D<B> {
    /// Maximum number of levels that can be represented with this `FixedDepthStorage2D`. The level depends on the number of bits
    /// that the `B` generic parameter can store
    const MAX_LEVELS: usize = B::BITS / 2;
}

impl<B: FixedStorageType> Storage2D for FixedDepthStorage2D<B> {
    fn max_depth() -> Option<usize> {
        Some(Self::MAX_LEVELS)
    }

    fn try_from_quadrants(quadrants: &[Quadrant]) -> Result<Self, crate::Error> {
        if quadrants.len() > Self::MAX_LEVELS {
            return Err(crate::Error::DepthLimitedExceeded {
                max_depth: Self::MAX_LEVELS,
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

    fn depth(&self) -> usize {
        Self::MAX_LEVELS
    }

    unsafe fn get_cell_at_level_unchecked(&self, level: usize) -> Quadrant {
        let bits = self.bits.get_bits(2 * level..2 * (level + 1)).as_u8() as usize;
        Quadrant::try_from(bits).unwrap()
    }

    unsafe fn set_cell_at_level_unchecked(&mut self, level: usize, cell: Quadrant) {
        let cell_index: usize = cell.into();
        self.bits
            .set_bits(2 * level..2 * (level + 1), B::from_u8(cell_index as u8));
    }
}

impl<'a, B: FixedStorageType> IntoIterator for &'a FixedDepthStorage2D<B> {
    type Item = Quadrant;
    type IntoIter = CellIter2D<'a, FixedDepthStorage2D<B>>;

    fn into_iter(self) -> Self::IntoIter {
        CellIter2D {
            index: 0,
            storage: &self,
        }
    }
}

pub struct CellIter2D<'a, S: Storage2D> {
    storage: &'a S,
    index: usize,
}

impl<S: Storage2D> Iterator for CellIter2D<'_, S> {
    type Item = Quadrant;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.storage.depth() {
            return None;
        }
        let index = self.index;
        self.index += 1;
        unsafe { Some(self.storage.get_cell_at_level_unchecked(index)) }
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

impl<B: FixedStorageType> StaticStorage2D<B> {
    const MAX_LEVELS: usize = if B::BITS > 512 {
        // Max representable value of the `u8` parameter `depth`. We could go larger by using some dependent type
        // for storing `depth`, but it's probably not worth it at the moment
        256
    } else {
        B::BITS / 2
    };

    fn get_cell_at_level_or_none(&self, level: usize) -> Option<Quadrant> {
        if level >= self.depth.into() {
            None
        } else {
            // Is safe because of level check above
            unsafe { Some(self.get_cell_at_level_unchecked(level)) }
        }
    }
}

impl<B: FixedStorageType> PartialOrd for StaticStorage2D<B> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<B: FixedStorageType> Ord for StaticStorage2D<B> {
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

impl<B: FixedStorageType> PartialEq for StaticStorage2D<B> {
    fn eq(&self, other: &Self) -> bool {
        // TODO Should we maybe only compare self.bits.bits(0..2*self.depth)?
        self.bits == other.bits && self.depth == other.depth
    }
}

impl<B: FixedStorageType> Eq for StaticStorage2D<B> {}

impl<B: FixedStorageType> Storage2D for StaticStorage2D<B> {
    fn max_depth() -> Option<usize> {
        Some(Self::MAX_LEVELS)
    }

    fn try_from_quadrants(quadrants: &[Quadrant]) -> Result<Self, crate::Error> {
        if quadrants.len() > Self::MAX_LEVELS {
            return Err(crate::Error::DepthLimitedExceeded {
                max_depth: Self::MAX_LEVELS,
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

    fn depth(&self) -> usize {
        self.depth.into()
    }

    unsafe fn get_cell_at_level_unchecked(&self, level: usize) -> Quadrant {
        let bits = self.bits.get_bits(2 * level..2 * (level + 1)).as_u8() as usize;
        Quadrant::try_from(bits).unwrap()
    }

    unsafe fn set_cell_at_level_unchecked(&mut self, level: usize, cell: Quadrant) {
        let cell_index: usize = cell.into();
        self.bits
            .set_bits(2 * level..2 * (level + 1), B::from_u8(cell_index as u8));
    }
}

impl<B: FixedStorageType> VariableDepthStorage2D for StaticStorage2D<B> {
    fn child(&self, quadrant: Quadrant) -> Option<Self> {
        if self.depth() == StaticStorage2D::<B>::MAX_LEVELS {
            return None;
        }

        let mut ret = *self;
        // Safe because of depth check above
        unsafe {
            ret.set_cell_at_level_unchecked(self.depth(), quadrant);
        }
        ret.depth += 1;
        Some(ret)
    }

    fn parent(&self) -> Option<Self> {
        if self.depth() == 0 {
            return None;
        }

        let mut ret = *self;
        // Zero out the lowest quadrant to make sure that the new value does not have bits set at a higher position than
        // what depth() indicates!
        unsafe {
            ret.set_cell_at_level_unchecked(self.depth() - 1, Quadrant::Zero);
        }
        ret.depth -= 1;
        Some(ret)
    }
}

impl<'a, B: FixedStorageType> IntoIterator for &'a StaticStorage2D<B> {
    type Item = Quadrant;
    type IntoIter = CellIter2D<'a, StaticStorage2D<B>>;

    fn into_iter(self) -> Self::IntoIter {
        CellIter2D {
            index: 0,
            storage: self,
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

impl Storage2D for DynamicStorage2D {
    fn max_depth() -> Option<usize> {
        None
    }

    fn try_from_quadrants(quadrants: &[Quadrant]) -> Result<Self, crate::Error> {
        // Chunk into groups of 4 and combine those into u8 values
        let bits = quadrants
            .chunks(4)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(0_u8, |accum, (idx, quadrant)| {
                        let quadrant_index: usize = quadrant.into();
                        // maybe subtract from 6 so that low levels are stored in the more significant bits?
                        // does it matter though? As long as sorting works?
                        accum | ((quadrant_index as u8) << (2 * idx))
                    })
            })
            .collect::<Vec<_>>();
        Ok(Self {
            bits,
            depth: quadrants.len(),
        })
    }

    fn depth(&self) -> usize {
        self.depth
    }

    unsafe fn get_cell_at_level_unchecked(&self, level: usize) -> Quadrant {
        let byte_index = level / 4;
        let start_bit = (level % 4) * 2;
        let end_bit = start_bit + 2;
        let quadrant_index = self.bits[byte_index].get_bits(start_bit..end_bit) as usize;
        quadrant_index.try_into().unwrap()
    }

    unsafe fn set_cell_at_level_unchecked(&mut self, level: usize, cell: Quadrant) {
        let byte_index = level / 4;
        let start_bit = (level % 4) * 2;
        let end_bit = start_bit + 2;
        self.bits[byte_index].set_bits(start_bit..end_bit, cell.index() as u8);
    }
}

impl VariableDepthStorage2D for DynamicStorage2D {
    fn parent(&self) -> Option<Self> {
        match self.depth {
            0 => None,
            depth if depth % 4 == 1 => Some(Self {
                bits: self
                    .bits
                    .iter()
                    .copied()
                    .take(self.bits.len() - 1)
                    .collect(),
                depth: self.depth - 1,
            }),
            _ => {
                let mut ret = self.clone();
                unsafe {
                    ret.set_cell_at_level_unchecked(self.depth - 1, Quadrant::Zero);
                }
                ret.depth -= 1;
                Some(ret)
            }
        }
    }

    fn child(&self, quadrant: Quadrant) -> Option<Self> {
        match self.depth {
            depth if depth % 4 == 0 => {
                let mut ret = self.clone();
                ret.bits.push(quadrant.index() as u8);
                ret.depth += 1;
                Some(ret)
            }
            _ => {
                let mut ret = self.clone();
                unsafe {
                    ret.set_cell_at_level_unchecked(self.depth, quadrant);
                }
                ret.depth += 1;
                Some(ret)
            }
        }
    }
}

impl<'a> IntoIterator for &'a DynamicStorage2D {
    type Item = Quadrant;
    type IntoIter = CellIter2D<'a, DynamicStorage2D>;

    fn into_iter(self) -> Self::IntoIter {
        CellIter2D {
            index: 0,
            storage: self,
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
                depth: StaticStorage2D::<B>::MAX_LEVELS as u8,
            },
        }
    }
}

impl<B: FixedStorageType> From<MortonIndex2D<FixedDepthStorage2D<B>>>
    for MortonIndex2D<DynamicStorage2D>
{
    fn from(fixed_index: MortonIndex2D<FixedDepthStorage2D<B>>) -> Self {
        Self {
            storage: DynamicStorage2D {
                // TODO It would be epic if this is correct, but it has to be tested
                bits: unsafe { fixed_index.storage.bits.as_vec_u8() },
                depth: FixedDepthStorage2D::<B>::MAX_LEVELS,
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
        Self {
            storage: DynamicStorage2D {
                // TODO It would be epic if this is correct, but it has to be tested
                bits: unsafe { fixed_index.storage.bits.as_vec_u8() },
                depth: StaticStorage2D::<B>::MAX_LEVELS,
            },
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
                        assert_eq!(quadrants[level], idx.get_cell_at_level(level));
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
                fn convert_static_to_fixed() {
                    let quadrants = get_test_quadrants($max_levels);
                    let static_index = StaticType::try_from(quadrants.as_slice())
                        .expect("Could not create Morton index from quadrants");

                    let fixed_index: FixedType = static_index.into();

                    let expected_cells = static_index.cells().collect::<Vec<_>>();
                    let actual_cells = fixed_index.cells().collect::<Vec<_>>();
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

    test_dynamic!(DynamicMortonIndex, dynamic_morton_index_2d);

    test_conversions!(4, u8, conversions_2d8);
    test_conversions!(8, u16, conversions_2d16);
    test_conversions!(16, u32, conversions_2d32);
    test_conversions!(32, u64, conversions_2d64);
    test_conversions!(64, u128, conversions_2d128);
}
