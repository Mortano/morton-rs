use std::cmp::Ordering;
use std::fmt::Debug;
use std::hash::Hash;

use crate::MortonIndex;
use crate::dimensions::{Quadrant, Dim2D, Dimension};
use crate::number::Bits;

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

pub trait Storage2D : Default + PartialOrd + Ord + PartialEq + Eq + Debug + Hash {
    fn fixed_depth() -> Option<usize>;
    fn max_depth() -> Option<usize>;
    fn try_from_quadrants(quadrants: &[Quadrant]) -> Result<Self, crate::Error>;

    fn depth(&self) -> usize;
    unsafe fn get_cell_at_level_unchecked(&self, level: usize) -> Quadrant;
    unsafe fn set_cell_at_level_unchecked(&mut self, level: usize, cell: Quadrant);
}

#[derive(Default, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct MortonIndex2D<S: Storage2D> {
    storage: S,
}

impl<S: Storage2D + Clone> Clone for MortonIndex2D<S> {
    fn clone(&self) -> Self {
        Self { storage: self.storage.clone() }
    }
}

impl<S: Storage2D + Copy> Copy for MortonIndex2D<S> {}

impl<'a, S: Storage2D + 'a> MortonIndex2D<S> where &'a S : IntoIterator<Item = Quadrant> {
    pub fn cells(&'a self) -> <&'a S as IntoIterator>::IntoIter {
        self.storage.into_iter()
    }
}

impl <B: FixedStorageType> MortonIndex2D<StaticStorage2D<B>> {
    /// Returns a Morton index with the given `depth` where all cells are zeroed (i.e. representing `Quadrant::Zero`). If 
    /// the `depth` is larger than the maximum depth of the `StaticStorage2D<B>`, `None` is returned
    pub fn zeroed(depth: u8) -> Option<Self> {
        if depth as usize > StaticStorage2D::<B>::MAX_LEVELS {
            return None;
        }
        Some(Self {
            storage: StaticStorage2D::<B>{
                bits: Default::default(),
                depth,
            }
        })
    }

    /// Returns a Morton index for the child `quadrant` of this Morton index. If this index is already at
    /// the maximum depth, `None` is returned instead
    pub fn child(&self, quadrant: Quadrant) -> Option<Self> {
        self.storage.child(quadrant).map(|storage| Self {
            storage,
        })
    }
}

impl<S: Storage2D> TryFrom<&[Quadrant]> for MortonIndex2D<S> {
    type Error = crate::Error;

    fn try_from(value: &[Quadrant]) -> Result<Self, Self::Error> {
        S::try_from_quadrants(value).map(|storage| {
            Self {
                storage,
            }
        })
    }
}

impl <S:Storage2D> MortonIndex for MortonIndex2D<S> {
    type Dimension = Dim2D;

    fn get_cell_at_level(&self, level: usize) -> Quadrant {
        if level >= self.depth() {
            panic!("level must not be >= self.depth()");
        }
        unsafe {
            self.storage.get_cell_at_level_unchecked(level)
        }
    }

    unsafe fn get_cell_at_level_unchecked(&self, level: usize) -> Quadrant {
        self.storage.get_cell_at_level_unchecked(level)
    }

    fn set_cell_at_level(&mut self, level: usize, cell: Quadrant) {
        if level >= self.depth() {
            panic!("level must not be >= self.depth()");
        }
        unsafe {
            self.storage.set_cell_at_level_unchecked(level, cell)
        }
    }
    
    unsafe fn set_cell_at_level_unchecked(&mut self, level: usize, cell: Quadrant) {
        self.storage.set_cell_at_level_unchecked(level, cell)
    }

    fn depth(&self) -> usize {
        self.storage.depth()
    }

    fn to_string(&self, _naming: crate::MortonIndexNaming) -> String {
        todo!()
    }

    fn to_grid_index(&self) -> <Dim2D as Dimension>::GridIndex {
        todo!()
    }
}

pub trait FixedStorageType : Bits + Copy + Clone + PartialEq + Eq + PartialOrd + Ord + Default + Debug + Hash {}

impl <B: Bits + Copy + Clone + PartialEq + Eq + PartialOrd + Ord + Default + Debug + Hash> FixedStorageType for B {}

/// Storage for a 2D Morton index that always stores a Morton index with a fixed depth (fixed number of levels)
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FixedDepthStorage2D<B: FixedStorageType> {
    bits: B,
}

impl<B: FixedStorageType> FixedDepthStorage2D<B> {
    /// Maximum number of levels that can be represented with this `FixedDepthStorage2D`. The level depends on the number of bits
    /// that the `B` generic parameter can store
    const MAX_LEVELS : usize = B::BITS / 2;
}

impl<B: FixedStorageType> Storage2D for FixedDepthStorage2D<B> {
    fn fixed_depth() -> Option<usize> {
        Some(Self::MAX_LEVELS)
    }

    fn max_depth() -> Option<usize> {
        Some(Self::MAX_LEVELS)
    }

    fn try_from_quadrants(quadrants: &[Quadrant]) -> Result<Self, crate::Error> {
        if quadrants.len() > Self::MAX_LEVELS {
            return Err(crate::Error::DepthLimitedExceeded {
                max_depth: Self::MAX_LEVELS,
            });
        }
        let mut ret : Self = Default::default();
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
        let bits = self.bits.get_bits(2*level..2*(level+1)).as_u8() as usize;
        Quadrant::try_from(bits).unwrap()
    }

    unsafe fn set_cell_at_level_unchecked(&mut self, level: usize, cell: Quadrant) {
        let cell_index : usize = cell.into();
        self.bits.set_bits(2*level..2*(level+1), B::from_u8(cell_index as u8));
    }
}

impl<'a, B: FixedStorageType> IntoIterator for &'a FixedDepthStorage2D<B> {
    type Item = Quadrant;
    type IntoIter = FixedDepthIter2D<'a, FixedDepthStorage2D<B>>;

    fn into_iter(self) -> Self::IntoIter {
        FixedDepthIter2D {
            index: 0,
            storage: &self
        }
    }
}

pub struct FixedDepthIter2D<'a, S: Storage2D> {
    storage: &'a S,
    index: usize,
}

impl <S: Storage2D> Iterator for FixedDepthIter2D<'_, S>  {
    type Item = Quadrant;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.storage.depth() {
            return None;
        }
        let index = self.index;
        self.index += 1;
        unsafe {
            Some(self.storage.get_cell_at_level_unchecked(index))
        }
    }
}

#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq)]
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

    fn get_cell_at_level_or_none(&self, level: usize) -> Option<Quadrant> {
        if level >= self.depth.into() {
            None
        } else {
            // Is safe because of level check above
            unsafe {
                Some(self.get_cell_at_level_unchecked(level))
            }
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

impl<B: FixedStorageType> Storage2D for StaticStorage2D<B> {
    fn fixed_depth() -> Option<usize> {
        None
    }

    fn max_depth() -> Option<usize> {
        Some(Self::MAX_LEVELS)
    }

    fn try_from_quadrants(quadrants: &[Quadrant]) -> Result<Self, crate::Error> {
        if quadrants.len() > Self::MAX_LEVELS {
            return Err(crate::Error::DepthLimitedExceeded {
                max_depth: Self::MAX_LEVELS,
            });
        }
        let mut ret : Self = Default::default();
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
        let bits = self.bits.get_bits(2*level..2*(level+1)).as_u8() as usize;
        Quadrant::try_from(bits).unwrap()
    }

    unsafe fn set_cell_at_level_unchecked(&mut self, level: usize, cell: Quadrant) {
        let cell_index : usize = cell.into();
        self.bits.set_bits(2*level..2*(level+1), B::from_u8(cell_index as u8));
    }
}

impl<'a, B: FixedStorageType> IntoIterator for &'a StaticStorage2D<B> {
    type Item = Quadrant;
    type IntoIter = FixedDepthIter2D<'a, StaticStorage2D<B>>;

    fn into_iter(self) -> Self::IntoIter {
        FixedDepthIter2D {
            index: 0,
            storage: self,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    /// Returns a bunch of test quadrants 
    fn get_test_quadrants(count: usize) -> Vec<Quadrant> {
        let mut rng = thread_rng();
        (0..count).map(|_| {
            let num : usize = rng.gen_range(0..4);
            Quadrant::try_from(num).unwrap()
        }).collect()
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
                    let idx = $typename::try_from(quadrants.as_slice()).expect("Could not create Morton index from quadrants");
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
                    let idx = $typename::try_from(quadrants.as_slice()).expect("Could not create Morton index from quadrants");
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
                    let idx = $typename::try_from(quadrants.as_slice()).expect("Could not create Morton index from quadrants");
                    let collected_quadrants = idx.cells().collect::<Vec<_>>();
                    assert_eq!(quadrants.as_slice(), collected_quadrants.as_slice());
                }
            }
        }
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
                    let idx = $typename::try_from(quadrants.as_slice()).expect("Could not create Morton index from quadrants");
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
                    let idx = $typename::try_from(quadrants.as_slice()).expect("Could not create Morton index from quadrants");
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
                    let idx = $typename::try_from(quadrants.as_slice()).expect("Could not create Morton index from quadrants");
                    let collected_quadrants = idx.cells().collect::<Vec<_>>();
                    assert_eq!(quadrants.as_slice(), collected_quadrants.as_slice());
                }

                #[test]
                fn child() {
                    let idx = $typename::default();
                    let child_idx = idx.child(Quadrant::Three).expect("child() should not return None");
                    assert_eq!(1, child_idx.depth());
                    assert_eq!(Quadrant::Three, child_idx.get_cell_at_level(0));
                }

                #[test]
                fn child_oob_yields_none() {
                    let quadrants = get_test_quadrants(MAX_LEVELS);
                    let idx = $typename::try_from(quadrants.as_slice()).expect("Could not create Morton index from quadrants");
                    assert_eq!(None, idx.child(Quadrant::Zero));
                }
            }
        }
    }

    test_fixed_depth!(FixedDepthMortonIndex2D8, fixed_depth_morton_index_2d8, 4);
    test_fixed_depth!(FixedDepthMortonIndex2D16, fixed_depth_morton_index_2d16, 8);
    test_fixed_depth!(FixedDepthMortonIndex2D32, fixed_depth_morton_index_2d32, 16);
    test_fixed_depth!(FixedDepthMortonIndex2D64, fixed_depth_morton_index_2d64, 32);
    test_fixed_depth!(FixedDepthMortonIndex2D128, fixed_depth_morton_index_2d128, 64);

    test_static!(StaticMortonIndex2D8, static_morton_index_2d8, 4);
    test_static!(StaticMortonIndex2D16, static_morton_index_2d16, 8);
    test_static!(StaticMortonIndex2D32, static_morton_index_2d32, 16);
    test_static!(StaticMortonIndex2D64, static_morton_index_2d64, 32);
    test_static!(StaticMortonIndex2D128, static_morton_index_2d128, 64);

}