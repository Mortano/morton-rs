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

impl<'a, S: Storage2D + 'a> MortonIndex2D<S> where &'a S : IntoIterator<Item = Quadrant> {
    pub fn cells(&'a self) -> <&'a S as IntoIterator>::IntoIter {
        self.storage.into_iter()
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

pub trait FixedStorageType :  Bits + Copy + Clone + PartialEq + Eq + PartialOrd + Ord + Default + Debug + Hash {}

impl <B: Bits + Copy + Clone + PartialEq + Eq + PartialOrd + Ord + Default + Debug + Hash> FixedStorageType for B {}

/// Storage for a 2D Morton index that always stores a Morton index with a fixed depth (fixed number of levels)
#[derive(Default, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    type IntoIter = FixedDepthIter2D<'a, B>;

    fn into_iter(self) -> Self::IntoIter {
        FixedDepthIter2D {
            index: 0,
            storage: &self
        }
    }
}

pub struct FixedDepthIter2D<'a, B: FixedStorageType> {
    storage: &'a FixedDepthStorage2D<B>,
    index: usize,
}

impl <B: FixedStorageType> Iterator for FixedDepthIter2D<'_, B>  {
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

    test_fixed_depth!(FixedDepthMortonIndex2D8, fixed_depth_morton_index_2d8, 4);
    test_fixed_depth!(FixedDepthMortonIndex2D16, fixed_depth_morton_index_2d16, 8);
    test_fixed_depth!(FixedDepthMortonIndex2D32, fixed_depth_morton_index_2d32, 16);
    test_fixed_depth!(FixedDepthMortonIndex2D64, fixed_depth_morton_index_2d64, 32);
    test_fixed_depth!(FixedDepthMortonIndex2D128, fixed_depth_morton_index_2d128, 64);

}