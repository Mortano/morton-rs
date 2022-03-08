use crate::{dimensions};
use crate::{dimensions::Dimension, FixedStorageType, Storage};
use std::fmt::Debug;
use std::hash::Hash;

/// Trait for all storage types that represent a fixed-depth Morton index
pub trait FixedDepthStorage: Default + PartialOrd + Ord + PartialEq + Eq + Debug + Hash {
    type Dimension: dimensions::Dimension;
    /// The primitive type used for the storage of the cells. See the comment in `FixedStorageType` for an explanation
    type BitType: FixedStorageType;
    /// The maximum number of levels that this storage can represent
    const MAX_LEVELS: usize;
    /// The dimensionality of the associated `Dimension`
    const DIMENSIONALITY: usize;

    /// Immutable access to the `BitType`
    fn bits(&self) -> &Self::BitType;
    /// Mutable access to the `BitType`
    fn bits_mut(&mut self) -> &mut Self::BitType;
}

impl<D, B, S> Storage<D> for S
where
    D: Dimension,
    B: FixedStorageType,
    S: FixedDepthStorage<Dimension = D, BitType = B>,
{
    fn max_depth() -> Option<usize> {
        Some(S::MAX_LEVELS)
    }

    fn depth(&self) -> usize {
        S::MAX_LEVELS
    }

    unsafe fn get_cell_at_level_unchecked(&self, level: usize) -> <D as Dimension>::Cell {
        let start_bit = B::BITS - ((level + 1) * S::DIMENSIONALITY);
        let end_bit = start_bit + S::DIMENSIONALITY;
        // TODO This only works up to dimensionality 8. Maybe raise a compile error for larger dimensions? Or do some
        // conditional implementation only as long as the DIMENSIONALITY of S is <= 8. But how would that work?
        let bits = self.bits().get_bits(start_bit..end_bit).as_u8() as usize;
        <D as Dimension>::Cell::try_from(bits).unwrap()
    }

    unsafe fn set_cell_at_level_unchecked(&mut self, level: usize, cell: <D as Dimension>::Cell) {
        let cell_index: usize = cell.into();
        let start_bit = B::BITS - ((level + 1) * S::DIMENSIONALITY);
        let end_bit = start_bit + S::DIMENSIONALITY;
        self.bits_mut()
            .set_bits(start_bit..end_bit, B::from_u8(cell_index as u8));
    }
}

// Can't implement TryFrom generically, because there is a blanket implementation impl<T, U> TryFrom<U> for T where U: Into<T>
// impl<'a, D, S> TryFrom<&'a [D::Cell]> for S
// where
//     D: Dimension,
//     S: FixedDepthStorage<D>,
// {
// }

// Can't implement IntoIterator generically, because there also is a blanket implementation...
// impl<'a, D, S> IntoIterator for &'a S
// where
//     D: Dimension,
//     S: FixedDepthStorage<Dimension = D>,
// {
//     type Item = D::Cell;

//     type IntoIter = CellIter<'a, D, S>;

//     fn into_iter(self) -> Self::IntoIter {
//         todo!()
//     }
// }
