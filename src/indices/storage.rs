use crate::dimensions;
use crate::number::Bits;
use crate::{dimensions::Dimension, FixedStorageType};
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;

pub trait StorageType {
    type Dimension: dimensions::Dimension;
    const MAX_LEVELS: usize;
    const DIMENSIONALITY: usize;

    fn max_depth() -> Option<usize>;
    unsafe fn get_cell_at_level_unchecked<T: Bits>(
        bits: &T,
        level: usize,
    ) -> <Self::Dimension as dimensions::Dimension>::Cell;
    unsafe fn set_cell_at_level_unchecked<T: Bits>(
        bits: &mut T,
        level: usize,
        cell: <Self::Dimension as dimensions::Dimension>::Cell,
    );
}

/// Trait for the different storage types of a Morton index. A `Storage` is responsible for storing the different cells at
/// each level of the Morton index. While it would be possible to simply have a `Vec<Cell>`, it is usually more efficient
/// to do some bit-packing. Implementors of this trait can store the cells in any way they choose, `Storage` just defines
/// the common API that each Morton index must support. The actual storage depends on the `Dimension` of the Morton index,
/// so `Storage` requires a generic parameter that implements `Dimension`.
pub trait Storage<D: Dimension>:
    Default + PartialOrd + Ord + PartialEq + Eq + Debug + Hash
{
    type StorageType: StorageType<Dimension = D>;
    type Bits: Bits;

    fn bits(&self) -> &Self::Bits;
    fn bits_mut(&mut self) -> &mut Self::Bits;
    /// The current depth of the index stored within this storage type
    fn depth(&self) -> usize;

    /// Get the maximum depth that this storage type can represent. If there is no maximum depth, `None` is returned
    fn max_depth() -> Option<usize> {
        Self::StorageType::max_depth()
    }
    /// Returns the value of the cell at `level` within this storage type
    ///
    /// # Safety
    ///
    /// This operation performs no depth checks and assumes that `level < self.depth()`. Violating this contract is UB
    unsafe fn get_cell_at_level_unchecked(&self, level: usize) -> D::Cell {
        // How do we get access to the actual data? We implement this trait for some T, and this T must provide access
        // to the bits I think?
        Self::StorageType::get_cell_at_level_unchecked(self.bits(), level)
    }
    /// Set the value of the cell at `level` within this storage type to the given `cell`
    ///
    /// # Safety
    ///
    /// This operation performs no depth checks and assumes that `level < self.depth()`. Violating this contract is UB
    unsafe fn set_cell_at_level_unchecked(&mut self, level: usize, cell: D::Cell) {
        Self::StorageType::set_cell_at_level_unchecked(self.bits_mut(), level, cell)
    }
}

/// Trait for any storage type of a Morton index that supports variable depth
pub trait VariableDepthStorage<D: Dimension>: Storage<D> {
    /// Returns a storage representing the parent index of the index stored in this storage. If this storage stores the
    /// root node (i.e. it is empty), `None` is returned instead
    fn parent(&self) -> Option<Self>;
    /// Returns a storage representing the child at the given `cell` for the index stored in this storage. If this
    /// storage is already at its maximum depth, `None` is returned instead
    fn child(&self, cell: D::Cell) -> Option<Self>;
}

pub struct FixedDepthStorage<D: Dimension, B: FixedStorageType> {
    _phantom: PhantomData<(D, B)>,
}

impl<D: Dimension, B: FixedStorageType> StorageType for FixedDepthStorage<D, B> {
    type Dimension = D;

    const MAX_LEVELS: usize = B::BITS / D::DIMENSIONALITY;

    const DIMENSIONALITY: usize = D::DIMENSIONALITY;

    fn max_depth() -> Option<usize> {
        Some(Self::MAX_LEVELS)
    }

    unsafe fn get_cell_at_level_unchecked<T: Bits>(
        bits: &T,
        level: usize,
    ) -> <Self::Dimension as dimensions::Dimension>::Cell {
        let start_bit = B::BITS - ((level + 1) * Self::DIMENSIONALITY);
        let end_bit = start_bit + Self::DIMENSIONALITY;
        // TODO This only works up to dimensionality 8. Maybe raise a compile error for larger dimensions? Or do some
        // conditional implementation only as long as the DIMENSIONALITY of S is <= 8. But how would that work?
        let bits = bits.get_bits(start_bit..end_bit).as_u8() as usize;
        <D as Dimension>::Cell::try_from(bits).unwrap()
    }

    unsafe fn set_cell_at_level_unchecked<T: Bits>(
        bits: &mut T,
        level: usize,
        cell: <Self::Dimension as dimensions::Dimension>::Cell,
    ) {
        let cell_index: usize = cell.into();
        let start_bit = B::BITS - ((level + 1) * Self::DIMENSIONALITY);
        let end_bit = start_bit + Self::DIMENSIONALITY;
        bits.set_bits(start_bit..end_bit, T::from_u8(cell_index as u8));
    }
}

pub struct StaticStorage<D: Dimension, B: FixedStorageType> {
    _phantom: PhantomData<(D, B)>,
}

impl<D: Dimension, B: FixedStorageType> StorageType for StaticStorage<D, B> {
    type Dimension = D;

    const MAX_LEVELS: usize = B::BITS / D::DIMENSIONALITY;

    const DIMENSIONALITY: usize = D::DIMENSIONALITY;

    fn max_depth() -> Option<usize> {
        Some(Self::MAX_LEVELS)
    }

    unsafe fn get_cell_at_level_unchecked<T: Bits>(
        bits: &T,
        level: usize,
    ) -> <Self::Dimension as dimensions::Dimension>::Cell {
        let start_bit = B::BITS - ((level + 1) * Self::DIMENSIONALITY);
        let end_bit = start_bit + Self::DIMENSIONALITY;
        let bits = bits.get_bits(start_bit..end_bit).as_u8() as usize;
        <D as Dimension>::Cell::try_from(bits).unwrap()
    }

    unsafe fn set_cell_at_level_unchecked<T: Bits>(
        bits: &mut T,
        level: usize,
        cell: <Self::Dimension as dimensions::Dimension>::Cell,
    ) {
        let cell_index: usize = cell.into();
        let start_bit = B::BITS - ((level + 1) * Self::DIMENSIONALITY);
        let end_bit = start_bit + Self::DIMENSIONALITY;
        bits.set_bits(start_bit..end_bit, T::from_u8(cell_index as u8));
    }
}

pub struct DynamicStorage<D: Dimension> {
    _phantom: PhantomData<D>,
}

impl<D: Dimension> StorageType for DynamicStorage<D> {
    type Dimension = D;

    const MAX_LEVELS: usize = 0; // TODO This should be Option<usize>

    const DIMENSIONALITY: usize = D::DIMENSIONALITY;

    fn max_depth() -> Option<usize> {
        None
    }

    unsafe fn get_cell_at_level_unchecked<T: Bits>(
        bits: &T,
        level: usize,
    ) -> <Self::Dimension as dimensions::Dimension>::Cell {
        // We store the data in LittleEndian
        let num_bytes = bits.size();
        let start_bit = (num_bytes * 8) - (level + 1) * Self::DIMENSIONALITY;
        let end_bit = start_bit + Self::DIMENSIONALITY;
        let cell_index = bits.get_bits_as_usize(start_bit..end_bit);
        cell_index.try_into().unwrap()
    }

    unsafe fn set_cell_at_level_unchecked<T: Bits>(
        bits: &mut T,
        level: usize,
        cell: <Self::Dimension as dimensions::Dimension>::Cell,
    ) {
        let cell_index: usize = cell.into();
        let num_bytes = bits.size();
        let start_bit = (num_bytes * 8) - (level + 1) * Self::DIMENSIONALITY;
        let end_bit = start_bit + Self::DIMENSIONALITY;
        bits.set_bits_from_usize(start_bit..end_bit, cell_index);
    }
}
