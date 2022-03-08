mod morton_index_2d;
pub use self::morton_index_2d::*;

mod morton_index_3d;
pub use self::morton_index_3d::*;

mod storage;
pub use self::storage::*;

use crate::dimensions::Dimension;
use crate::number::Bits;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;

// Unify the storage traits (FixedDepth, Static, Dynamic) for any dimension here
// Unify cell iterator for any dimensions (should work by just calling the appropriate functions on the storage)
// TODO Can we specify trait implementations for specific dimensions? It's probably hard to generalize `from_grid_index`?!

/// Trait for the different storage types of a Morton index. A `Storage` is responsible for storing the different cells at
/// each level of the Morton index. While it would be possible to simply have a `Vec<Cell>`, it is usually more efficient
/// to do some bit-packing. Implementors of this trait can store the cells in any way they choose, `Storage` just defines
/// the common API that each Morton index must support. The actual storage depends on the `Dimension` of the Morton index,
/// so `Storage` requires a generic parameter that implements `Dimension`.
pub trait Storage<D: Dimension>:
    Default + PartialOrd + Ord + PartialEq + Eq + Debug + Hash
{
    /// Get the maximum depth that this storage type can represent. If there is no maximum depth, `None` is returned
    fn max_depth() -> Option<usize>;
    /// The current depth of the index stored within this storage type
    fn depth(&self) -> usize;
    /// Returns the value of the cell at `level` within this storage type
    ///
    /// # Safety
    ///
    /// This operation performs no depth checks and assumes that `level < self.depth()`. Violating this contract is UB
    unsafe fn get_cell_at_level_unchecked(&self, level: usize) -> D::Cell;
    /// Set the value of the cell at `level` within this storage type to the given `cell`
    ///
    /// # Safety
    ///
    /// This operation performs no depth checks and assumes that `level < self.depth()`. Violating this contract is UB
    unsafe fn set_cell_at_level_unchecked(&mut self, level: usize, cell: D::Cell);
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

/// Iterator over cells of a Morton index
pub struct CellIter<'a, D: Dimension, S: Storage<D>> {
    storage: &'a S,
    index: usize,
    _phantom: PhantomData<D>,
}

impl<'a, D: Dimension, S: Storage<D>> Iterator for CellIter<'a, D, S> {
    type Item = D::Cell;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.storage.depth() {
            return None;
        }
        let index = self.index;
        self.index += 1;
        unsafe { Some(self.storage.get_cell_at_level_unchecked(index)) }
    }
}

/// Trait for any type that can be used as the data type in a fixed-depth storage of a Morton index. The idea here is to generalize
/// unsigned integer types with an arbitrary number of bits, so that we can use `u8`, `u16`, `u32` etc. as the internal data type
/// of the `FixedDepthStorageND` types. Ultimately, this should also allow us to use `[u8; X]` types to have statically-sized Morton
/// indices with arbitrary depth, beyond the native integer types of the current plattform, but this is a TODO.
pub trait FixedStorageType:
    Bits + Copy + Clone + PartialEq + Eq + PartialOrd + Ord + Default + Debug + Hash
{
}

impl<B: Bits + Copy + Clone + PartialEq + Eq + PartialOrd + Ord + Default + Debug + Hash>
    FixedStorageType for B
{
}
