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

// TODO Can we specify trait implementations for specific dimensions? It's probably hard to generalize `from_grid_index`?!

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
