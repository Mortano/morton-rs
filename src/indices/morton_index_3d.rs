// TODO This whole file will be VERY similar to `morton_index_2d.rs`. Is there some way to combine the two?
// The biggest hurdle I see is that many of the types are different, and the implementations will be different
// for some of the functions that have to care about dimensionality...

use crate::dimensions::Octant;
use std::fmt::Debug;
use std::hash::Hash;

/// Trait for any storage type of a 3D Morton index
pub trait Storage3D: Default + PartialOrd + Ord + PartialEq + Eq + Debug + Hash {
    /// Get the maximum depth that this storage type can represent. If there is no maximum depth, `None` is returned
    fn max_depth() -> Option<usize>;
    /// Try to create an instance of this storage type from the given slice of `Octant`s. This operation may fail
    /// if the number of octants exceeds the maximum depth of this storage type, as given by [max_depth](Self::max_depth)
    fn try_from_octants(octants: &[Octant]) -> Result<Self, crate::Error>;
    /// The current depth of the index stored within this storage type
    fn depth(&self) -> usize;
    /// Returns the value of the cell at `level` within this storage type
    ///
    /// # Safety
    ///
    /// This operation performs no depth checks and assumes that `level < self.depth()`. Violating this contract is UB
    unsafe fn get_cell_at_level_unchecked(&self, level: usize) -> Octant;
    /// Set the value of the cell at `level` within this storage type to the given `Octant`
    ///
    /// # Safety
    ///
    /// This operation performs no depth checks and assumes that `level < self.depth()`. Violating this contract is UB
    unsafe fn set_cell_at_level_unchecked(&mut self, level: usize, cell: Octant);
}

/// Trait for any storage type of a 3D Morton index that supports variable depth
pub trait VariableDepthStorage3D: Storage3D {
    fn max_depth() -> Option<usize> {
        None
    }

    /// Returns a storage representing the parent index of the index stored in this storage. If this storage stores the
    /// root node (i.e. it is empty), `None` is returned instead
    fn parent(&self) -> Option<Self>;
    /// Returns a storage representing the child at the given `octant` for the index stored in this storage. If this
    /// storage is already at its maximum depth, `None` is returned instead
    fn child(&self, octant: Octant) -> Option<Self>;
}
