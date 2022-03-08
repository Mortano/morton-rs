// TODO This whole file will be VERY similar to `morton_index_2d.rs`. Is there some way to combine the two?
// The biggest hurdle I see is that many of the types are different, and the implementations will be different
// for some of the functions that have to care about dimensionality...

use crate::{
    dimensions::{Dim3D, Octant},
    Storage, VariableDepthStorage,
};

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

impl<S: VariableDepthStorage<Dim3D>> MortonIndex3D<S> {
    /// Returns a Morton index for the child `octant` of this Morton index. If this index is already at
    /// the maximum depth, `None` is returned instead
    pub fn child(&self, octant: Octant) -> Option<Self> {
        self.storage.child(octant).map(|storage| Self { storage })
    }

    /// Returns a Morton index for the parent node of this Morton index. If this index represents the root
    /// node, `None` is returned instead
    pub fn parent(&self) -> Option<Self> {
        self.storage.parent().map(|storage| Self { storage })
    }
}
