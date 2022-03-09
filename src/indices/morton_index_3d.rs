// TODO This whole file will be VERY similar to `morton_index_2d.rs`. Is there some way to combine the two?
// The biggest hurdle I see is that many of the types are different, and the implementations will be different
// for some of the functions that have to care about dimensionality...

use nalgebra::Vector3;

use crate::{
    dimensions::{Dim3D, Octant, OctantOrdering},
    FixedDepthStorage, FixedStorageType, MortonIndex, MortonIndexNaming, Storage, StorageType,
    VariableDepthStorage,
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

impl<'a, S: Storage<Dim3D> + TryFrom<&'a [Octant], Error = crate::Error>> TryFrom<&'a [Octant]>
    for MortonIndex3D<S>
{
    type Error = crate::Error;

    fn try_from(value: &'a [Octant]) -> Result<Self, Self::Error> {
        S::try_from(value).map(|storage| Self { storage })
    }
}

impl<S: Storage<Dim3D>> MortonIndex for MortonIndex3D<S> {
    type Dimension = Dim3D;

    fn get_cell_at_level(&self, level: usize) -> Octant {
        if level >= self.depth() {
            panic!("level must not be >= self.depth()");
        }
        unsafe { self.storage.get_cell_at_level_unchecked(level) }
    }

    unsafe fn get_cell_at_level_unchecked(&self, level: usize) -> Octant {
        self.storage.get_cell_at_level_unchecked(level)
    }

    fn set_cell_at_level(&mut self, level: usize, cell: Octant) {
        if level >= self.depth() {
            panic!("level must not be >= self.depth()");
        }
        unsafe { self.storage.set_cell_at_level_unchecked(level, cell) }
    }

    unsafe fn set_cell_at_level_unchecked(&mut self, level: usize, cell: Octant) {
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
                let grid_index = self.to_grid_index(OctantOrdering::default());
                format!(
                    "{}-{}-{}-{}",
                    self.depth(),
                    grid_index.x,
                    grid_index.y,
                    grid_index.z
                )
            }
        }
    }

    fn to_grid_index(&self, ordering: OctantOrdering) -> Vector3<usize> {
        let mut index = Vector3::new(0, 0, 0);
        for level in 0..self.depth() {
            let octant = unsafe { self.get_cell_at_level_unchecked(level) };
            let octant_index = ordering.to_index(octant);
            let shift = self.depth() - level - 1;
            index.x |= octant_index.x << shift;
            index.y |= octant_index.y << shift;
            index.z |= octant_index.z << shift;
        }
        index
    }
}

// Storage types:

/// Storage for a 3D Morton index that always stores a Morton index with a fixed depth (fixed number of levels)
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FixedDepthStorage3D<B: FixedStorageType> {
    bits: B,
}

impl<B: FixedStorageType> Storage<Dim3D> for FixedDepthStorage3D<B> {
    type StorageType = FixedDepthStorage<Dim3D, B>;
    type Bits = B;

    fn bits(&self) -> &Self::Bits {
        &self.bits
    }

    fn bits_mut(&mut self) -> &mut Self::Bits {
        &mut self.bits
    }

    fn depth(&self) -> usize {
        FixedDepthStorage::<Dim3D, B>::MAX_LEVELS
    }
}
