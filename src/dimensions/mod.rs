mod dim_2d;
pub use self::dim_2d::*;

mod dim_3d;
pub use self::dim_3d::*;

/// Trait for the dimensionality of a Morton index
pub trait Dimension {
    /// What is the type of each cell entry? This would be a quadrant in the 2D case, or an octant in the 3D case etc.
    type Cell;
    /// Type that represents a grid index for this dimension. For 2D, this will be a Vector2, for 3D a Vector3, and so on.
    type GridIndex;
    /// Type that defines different possible orderings for cells. With this, a mapping between the index of a cell and its
    /// position in N-dimensional space can be made. As an example, take quadrants in 2D space. There are exactly 4 quadrants,
    /// which we can label `Quadrant0` to `Quadrant3`, but it is ambiguous where e.g. `Quadrant1` sits in relation to its
    /// parent node. In 2D, there are two ways of ordering the quadrants, which look like this:
    /// ```ignore
    ///  ___ ___   ___ ___
    /// | 0 | 1 | | 0 | 2 |
    /// |___|___| |___|___|
    /// | 2 | 3 | | 1 | 3 |
    /// |___|___| |___|___|
    ///     XY        YX
    /// ```
    ///
    /// With ordering `XY`, `Quadrant1` would have the index `(1,0)`, with ordering `YZ` it would have index `(0,1)`. This extends
    /// to all higher dimensions, but the number of different orderings increases (in general it is `N!` for N-dimensional space)
    type CellOrdering: Default;
}
