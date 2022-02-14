
mod dim_2d;
pub use self::dim_2d::*;

/// Trait for the dimensionality of a Morton index
pub trait Dimension {
    /// What is the type of each cell entry? This would be a quadrant in the 2D case, or an octant in the 3D case etc.
    type Cell;
    /// Type that represents a grid index for this dimension. For 2D, this will be a Vector2, for 3D a Vector3, and so on.
    type GridIndex;
}