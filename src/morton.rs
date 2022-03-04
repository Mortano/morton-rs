use std::fmt::Debug;
use std::hash::Hash;

/// When converting a Morton index to a string, these are the different ways to construct the string
pub enum MortonIndexNaming {
    /// Construct a string by concatenating the cells of the N-ary tree at each level. Each cell can be represented by
    /// a number in `[0;(2^N)-1]`, resulting in a string such as '2630'. This represents a 4-level Morton index with cell
    /// '2' below the root, cell '6' below cell '2', cell '3' below cell '6', and finally cell '0' below cell '3'.
    CellConcatenation,
    /// Like `CellConcatenation`, but prefixes all strings with `r`, so that an empty Morton index yields `r` instead of
    /// the empty string
    CellConcatenationWithRoot,
    /// Converts the Morton index to an index within an N-dimensional grid together with the depth of the Morton index.
    /// For a 3D Morton index with depth 4, this might yield the string `4-15-7-3`, which can be read as `depth: 4`,
    /// `x-index: 15`, `y-index: 7`, `z-index: 3`. Each index is a number in `[0;(2^N)-1]`. This uses the default `CellOrdering`
    /// to compute the grid index, as given by the `Dimension` of the respective `MortonIndex`.
    GridIndex,
}

/// Trait for any type of Morton index, independent of the maximum levels or dimensionality. This contains a core set of common
/// functions that are identical to all Morton indices:
/// - Getting/setting cell values at specific levels
/// - Querying the depth of the Morton index
/// - Conversion to a `String`
/// - Conversion to an index within an N-dimensional grid
pub trait MortonIndex: PartialOrd + Ord + PartialEq + Eq + Debug + Hash {
    type Dimension: crate::dimensions::Dimension;

    /// Returns the cell at the given `level`.
    ///
    /// This performs a depth check, if you want to omit it consider using the `get_cell_at_level_unchecked` variant.
    ///
    /// # Panics
    ///
    /// If `level` is `>= self.depth()`
    fn get_cell_at_level(
        &self,
        level: usize,
    ) -> <Self::Dimension as crate::dimensions::Dimension>::Cell;
    /// Returns the cell at the given `level`. Performs no depth checks.
    ///
    /// # Safety
    ///
    /// Calling this function with a value for `level` that is `>= self.depth()` is UB
    unsafe fn get_cell_at_level_unchecked(
        &self,
        level: usize,
    ) -> <Self::Dimension as crate::dimensions::Dimension>::Cell;
    /// Sets the cell at the given `level` to `cell`. Note that some Morton indices are able to store a variable number of levels,
    /// so these implementations will provide methods to append cells to increase the number of levels.
    ///
    /// This performs a depth check, if you want to omit it consider using the `set_cell_at_level_unchecked` variant.
    ///
    /// # Panics
    ///
    /// If `level` is `>= self.depth()`
    fn set_cell_at_level(
        &mut self,
        level: usize,
        cell: <Self::Dimension as crate::dimensions::Dimension>::Cell,
    );
    /// Sets the cell at the given `level` to `cell`. Performs no depth checks.
    ///
    /// # Safety
    ///
    /// Calling this function with a value for `level` that is `>= self.depth()` is UB
    unsafe fn set_cell_at_level_unchecked(
        &mut self,
        level: usize,
        cell: <Self::Dimension as crate::dimensions::Dimension>::Cell,
    );
    /// Returns the current depth of this `MortonIndex`. A depth of `0` is equal to an empty Morton index. By definition, such an index
    /// always represents the root node of an N-ary tree
    fn depth(&self) -> usize;
    /// Converts this `MortonIndex` into a `String` representation, using the given `naming` convention
    fn to_string(&self, naming: MortonIndexNaming) -> String;
    /// Converts this `MortonIndex` into a N-dimensional grid index, using the given `ordering`. This index describes the position
    /// of the cell that the `MortonIndex` corresponds to in an N-dimensional grid. There is an intimate relationship between
    /// N-ary trees and N-dimensional grids when it comes to Morton indices, which is illustrated in the following image for the
    /// 2D case:
    /// ```text
    ///    ₀  ₁  ₂  ₃  ₄  ₅  ₆  ₇
    ///   ┌──┬──┬──┬──┬──┬──┬──┬──┐   ┌───────────┬─────┬─────┐
    /// ⁰ ├──┼──┼──┼──┼──┼──┼──┼──┤   │           │     │     │
    /// ¹ ├──┼──┼──┼──┼──┼──┼──┼──┤   │           ├──┬──┼─────┤
    /// ² ├──┼──┼──┼──┼──┼──┼──┼──┤   │           ├──┼──┤     │
    /// ³ ├──┼──┼──┼──┼──┼──┼──┼──┤   ├───────────┼──┴──┴─────┤
    /// ⁴ ├──┼──┼──┼──┼──┼──┼──┼──┤   │           │           │
    /// ⁵ ├──┼──┼──┼──┼──┼──┼──┼──┤   │           │           │
    /// ⁶ ├──┼──┼──┼──┼──┼──┼──┼──┤   │           │           │
    /// ⁷ └──┴──┴──┴──┴──┴──┴──┴──┘   └───────────┴───────────┘
    /// ```
    ///
    /// Each N-ary tree is a 'less dense' version of the corresponding N-dimensional grid. As such, each node in an N-ary tree
    /// corresponds to a cell in an N-dimensional grid with a resolution of `2 ^ D`, where `D` is the depth of the node in the
    /// N-ary tree.
    fn to_grid_index(
        &self,
        ordering: <Self::Dimension as crate::dimensions::Dimension>::CellOrdering,
    ) -> <Self::Dimension as crate::dimensions::Dimension>::GridIndex;
}
