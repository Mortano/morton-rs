use morton_rs::{
    dimensions::{Quadrant, QuadrantOrdering},
    FixedDepthMortonIndex2D8, MortonIndex, MortonIndexNaming,
};
use nalgebra::Vector2;

fn main() -> morton_rs::Result<()> {
    // You have a choice of three different storage types
    // The simplest one is the `FixedDepth` storage type. Morton indices with this storage always have the
    // same depth. As an example, here is a 2D Morton index which uses 8 bits internal storage, so always
    // represents a depth of 4
    let mut fixed_index = FixedDepthMortonIndex2D8::try_from(
        [
            Quadrant::Zero,
            Quadrant::One,
            Quadrant::Two,
            Quadrant::Three,
        ]
        .as_slice(),
    )?;
    assert_eq!(4, fixed_index.depth());

    // You can access the cells by their level (where level 0 is the quadrant below the root node of a quadtree)
    assert_eq!(Quadrant::One, fixed_index.get_cell_at_level(1));

    // Or you can get an iterator over all cells, starting from the root node down into the quadtree
    assert_eq!(Some(Quadrant::Three), fixed_index.cells().last());

    // You can also set the cells at each level
    fixed_index.set_cell_at_level(0, Quadrant::Two);
    assert_eq!(Quadrant::Two, fixed_index.get_cell_at_level(0));

    // Morton indices can be converted to either strings or indices within a regular grid
    // For strings, there are several different naming schemes:
    assert_eq!(
        "2123".to_owned(),
        fixed_index.to_string(MortonIndexNaming::CellConcatenation)
    );
    assert_eq!(
        "r2123".to_owned(),
        fixed_index.to_string(MortonIndexNaming::CellConcatenationWithRoot)
    );
    assert_eq!(
        "4-5-11".to_owned(),
        fixed_index.to_string(MortonIndexNaming::GridIndex)
    );

    // The `GridIndex` naming shows the depth and a 2D index within a regular grid. You can get this grid index
    // as a Vector2<usize>. You have to define an ordering of the quadrants, because the grid index depends on
    // which quadrant maps to which (x,y) coordinate. The `QuadrantOrdering` type has more information
    assert_eq!(
        Vector2::new(5_usize, 11_usize),
        fixed_index.to_grid_index(QuadrantOrdering::XY)
    );

    // You can also go the other way around: From a grid index to a Morton index. This will be most useful if you
    // want to calculate Morton indices from points in N-d space, where you would first calculate the grid index
    // of each point and then convert it into a Morton index
    assert_eq!(
        fixed_index,
        FixedDepthMortonIndex2D8::from_grid_index(Vector2::new(5, 11), QuadrantOrdering::XY),
    );

    Ok(())
}
