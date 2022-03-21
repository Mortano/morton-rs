use morton_index::{
    dimensions::{Octant, OctantOrdering},
    FixedDepthMortonIndex3D16, MortonIndex, MortonIndexNaming,
};
use nalgebra::Vector3;

fn main() -> morton_index::Result<()> {
    // This example continues the `morton_2d.rs` example
    // First, we must decide on a type of Morton index to use. Since we are in 3D, we need one more bit per dimension,
    // so let's use a 16-bit type for a total of ⌊(16/3)⌋=5 dimensions:
    let mut fixed_index = FixedDepthMortonIndex3D16::try_from([
        // We are in 3D, so the cells of a Morton index in 3D are octants, not quadrants!
        Octant::Zero,
        Octant::One,
        Octant::Two,
        Octant::Three,
        Octant::Four,
    ])?;
    assert_eq!(5, fixed_index.depth());

    // You can access the cells by their level (where level 0 is the octant below the root node of an octree)
    assert_eq!(Octant::One, fixed_index.get_cell_at_level(1));

    // Or you can get an iterator over all cells, starting from the root node down into the octree
    assert_eq!(Some(Octant::Four), fixed_index.cells().last());

    // You can also set the cells at each level
    fixed_index.set_cell_at_level(0, Octant::Two);
    assert_eq!(Octant::Two, fixed_index.get_cell_at_level(0));

    // Morton indices can be converted to either strings or indices within a regular grid
    // For strings, there are several different naming schemes:
    assert_eq!(
        "21234".to_owned(),
        fixed_index.to_string(MortonIndexNaming::CellConcatenation)
    );
    assert_eq!(
        "r21234".to_owned(),
        fixed_index.to_string(MortonIndexNaming::CellConcatenationWithRoot)
    );
    assert_eq!(
        "5-10-22-1".to_owned(),
        fixed_index.to_string(MortonIndexNaming::GridIndex)
    );

    // The `GridIndex` naming shows the depth and a 3D index within a regular grid. You can get this grid index
    // as a Vector3<usize>. You have to define an ordering of the octant, because the grid index depends on
    // which octant maps to which (x,y,z) coordinate. The `OctantOrdering` type has more information
    assert_eq!(
        Vector3::new(10_usize, 22_usize, 1_usize),
        fixed_index.to_grid_index(OctantOrdering::XYZ)
    );

    // You can also go the other way around: From a grid index to a Morton index. This will be most useful if you
    // want to calculate Morton indices from points in N-d space, where you would first calculate the grid index
    // of each point and then convert it into a Morton index
    assert_eq!(
        fixed_index,
        FixedDepthMortonIndex3D16::from_grid_index(Vector3::new(10, 22, 1), OctantOrdering::XYZ),
    );

    Ok(())
}
