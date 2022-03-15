# `morton-rs` - A Rust library for working with Morton indices

This library enables working with *Morton indices* in your Rust code. Morton indices provide a one-dimensional mapping for the nodes of an n-ary tree (quadtree in 2D, octree in 3D, and so on). Here are the key features of this library in a nutshell:
- Morton indices for 2D and 3D (higher dimensions will be added in the future), providing the typical ordering of Morton indices. Sort a bunch of 2D Morton indices and you have a quadtree, do the same thing in 3D and you have an octree!
- Simple creation of Morton indices from a range of *cells* (`Quadrant` in 2D, or `Octant` in 3D)
- Conversions between Morton indices and indices within an N-dimensional grid. This can be used to easily calculate Morton indices for points within a bounding box
- Converting Morton indices to strings with various naming schemes
- Different storage types to precisely control how many levels a Morton index should have and how much memory it should take. Also define whether you need a Morton index with a dynamic number of levels or a fixed number of levels. Dynamic number of levels is also possible without dynamic allocations using the `StaticStorage` flavor!

## Example

```Rust
// Create a Morton index with 8 bits of fixed-depth storage, so a fixed depth of 4
let mut fixed_index = FixedDepthMortonIndex2D8::try_from([
    Quadrant::Zero,
    Quadrant::One,
    Quadrant::Two,
    Quadrant::Three,
])?;
assert_eq!(4, fixed_index.depth());

// Access cells by their level
assert_eq!(Quadrant::One, fixed_index.get_cell_at_level(1));

// Or get an iterator over all cells, starting at the root of the quadtree
assert_eq!(Some(Quadrant::Three), fixed_index.cells().last());
```

For more examples, check out the [examples folder](./examples/).