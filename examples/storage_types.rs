use morton_index::{
    dimensions::Quadrant, DynamicMortonIndex2D, DynamicStorage2D, FixedDepthMortonIndex2D8,
    FixedDepthStorage2D, MortonIndex, StaticMortonIndex2D8, Storage, VariableDepthMortonIndex,
};

fn main() -> morton_index::Result<()> {
    // morton_index supports three different storage types for each Morton index:
    // `FixedDepthStorage`, `StaticStorage`, and `DynamicStorage`
    //
    // `FixedDepthStorage` always encodes Morton indices with a fixed depth, so all Morton indices with a FixedDepthStorage
    // have the exact same depth
    // `StaticStorage` is similar to `FixedDepthStorage`, but allows a variable depth up to some statically determined
    // maximum depth
    // Lastly, `DynamicStorage` allows a variable depth with no restrictions to the number of levels in a Morton index
    //
    // Let's look at some examples:

    // Since the `FixedDepthStorage` implies a fixed depth at *compile time*, we have to decide how many levels we want. This
    // effectively means choosing the number of bits in the Morton index. For the `FixedDepthStorage`, we can use any of the
    // unsigned integer types in Rust as storage, and from its number of bits the fixed depth can be deduced:
    assert_eq!(Some(4), FixedDepthStorage2D::<u8>::max_depth());
    assert_eq!(Some(8), FixedDepthStorage2D::<u16>::max_depth());
    // Notice how the number of levels is always half of the number of bits. This is because we are in 2D, where each level takes
    // exactly 2 bits to represent

    // Since we don't always want to type MortonIndex<FixedDepthStorage2D<u8>>, there are a bunch of type definitions for these
    // Morton index types. Let's use `FixedDepthMortonIndex2D8`, which implies a 2D Morton index with fixed depth and 8 bits of
    // storage:
    let fixed1 = FixedDepthMortonIndex2D8::try_from([
        Quadrant::Three,
        Quadrant::Two,
        Quadrant::Zero,
        Quadrant::One,
    ])?;
    assert_eq!(4, fixed1.depth());

    // Let's create another Morton index of the same type:
    let fixed2 = FixedDepthMortonIndex2D8::try_from([
        Quadrant::One,
        Quadrant::Zero,
        Quadrant::Three,
        Quadrant::Two,
    ])?;
    assert_eq!(fixed1.depth(), fixed2.depth());

    // Even if we initialize from less than 4 quadrants, the FixedDepthMortonIndex2D8 always has depth 4:
    let fixed3 = FixedDepthMortonIndex2D8::try_from([Quadrant::One])?;
    assert_eq!(4, fixed3.depth());

    // If we want a variable depth, we can use `StaticStorage`. It works very similar to `FixedDepthStorage`, and there
    // are the same type definitions as for the `FixedDepthStorage`:
    let static1 = StaticMortonIndex2D8::try_from([
        Quadrant::Three,
        Quadrant::Two,
        Quadrant::Zero,
        Quadrant::One,
    ])?;
    assert_eq!(4, static1.depth());

    // With `StaticStorage`, we can actually have a Morton index of the same type but with less than 4 levels!
    let static2 = StaticMortonIndex2D8::try_from([Quadrant::Three])?;
    assert_eq!(1, static2.depth());

    // `StaticStorage` is one of the two `VariableStorage` types in morton-index. As such, it allows getting parent/child nodes from
    // a Morton index with such a `VariableStorage` type:
    let parent_of_static2 = static2.parent().unwrap();
    let root_node = StaticMortonIndex2D8::default();
    assert_eq!(root_node, parent_of_static2);

    let child_of_root = root_node.child(Quadrant::Three).unwrap();
    assert_eq!(static2, child_of_root);

    // The problem with `StaticStorage` is that its maximum depth is still determined at compile time, so if you have too many cells,
    // you can't represent them with a `StaticStorage` type:
    let too_many_levels = StaticMortonIndex2D8::try_from([
        Quadrant::Zero,
        Quadrant::Zero,
        Quadrant::Zero,
        Quadrant::Zero,
        Quadrant::Zero,
    ]);
    assert!(too_many_levels.is_err());

    // To fix this, we can use the last storage type: `DynamicStorage`. It has no maximum depth:
    assert_eq!(None, DynamicStorage2D::max_depth());

    let dynamic = DynamicMortonIndex2D::try_from([
        Quadrant::Zero,
        Quadrant::Zero,
        Quadrant::Zero,
        Quadrant::Zero,
        Quadrant::Zero,
    ])?;
    assert_eq!(5, dynamic.depth());

    Ok(())
}
