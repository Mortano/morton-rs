use std::fmt::Display;

use crate::{MortonIndex, MortonIndexNaming};

/// Helper struct that wraps around a slice of any type that implements `MortonIndex` and makes this slice displayable in a nice
/// way using any of the `MortonIndexNaming` options. This is useful is you have a slice of Morton indices that you quickly want 
/// to display, e.g. as `['r1', 'r234', 'r4456']` instead of the default `Debug` representation (which will print the internal bit
/// representation for many of the Morton index types in this library)
pub struct DisplayableMortonIndexSlice<'a, T: MortonIndex> {
    morton_indices: &'a [T],
    naming: MortonIndexNaming,
}

impl<'a, T: MortonIndex> std::fmt::Debug for DisplayableMortonIndexSlice<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl<'a, T: MortonIndex> Display for DisplayableMortonIndexSlice<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (idx, morton) in self.morton_indices.iter().enumerate() {
            write!(f, "{}", morton.to_string(self.naming))?;
            if idx < self.morton_indices.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

impl<'a, T: MortonIndex> PartialEq for DisplayableMortonIndexSlice<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.morton_indices == other.morton_indices
    }
}

impl<'a, T: MortonIndex> Eq for DisplayableMortonIndexSlice<'a, T> {
}

impl<'a, T: MortonIndex> PartialOrd for DisplayableMortonIndexSlice<'a, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.morton_indices.partial_cmp(&other.morton_indices)
    }
}

impl<'a, T: MortonIndex> Ord for DisplayableMortonIndexSlice<'a, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.morton_indices.cmp(&other.morton_indices)
    }
}


/// Wraps a slice of morton indices into a type that implements `Display` in a nice way using the given `naming`. Allows one to use a
/// slice of morton indices in `print!` or `assert!` statements and get a human-readable representation
pub fn make_displayable<'a, T: MortonIndex>(morton_indices: &'a [T], naming: MortonIndexNaming) -> DisplayableMortonIndexSlice<'a, T> {
    DisplayableMortonIndexSlice { morton_indices, naming, }
}

