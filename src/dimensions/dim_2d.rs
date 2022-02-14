use super::Dimension;
use nalgebra::Vector2;

pub struct Dim2D;

impl Dimension for Dim2D {
    type Cell = Quadrant;
    //TODO `usize` might be too small to store the grid index for a large Morton index. It would be better to only state
    //that the GridIndex is Vector2, and leave the <T> part up to the actual implementation of the Morton index, but this
    //requires generic associated types, which are unstable :(
    type GridIndex = Vector2<usize>; 
}

/// All quadrants of a quadtree
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Quadrant {
    Zero,
    One,
    Two,
    Three,
}

impl Into<usize> for Quadrant {
    fn into(self) -> usize {
        match self {
            Quadrant::Zero => 0,
            Quadrant::One => 1,
            Quadrant::Two => 2,
            Quadrant::Three => 3,
        }
    }
}

impl TryFrom<usize> for Quadrant {
    type Error = crate::Error;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Quadrant::Zero),
            1 => Ok(Quadrant::One),
            2 => Ok(Quadrant::Two),
            3 => Ok(Quadrant::Three),
            _ => Err(crate::Error::CellIndexOutOfRange),
        }
    }
}