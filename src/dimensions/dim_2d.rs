use super::Dimension;
use nalgebra::Vector2;

/// A `Dimension` that represents 2D space
pub struct Dim2D;

impl Dimension for Dim2D {
    type Cell = Quadrant;
    //TODO `usize` might be too small to store the grid index for a large Morton index. It would be better to only state
    //that the GridIndex is Vector2, and leave the <T> part up to the actual implementation of the Morton index, but this
    //requires generic associated types, which are unstable :(
    type GridIndex = Vector2<usize>;
    type CellOrdering = QuadrantOrdering;

    const DIMENSIONALITY: usize = 2;
}

/// Ordering of quadrants in 2D space
#[derive(Debug)]
pub enum QuadrantOrdering {
    /// 'X-major' ordering which maps quadrant 1 to index `(X=1,Y=0)`. This is the default `QuadrantOrdering`!
    XY,
    /// 'Y-major' ordering which maps quadrant 1 to index `(X=0,Y=1)`
    YX,
}

impl QuadrantOrdering {
    pub fn to_index(&self, quadrant: Quadrant) -> Vector2<usize> {
        let quadrant_index = quadrant.index();
        match self {
            QuadrantOrdering::XY => Vector2::new(quadrant_index & 1, (quadrant_index >> 1) & 1),
            QuadrantOrdering::YX => Vector2::new((quadrant_index >> 1) & 1, quadrant_index & 1),
        }
    }
}

impl Default for QuadrantOrdering {
    fn default() -> Self {
        Self::XY
    }
}

/// All quadrants of a quadtree. The order of the quadrants is unspecified, all functions that care about quadrant order
/// take an additional `QuadrantOrder` parameter to specify the order
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Quadrant {
    /// The first quadrant
    Zero,
    /// The second quadrant
    One,
    /// The third quadrant
    Two,
    /// The foruth quadrant
    Three,
}

impl Quadrant {
    pub fn index(&self) -> usize {
        self.into()
    }
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

impl Into<usize> for &Quadrant {
    fn into(self) -> usize {
        (*self).into()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadrant_orderings() {
        assert_eq!(
            Vector2::new(0_usize, 0_usize),
            QuadrantOrdering::XY.to_index(Quadrant::Zero)
        );
        assert_eq!(
            Vector2::new(1_usize, 0_usize),
            QuadrantOrdering::XY.to_index(Quadrant::One)
        );
        assert_eq!(
            Vector2::new(0_usize, 1_usize),
            QuadrantOrdering::XY.to_index(Quadrant::Two)
        );
        assert_eq!(
            Vector2::new(1_usize, 1_usize),
            QuadrantOrdering::XY.to_index(Quadrant::Three)
        );

        assert_eq!(
            Vector2::new(0_usize, 0_usize),
            QuadrantOrdering::YX.to_index(Quadrant::Zero)
        );
        assert_eq!(
            Vector2::new(0_usize, 1_usize),
            QuadrantOrdering::YX.to_index(Quadrant::One)
        );
        assert_eq!(
            Vector2::new(1_usize, 0_usize),
            QuadrantOrdering::YX.to_index(Quadrant::Two)
        );
        assert_eq!(
            Vector2::new(1_usize, 1_usize),
            QuadrantOrdering::YX.to_index(Quadrant::Three)
        );
    }
}
