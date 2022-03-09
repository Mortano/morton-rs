use nalgebra::Vector3;

use super::Dimension;

/// A `Dimension` representing 3D space
pub struct Dim3D;

impl Dimension for Dim3D {
    type Cell = Octant;
    type GridIndex = Vector3<usize>;
    type CellOrdering = OctantOrdering;

    const DIMENSIONALITY: usize = 3;
}

/// Ordering of octants in 3D space
#[derive(Debug, Clone, Copy)]
pub enum OctantOrdering {
    /// X-then-Y-then-Z ordering, which maps octant 1 to index `(X=1, Y=0, Z=0)`, octant 2 to index `(X=0, Y=1, Z=0)` and octant
    /// 4 to index `(X=0, Y=0, Z=1)`. This is the default `OctantOrdering`!
    XYZ,
    /// X-then-Z-then-Y ordering, which maps octant 1 to index `(X=1, Y=0, Z=0)`, octant 2 to index `(X=0, Y=0, Z=1)` and octant
    /// 4 to index `(X=0, Y=1, Z=0)`
    XZY,
    /// Y-then-X-then-Z ordering, which maps octant 1 to index `(X=0, Y=1, Z=0)`, octant 2 to index `(X=1, Y=0, Z=0)` and octant
    /// 4 to index `(X=0, Y=0, Z=1)`
    YXZ,
    /// Y-then-Z-then-X ordering, which maps octant 1 to index `(X=0, Y=1, Z=0)`, octant 2 to index `(X=0, Y=0, Z=1)` and octant
    /// 4 to index `(X=1, Y=0, Z=0)`
    YZX,
    /// Z-then-X-then-Y ordering, which maps octant 1 to index `(X=0, Y=0, Z=1)`, octant 2 to index `(X=1, Y=0, Z=0)` and octant
    /// 4 to index `(X=0, Y=1, Z=0)`
    ZXY,
    /// Z-then-Y-then-X ordering, which maps octant 1 to index `(X=0, Y=0, Z=1)`, octant 2 to index `(X=0, Y=1, Z=0)` and octant
    /// 4 to index `(X=1, Y=0, Z=0)`
    ZYX,
}

impl OctantOrdering {
    pub fn to_index(&self, octant: Octant) -> Vector3<usize> {
        let octant_index = octant.index();
        match self {
            OctantOrdering::XYZ => Vector3::new(
                (octant_index >> 0) & 1,
                (octant_index >> 1) & 1,
                (octant_index >> 2) & 1,
            ),
            OctantOrdering::XZY => Vector3::new(
                (octant_index >> 0) & 1,
                (octant_index >> 2) & 1,
                (octant_index >> 1) & 1,
            ),
            OctantOrdering::YXZ => Vector3::new(
                (octant_index >> 1) & 1,
                (octant_index >> 0) & 1,
                (octant_index >> 2) & 1,
            ),
            OctantOrdering::YZX => Vector3::new(
                (octant_index >> 2) & 1,
                (octant_index >> 0) & 1,
                (octant_index >> 1) & 1,
            ),
            OctantOrdering::ZXY => Vector3::new(
                (octant_index >> 1) & 1,
                (octant_index >> 2) & 1,
                (octant_index >> 0) & 1,
            ),
            OctantOrdering::ZYX => Vector3::new(
                (octant_index >> 2) & 1,
                (octant_index >> 1) & 1,
                (octant_index >> 0) & 1,
            ),
        }
    }
}

impl Default for OctantOrdering {
    fn default() -> Self {
        OctantOrdering::XYZ
    }
}

/// All octants of an octree. The order of the octants is unspecified, all functions that care about octant order
/// take an additional `OctantOrder` parameter to specify the order
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Octant {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
}

impl Octant {
    pub fn index(&self) -> usize {
        self.into()
    }
}

impl Into<usize> for Octant {
    fn into(self) -> usize {
        match self {
            Octant::Zero => 0,
            Octant::One => 1,
            Octant::Two => 2,
            Octant::Three => 3,
            Octant::Four => 4,
            Octant::Five => 5,
            Octant::Six => 6,
            Octant::Seven => 7,
        }
    }
}

impl Into<usize> for &Octant {
    fn into(self) -> usize {
        (*self).into()
    }
}

impl TryFrom<usize> for Octant {
    type Error = crate::Error;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Octant::Zero),
            1 => Ok(Octant::One),
            2 => Ok(Octant::Two),
            3 => Ok(Octant::Three),
            4 => Ok(Octant::Four),
            5 => Ok(Octant::Five),
            6 => Ok(Octant::Six),
            7 => Ok(Octant::Seven),
            _ => Err(crate::Error::CellIndexOutOfRange),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_bit_shifts_for_xyz(octant_ordering: OctantOrdering) -> (usize, usize, usize) {
        match octant_ordering {
            OctantOrdering::XYZ => (0, 1, 2),
            OctantOrdering::XZY => (0, 2, 1),
            OctantOrdering::YXZ => (1, 0, 2),
            OctantOrdering::YZX => (2, 0, 1),
            OctantOrdering::ZXY => (1, 2, 0),
            OctantOrdering::ZYX => (2, 1, 0),
        }
    }

    #[test]
    fn octant_orderings() {
        // This tests all octants with all possible octant orderings, procedurally
        for ordering in [
            OctantOrdering::XYZ,
            OctantOrdering::XZY,
            OctantOrdering::YXZ,
            OctantOrdering::YZX,
            OctantOrdering::ZXY,
            OctantOrdering::ZYX,
        ] {
            let (x_sh, y_sh, z_sh) = get_bit_shifts_for_xyz(ordering);
            for octant_index in 0..8 {
                let octant = Octant::try_from(octant_index).unwrap();
                let expected_vec = Vector3::new(
                    (octant_index >> x_sh) & 1,
                    (octant_index >> y_sh) & 1,
                    (octant_index >> z_sh) & 1,
                );
                assert_eq!(expected_vec, ordering.to_index(octant));
            }
        }
    }
}
