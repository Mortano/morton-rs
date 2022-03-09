/// Trait for aligning a numeric value to a given byte boundary
pub trait Alignable {
    /// Align the associated value to an `alignment` bytes boundary
    fn align_to(&self, alignment: Self) -> Self;
}

impl Alignable for u8 {
    fn align_to(&self, alignment: Self) -> Self {
        if alignment == 0 {
            *self
        } else {
            ((self + alignment - 1) / alignment) * alignment
        }
    }
}

impl Alignable for u16 {
    fn align_to(&self, alignment: Self) -> Self {
        if alignment == 0 {
            *self
        } else {
            ((self + alignment - 1) / alignment) * alignment
        }
    }
}

impl Alignable for u32 {
    fn align_to(&self, alignment: Self) -> Self {
        if alignment == 0 {
            *self
        } else {
            ((self + alignment - 1) / alignment) * alignment
        }
    }
}

impl Alignable for u64 {
    fn align_to(&self, alignment: Self) -> Self {
        if alignment == 0 {
            *self
        } else {
            ((self + alignment - 1) / alignment) * alignment
        }
    }
}

impl Alignable for u128 {
    fn align_to(&self, alignment: Self) -> Self {
        if alignment == 0 {
            *self
        } else {
            ((self + alignment - 1) / alignment) * alignment
        }
    }
}

impl Alignable for usize {
    fn align_to(&self, alignment: Self) -> Self {
        if alignment == 0 {
            *self
        } else {
            ((self + alignment - 1) / alignment) * alignment
        }
    }
}
