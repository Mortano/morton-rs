use core::ops::Range;

macro_rules! impl_bits {
    () => {
        unsafe fn get_bits(&self, bit_range: Range<usize>) -> Self {
            let shift = bit_range.start as Self;
            let num_bits = bit_range.len() as Self;
            let mask = ((1 as Self) << num_bits) - 1;
            (*self >> shift) & mask
        }
        unsafe fn set_bits(&mut self, bit_range: Range<usize>, new_value: Self) {
            let shift = bit_range.start as Self;
            let num_bits = bit_range.len() as Self;
            let mask = ((1 as Self) << num_bits) - 1;
            let clear_mask = !(mask << shift);
            *self = (*self & clear_mask) | (new_value << shift)
        }

        unsafe fn as_u8(&self) -> u8 { *self as u8 }
        unsafe fn as_u16(&self) -> u16 { *self as u16 }
        unsafe fn as_u32(&self) -> u32 { *self as u32 }
        unsafe fn as_u64(&self) -> u64 { *self as u64 }
        unsafe fn as_u128(&self) -> u128 { *self as u128 }
        unsafe fn as_vec_u8(&self) -> Vec<u8> {
            bytemuck::bytes_of(self).to_owned()
        }

        unsafe fn from_u8(val: u8) -> Self { val as Self }
        unsafe fn from_u16(val: u16) -> Self { val as Self }
        unsafe fn from_u32(val: u32) -> Self { val as Self }
        unsafe fn from_u64(val: u64) -> Self { val as Self }
        unsafe fn from_u128(val: u128) -> Self { val as Self }
    }
}

pub trait Bits : Sized {
    const BITS: usize;

    unsafe fn get_bits(&self, bit_range: Range<usize>) -> Self;
    unsafe fn set_bits(&mut self, bit_range: Range<usize>, new_value: Self);

    // We don't use From/Into because these aren't implemented for all combinations of unsigned numbers
    // Some of these conversions also aren't always well-defined (how to go from u64 to u8? Truncating? Shifting?), so
    // we provide our own unsafe variants for conversions

    unsafe fn as_u8(&self) -> u8;
    unsafe fn as_u16(&self) -> u16;
    unsafe fn as_u32(&self) -> u32;
    unsafe fn as_u64(&self) -> u64;
    unsafe fn as_u128(&self) -> u128;
    unsafe fn as_vec_u8(&self) -> Vec<u8>;

    unsafe fn from_u8(val: u8) -> Self;
    unsafe fn from_u16(val: u16) -> Self;
    unsafe fn from_u32(val: u32) -> Self;
    unsafe fn from_u64(val: u64) -> Self;
    unsafe fn from_u128(val: u128) -> Self;
}

impl Bits for u8 {
    const BITS: usize = 8;

    impl_bits!{}
}

impl Bits for u16 {
    const BITS: usize = 16;

    impl_bits!{}
}

impl Bits for u32 {
    const BITS: usize = 32;

    impl_bits!{}
}

impl Bits for u64 {
    const BITS: usize = 64;

    impl_bits!{}
}

impl Bits for u128 {
    const BITS: usize = 128;

    impl_bits!{}
}

// TODO We could implement the following, but it is a bit more involved due to the get/set bits calls

// impl <const N: usize> Bits for [u8; N] {
//     const BITS : usize = N * 8;

//     unsafe fn get_bits(&self, bit_range: Range<usize>) -> Self {
//         todo!()
//     }

//     unsafe fn set_bits(&mut self, bit_range: Range<usize>, new_value: Self) {
//         todo!()
//     }
// }