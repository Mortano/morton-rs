use core::ops::Range;

macro_rules! impl_bits {
    () => {
        unsafe fn get_bits(&self, bit_range: Range<usize>) -> Self {
            let shift = bit_range.start as Self;
            let shift_right = std::mem::size_of::<Self>() * 8 - bit_range.len() as usize;
            // The more you know: Shifting left or right by WORDSIZE is actually undefined behaviour
            // (see here: https://users.rust-lang.org/t/intentionally-overflow-on-shift/11859)
            // So we have to use a trick to get the correct bit mask here
            let mask = (!(0 as Self)).checked_shr(shift_right as u32).unwrap_or(0);
            (*self >> shift) & mask
        }
        unsafe fn set_bits(&mut self, bit_range: Range<usize>, new_value: Self) {
            let shift = bit_range.start as Self;
            let shift_right = std::mem::size_of::<Self>() * 8 - bit_range.len() as usize;
            let mask = (!(0 as Self)).checked_shr(shift_right as u32).unwrap_or(0);
            let clear_mask = !(mask << shift);
            *self = (*self & clear_mask) | (new_value << shift)
        }

        unsafe fn as_u8(&self) -> u8 {
            *self as u8
        }
        unsafe fn as_u16(&self) -> u16 {
            *self as u16
        }
        unsafe fn as_u32(&self) -> u32 {
            *self as u32
        }
        unsafe fn as_u64(&self) -> u64 {
            *self as u64
        }
        unsafe fn as_u128(&self) -> u128 {
            *self as u128
        }
        unsafe fn as_u8_slice(&self) -> &[u8] {
            bytemuck::bytes_of(self)
        }

        unsafe fn from_u8(val: u8) -> Self {
            val as Self
        }
        unsafe fn from_u16(val: u16) -> Self {
            val as Self
        }
        unsafe fn from_u32(val: u32) -> Self {
            val as Self
        }
        unsafe fn from_u64(val: u64) -> Self {
            val as Self
        }
        unsafe fn from_u128(val: u128) -> Self {
            val as Self
        }
    };
}

pub enum Endianness {
    BigEndian,
    LittleEndian,
}

pub trait Bits: Sized {
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
    unsafe fn as_u8_slice(&self) -> &[u8];

    unsafe fn from_u8(val: u8) -> Self;
    unsafe fn from_u16(val: u16) -> Self;
    unsafe fn from_u32(val: u32) -> Self;
    unsafe fn from_u64(val: u64) -> Self;
    unsafe fn from_u128(val: u128) -> Self;
    /// Converts a slice of bytes into this `Bits` value using the given `endianness`. Missing bytes are filled with
    /// zero values, excess bytes are ignored!
    unsafe fn from_u8_slice(bytes: &[u8], endianness: Endianness) -> Self;
}

impl Bits for u8 {
    const BITS: usize = 8;

    impl_bits! {}

    unsafe fn from_u8_slice(bytes: &[u8], _endianness: Endianness) -> Self {
        if bytes.len() == 0 {
            0
        } else {
            bytes[0]
        }
    }
}

impl Bits for u16 {
    const BITS: usize = 16;

    impl_bits! {}

    unsafe fn from_u8_slice(bytes: &[u8], endianness: Endianness) -> Self {
        if bytes.len() < 2 {
            let pad_start = bytes.len();
            let mut padded_bytes = [0_u8; 2];
            padded_bytes[0..pad_start].copy_from_slice(bytes);
            match endianness {
                Endianness::BigEndian => u16::from_be_bytes(padded_bytes),
                Endianness::LittleEndian => u16::from_le_bytes(padded_bytes),
            }
        } else {
            let trimmed_bytes = [bytes[0], bytes[1]];
            match endianness {
                Endianness::BigEndian => u16::from_be_bytes(trimmed_bytes),
                Endianness::LittleEndian => u16::from_le_bytes(trimmed_bytes),
            }
        }
    }
}

impl Bits for u32 {
    const BITS: usize = 32;

    impl_bits! {}

    unsafe fn from_u8_slice(bytes: &[u8], endianness: Endianness) -> Self {
        if bytes.len() < 4 {
            let pad_start = bytes.len();
            let mut padded_bytes = [0_u8; 4];
            padded_bytes[0..pad_start].copy_from_slice(bytes);
            match endianness {
                Endianness::BigEndian => u32::from_be_bytes(padded_bytes),
                Endianness::LittleEndian => u32::from_le_bytes(padded_bytes),
            }
        } else {
            let trimmed_bytes = [bytes[0], bytes[1], bytes[2], bytes[3]];
            match endianness {
                Endianness::BigEndian => u32::from_be_bytes(trimmed_bytes),
                Endianness::LittleEndian => u32::from_le_bytes(trimmed_bytes),
            }
        }
    }
}

impl Bits for u64 {
    const BITS: usize = 64;

    impl_bits! {}

    unsafe fn from_u8_slice(bytes: &[u8], endianness: Endianness) -> Self {
        if bytes.len() < 8 {
            let pad_start = bytes.len();
            let mut padded_bytes = [0_u8; 8];
            padded_bytes[0..pad_start].copy_from_slice(bytes);
            match endianness {
                Endianness::BigEndian => u64::from_be_bytes(padded_bytes),
                Endianness::LittleEndian => u64::from_le_bytes(padded_bytes),
            }
        } else {
            let trimmed_bytes = [
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ];
            match endianness {
                Endianness::BigEndian => u64::from_be_bytes(trimmed_bytes),
                Endianness::LittleEndian => u64::from_le_bytes(trimmed_bytes),
            }
        }
    }
}

impl Bits for u128 {
    const BITS: usize = 128;

    impl_bits! {}

    unsafe fn from_u8_slice(bytes: &[u8], endianness: Endianness) -> Self {
        if bytes.len() < 16 {
            let pad_start = bytes.len();
            let mut padded_bytes = [0_u8; 16];
            padded_bytes[0..pad_start].copy_from_slice(bytes);
            match endianness {
                Endianness::BigEndian => u128::from_be_bytes(padded_bytes),
                Endianness::LittleEndian => u128::from_le_bytes(padded_bytes),
            }
        } else {
            let trimmed_bytes = [
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                bytes[15],
            ];
            match endianness {
                Endianness::BigEndian => u128::from_be_bytes(trimmed_bytes),
                Endianness::LittleEndian => u128::from_le_bytes(trimmed_bytes),
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_bits_u8() {
        unsafe {
            let mut val: u8 = 0;
            val.set_bits(0..2, 0b11);
            assert_eq!(0b11, val);
            assert_eq!(0b11, val.get_bits(0..2));

            val = 0;
            val.set_bits(2..4, 0b11);
            assert_eq!(0b1100, val);
            assert_eq!(0b11, val.get_bits(2..4));

            val = 0;
            val.set_bits(6..8, 0b11);
            assert_eq!(0b11000000, val);
            assert_eq!(0b11, val.get_bits(6..8));

            val = 0;
            val.set_bits(0..8, 0b1);
            assert_eq!(0b1, val);
            assert_eq!(0b1, val.get_bits(0..8));

            val = 0;
            val.set_bits(0..8, 0b11111111);
            assert_eq!(0b11111111, val);
            assert_eq!(0b11111111, val.get_bits(0..8));
        }
    }

    #[test]
    fn set_bits_u16() {
        unsafe {
            let mut val: u16 = 0;
            val.set_bits(0..3, 0b111);
            assert_eq!(0b111, val);
            assert_eq!(0b111, val.get_bits(0..3));

            val = 0;
            val.set_bits(3..6, 0b111);
            assert_eq!(0b111000, val);
            assert_eq!(0b111, val.get_bits(3..6));

            val = 0;
            val.set_bits(8..11, 0b111);
            assert_eq!(0b111_00000000, val);
            assert_eq!(0b111, val.get_bits(8..11));

            val = 0;
            val.set_bits(6..9, 0b111);
            assert_eq!(0b1_11000000, val);
            assert_eq!(0b111, val.get_bits(6..9));

            val = 0;
            val.set_bits(0..8, u8::MAX as u16);
            assert_eq!(u8::MAX as u16, val);
            assert_eq!(u8::MAX as u16, val.get_bits(0..8));

            val = 0;
            val.set_bits(0..16, u16::MAX);
            assert_eq!(u16::MAX, val);
            assert_eq!(u16::MAX, val.get_bits(0..16));

            val = 0;
            val.set_bits(0..2, 0b11);
            val.set_bits(9..12, 0b101);
            assert_eq!(0b1010_00000011, val);
            assert_eq!(0b101, val.get_bits(9..12));
            assert_eq!(0b11, val.get_bits(0..2));
        }
    }
}
