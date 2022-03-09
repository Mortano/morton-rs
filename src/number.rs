use core::ops::Range;

use num_traits::Unsigned;

use crate::align::Alignable;

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
        unsafe fn get_bits_as_usize(&self, bit_range: Range<usize>) -> usize {
            self.get_bits(bit_range) as usize
        }
        unsafe fn set_bits(&mut self, bit_range: Range<usize>, new_value: Self) {
            let shift = bit_range.start as Self;
            let shift_right = std::mem::size_of::<Self>() * 8 - bit_range.len() as usize;
            let mask = (!(0 as Self)).checked_shr(shift_right as u32).unwrap_or(0);
            let clear_mask = !(mask << shift);
            *self = (*self & clear_mask) | (new_value << shift)
        }
        unsafe fn set_bits_from_usize(&mut self, bit_range: Range<usize>, new_value: usize) {
            self.set_bits(bit_range, new_value as Self)
        }

        unsafe fn as_u8(self) -> u8 {
            self as u8
        }
        unsafe fn as_u16(self) -> u16 {
            self as u16
        }
        unsafe fn as_u32(self) -> u32 {
            self as u32
        }
        unsafe fn as_u64(self) -> u64 {
            self as u64
        }
        unsafe fn as_u128(self) -> u128 {
            self as u128
        }
        unsafe fn as_usize(self) -> usize {
            self as usize
        }
        unsafe fn as_u8_slice(&self) -> &[u8] {
            bytemuck::bytes_of(self)
        }
        unsafe fn as_u8_slice_mut(&mut self) -> &mut [u8] {
            bytemuck::bytes_of_mut(self)
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
        unsafe fn from_usize(val: usize) -> Self {
            val as Self
        }

        fn size(&self) -> usize {
            std::mem::size_of::<Self>()
        }
    };
}

pub enum Endianness {
    BigEndian,
    LittleEndian,
}

pub trait Bits {
    const BITS: usize;

    unsafe fn get_bits(&self, bit_range: Range<usize>) -> Self;
    unsafe fn get_bits_as_usize(&self, bit_range: Range<usize>) -> usize;
    unsafe fn set_bits(&mut self, bit_range: Range<usize>, new_value: Self);
    unsafe fn set_bits_from_usize(&mut self, bit_range: Range<usize>, new_value: usize);

    // We don't use From/Into because these aren't implemented for all combinations of unsigned numbers
    // Some of these conversions also aren't always well-defined (how to go from u64 to u8? Truncating? Shifting?), so
    // we provide our own unsafe variants for conversions

    unsafe fn as_u8(self) -> u8;
    unsafe fn as_u16(self) -> u16;
    unsafe fn as_u32(self) -> u32;
    unsafe fn as_u64(self) -> u64;
    unsafe fn as_u128(self) -> u128;
    unsafe fn as_usize(self) -> usize;
    unsafe fn as_u8_slice(&self) -> &[u8];
    unsafe fn as_u8_slice_mut(&mut self) -> &mut [u8];

    unsafe fn from_u8(val: u8) -> Self;
    unsafe fn from_u16(val: u16) -> Self;
    unsafe fn from_u32(val: u32) -> Self;
    unsafe fn from_u64(val: u64) -> Self;
    unsafe fn from_u128(val: u128) -> Self;
    unsafe fn from_usize(val: usize) -> Self;
    /// Converts a slice of bytes into this `Bits` value using the given `endianness`. Missing bytes are filled with
    /// zero values, excess bytes are ignored!
    unsafe fn from_u8_slice(bytes: &[u8], endianness: Endianness) -> Self;

    fn size(&self) -> usize;
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

impl Bits for usize {
    const BITS: usize = std::mem::size_of::<usize>() * 8;

    impl_bits! {}

    #[cfg(target_pointer_width = "16")]
    unsafe fn from_u8_slice(bytes: &[u8], endianness: Endianness) -> Self {
        u16::from_u8_slice(bytes, endianness) as usize
    }

    #[cfg(target_pointer_width = "32")]
    unsafe fn from_u8_slice(bytes: &[u8], endianness: Endianness) -> Self {
        u32::from_u8_slice(bytes, endianness) as usize
    }

    #[cfg(target_pointer_width = "64")]
    unsafe fn from_u8_slice(bytes: &[u8], endianness: Endianness) -> Self {
        u64::from_u8_slice(bytes, endianness) as usize
    }

    #[cfg(not(any(
        target_pointer_width = "16",
        target_pointer_width = "32",
        target_pointer_width = "64"
    )))]
    unsafe fn from_u8_slice(bytes: &[u8], endianness: Endianness) -> Self {
        std::compile_error!("Invalid pointer width on the current platform! morton-rs only supports 16-bit, 32-bit, and 64-bit platforms")
    }
}

unsafe fn set_vec_u8_bits(vec: &mut Vec<u8>, bit_range: Range<usize>, new_value: &[u8]) {
    // Iterate over self in 8-bit chunks and set the appropriate stuff from new_value
    // This might require that we combine two u8 values into one

    let self_start_byte = bit_range.start / 8;
    let self_end_byte = (bit_range.end + 7) / 8;

    let mut self_start_bit = bit_range.start;
    let mut new_val_start_bit: usize = 0;

    for self_byte_idx in self_start_byte..self_end_byte {
        let self_end_bit = ((self_byte_idx + 1) * 8).min(bit_range.end);
        let num_bits_in_chunk = self_end_bit - self_start_bit;
        let new_val_end_bit = new_val_start_bit + num_bits_in_chunk;
        let new_val_byte_idx = new_val_start_bit / 8;

        // Get the u8 value to set into the current byte from new_value
        // dst_start..dst_end either fits into the bounds of a single u8, or we have
        // to combine 2 u8 values from new_value
        let combined_byte = {
            if new_val_byte_idx == new_value.len() - 1 {
                let dst_start_in_byte = new_val_start_bit % 8;
                let dst_end_in_byte = dst_start_in_byte + num_bits_in_chunk;

                new_value[new_val_byte_idx].get_bits(dst_start_in_byte..dst_end_in_byte)
            } else {
                let ptr = new_value.as_ptr().add(new_val_byte_idx) as *const u16;
                let two_bytes = ptr.read_unaligned();
                let start_bit_in_two_bytes = new_val_start_bit - (new_val_byte_idx * 8);
                let end_bit_in_two_bytes = new_val_end_bit - (new_val_byte_idx * 8);
                two_bytes.get_bits(start_bit_in_two_bytes..end_bit_in_two_bytes) as u8
            }
        };
        vec[self_byte_idx].set_bits(
            (self_start_bit - (self_byte_idx * 8))..(self_end_bit - (self_byte_idx * 8)),
            combined_byte,
        );

        self_start_bit = self_end_bit;
        new_val_start_bit += num_bits_in_chunk;
    }
}

impl Bits for Vec<u8> {
    const BITS: usize = 0;

    unsafe fn get_bits(&self, bit_range: Range<usize>) -> Self {
        // Copy all relevant bytes, then shift each byte to the left and carry over bits from higher to lower bytes

        let num_bits = bit_range.len();
        let num_bytes = (num_bits + 7) / 8;
        let start_byte = bit_range.start / 8;
        let end_byte = (bit_range.end + 7) / 8;
        let shift_within_byte = bit_range.start % 8;

        let mut bytes = self[start_byte..end_byte].to_owned();
        for idx in 0..bytes.len() {
            if idx > 0 && shift_within_byte > 0 {
                // Take the part of the byte that gets shifted 'out' of the current byte, and move it over to the
                // previous byte
                let carry = bytes[idx].get_bits(0..shift_within_byte);
                bytes[idx - 1].set_bits((8 - shift_within_byte)..8, carry);
            }
            if idx == bytes.len() - 1 {
                // Mask away the upper bits outside of the requested range on the last byte
                let excess_bits = (end_byte * 8) - bit_range.end;
                // There can never be overflow here, because excess_bits always will be less than 8!
                let mask = 0xFF_u8 >> excess_bits;
                bytes[idx] &= mask;
            }
            bytes[idx] >>= shift_within_byte;
        }

        // We might have to trim away the last byte if it has been shifted away completely
        if (end_byte - start_byte) > num_bytes {
            bytes.pop();
        }

        bytes
    }

    unsafe fn set_bits(&mut self, bit_range: Range<usize>, new_value: Self) {
        set_vec_u8_bits(self, bit_range, new_value.as_slice());
    }

    unsafe fn as_u8(self) -> u8 {
        self[0]
    }

    unsafe fn as_u16(self) -> u16 {
        u16::from_ne_bytes(self.try_into().unwrap())
    }

    unsafe fn as_u32(self) -> u32 {
        u32::from_ne_bytes(self.try_into().unwrap())
    }

    unsafe fn as_u64(self) -> u64 {
        u64::from_ne_bytes(self.try_into().unwrap())
    }

    unsafe fn as_u128(self) -> u128 {
        u128::from_ne_bytes(self.try_into().unwrap())
    }

    unsafe fn as_usize(self) -> usize {
        usize::from_ne_bytes(self.try_into().unwrap())
    }

    unsafe fn as_u8_slice(&self) -> &[u8] {
        self
    }

    unsafe fn as_u8_slice_mut(&mut self) -> &mut [u8] {
        self
    }

    unsafe fn from_u8(val: u8) -> Self {
        todo!()
    }

    unsafe fn from_u16(val: u16) -> Self {
        todo!()
    }

    unsafe fn from_u32(val: u32) -> Self {
        todo!()
    }

    unsafe fn from_u64(val: u64) -> Self {
        todo!()
    }

    unsafe fn from_u128(val: u128) -> Self {
        todo!()
    }

    unsafe fn from_usize(val: usize) -> Self {
        todo!()
    }

    unsafe fn from_u8_slice(bytes: &[u8], endianness: Endianness) -> Self {
        todo!()
    }

    unsafe fn get_bits_as_usize(&self, bit_range: Range<usize>) -> usize {
        let start_byte = bit_range.start / 8;
        let end_byte = (bit_range.end + 7) / 8;
        let mut current_start_bit = bit_range.start;
        let mut usize_start_bit: usize = 0;

        let mut ret: usize = 0;
        for idx in start_byte..end_byte {
            let current_end_bit = ((idx + 1) * 8).min(bit_range.end);
            let bits_within_current_byte = current_end_bit - current_start_bit;

            let start_within_byte = current_start_bit % 8;
            let end_within_byte = start_within_byte + bits_within_current_byte;

            let byte = self[idx].get_bits(start_within_byte..end_within_byte);

            // Put it at the right position in 'ret'
            let usize_end_bit = usize_start_bit + bits_within_current_byte;
            ret.set_bits(usize_start_bit..usize_end_bit, byte as usize);

            current_start_bit = current_end_bit;
            usize_start_bit += bits_within_current_byte;
        }

        ret
    }

    unsafe fn set_bits_from_usize(&mut self, bit_range: Range<usize>, new_value: usize) {
        let bytes = bytemuck::bytes_of(&new_value);
        set_vec_u8_bits(self, bit_range, bytes);
    }

    fn size(&self) -> usize {
        self.len()
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

/// Take the lower 4 bits of `val` and adds a zero behind every bit.
/// For example `0b1101` becomes `0b10_10_00_10`
pub fn add_zero_behind_every_bit_u8(val: u8) -> u8 {
    add_zero_before_every_bit_u8(val) << 1
}

/// Take the lower 4 bits of `val` and adds a zero before every bit.
/// For example `0b1101` becomes `0b01_01_00_01`
pub fn add_zero_before_every_bit_u8(val: u8) -> u8 {
    let mut ret = val;
    ret = (ret | (ret << 2)) & 0b00110011;
    ret = (ret | (ret << 1)) & 0b01010101;
    ret
}

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

    #[test]
    fn test_expand_u8_by_2() {
        assert_eq!(0 as u8, add_zero_behind_every_bit_u8(0 as u8));
        assert_eq!(0b10101010 as u8, add_zero_behind_every_bit_u8(0b1111 as u8));
        assert_eq!(0b00001010 as u8, add_zero_behind_every_bit_u8(0b0011 as u8));
        assert_eq!(0b10000010 as u8, add_zero_behind_every_bit_u8(0b1001 as u8));
        assert_eq!(0b10001000 as u8, add_zero_behind_every_bit_u8(0b1010 as u8));

        assert_eq!(0 as u8, add_zero_before_every_bit_u8(0 as u8));
        assert_eq!(0b01010101 as u8, add_zero_before_every_bit_u8(0b1111 as u8));
        assert_eq!(0b00000101 as u8, add_zero_before_every_bit_u8(0b0011 as u8));
        assert_eq!(0b01000001 as u8, add_zero_before_every_bit_u8(0b1001 as u8));
        assert_eq!(0b01000100 as u8, add_zero_before_every_bit_u8(0b1010 as u8));
    }

    #[test]
    fn test_get_bits_vec_u8() {
        // We expect reading to take place in the native endianness of the current machine
        // On little endian, we have the MSB on the right (last byte of the vec)
        // On big endian, we have the MSB on the left (first byte of the vec)
        let vec: Vec<u8> = vec![0b1010_1010, 0b0110_1001, 0b0011_1010, 0b1110_1110];

        #[cfg(target_endian = "little")]
        const EXPECTED: u32 = 0b11101110_00111010_01101001_10101010;
        #[cfg(target_endian = "big")]
        const EXPECTED: u32 = 0b10101010_01101001_00111010_11101110;

        unsafe {
            // Base case of a single aligned byte
            assert_eq!(vec![EXPECTED.get_bits(0..8) as u8], vec.get_bits(0..8));
            // Less than a byte, but within one byte
            assert_eq!(vec![EXPECTED.get_bits(2..6) as u8], vec.get_bits(2..6));

            // Across more than one byte
            #[cfg(target_endian = "little")]
            assert_eq!(
                vec![
                    EXPECTED.get_bits(0..8) as u8,
                    EXPECTED.get_bits(8..12) as u8
                ],
                vec.get_bits(0..12)
            );
            #[cfg(target_endian = "big")]
            assert_eq!(
                vec![
                    EXPECTED.get_bits(8..12) as u8,
                    EXPECTED.get_bits(0..8) as u8
                ],
                vec.get_bits(0..12)
            );

            // Across more than one byte, but resulting in only a single byte
            assert_eq!(vec![EXPECTED.get_bits(4..12) as u8], vec.get_bits(4..12));

            // Across more than one byte, multiple bytes
            #[cfg(target_endian = "little")]
            assert_eq!(
                vec![
                    EXPECTED.get_bits(12..20) as u8,
                    EXPECTED.get_bits(20..28) as u8,
                    EXPECTED.get_bits(28..32) as u8
                ],
                vec.get_bits(12..32)
            );
            #[cfg(target_endian = "big")]
            assert_eq!(
                vec![
                    EXPECTED.get_bits(28..32) as u8,
                    EXPECTED.get_bits(20..28) as u8,
                    EXPECTED.get_bits(12..20) as u8
                ],
                vec.get_bits(12..32)
            );

            // Identity case of all bytes
            assert_eq!(vec, vec.get_bits(0..32));
        }
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn set_bits_vec_u8() {
        let mut vec: Vec<u8> = vec![0; 4];

        unsafe {
            vec.set_bits(0..8, vec![0b11000101]);
            assert_eq!(vec![0b11000101, 0, 0, 0], vec);

            vec = vec![0; 4];

            vec.set_bits(2..6, vec![0b1111]);
            assert_eq!(vec![0b00111100, 0, 0, 0], vec);

            vec = vec![0; 4];

            vec.set_bits(20..24, vec![0b1111]);
            assert_eq!(vec![0, 0, 0b11110000, 0], vec);

            vec = vec![0; 4];

            vec.set_bits(0..12, vec![0b11001010, 0b1001]);
            assert_eq!(vec![0b11001010, 0b00001001, 0, 0], vec);

            vec = vec![0; 4];

            vec.set_bits(4..16, vec![0b11001010, 0b1001]);
            assert_eq!(vec![0b10100000, 0b10011100, 0, 0], vec);

            vec = vec![0; 4];

            vec.set_bits(3..23, vec![0b11001010, 0b10010111, 0b1110]);
            assert_eq!(vec![0b01010000, 0b10111110, 0b01110100, 0], vec);

            vec = vec![0; 4];

            let expected: Vec<u8> = vec![0b1010_1010, 0b0110_1001, 0b0011_1010, 0b1110_1110];
            vec.set_bits(0..32, expected.clone());
            assert_eq!(expected, vec);
        }
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn get_bits_as_usize_vec_u8() {
        let vec: Vec<u8> = vec![0b1010_1010, 0b0110_1001, 0b0011_1010, 0b1110_1110];
        const EXPECTED: usize = 0b11101110_00111010_01101001_10101010;

        unsafe {
            assert_eq!(EXPECTED.get_bits(0..8), vec.get_bits_as_usize(0..8));
            assert_eq!(EXPECTED.get_bits(2..6), vec.get_bits_as_usize(2..6));
            assert_eq!(EXPECTED.get_bits(20..24), vec.get_bits_as_usize(20..24));
            assert_eq!(EXPECTED.get_bits(0..12), vec.get_bits_as_usize(0..12));
            assert_eq!(EXPECTED.get_bits(1..13), vec.get_bits_as_usize(1..13));
            assert_eq!(EXPECTED.get_bits(4..24), vec.get_bits_as_usize(4..24));
            assert_eq!(EXPECTED.get_bits(3..23), vec.get_bits_as_usize(3..23));
            assert_eq!(EXPECTED.get_bits(0..32), vec.get_bits_as_usize(0..32));
        }
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn set_bits_from_usize_vec_u8() {
        let mut vec: Vec<u8> = vec![0; 4];

        unsafe {
            vec.set_bits_from_usize(0..8, 0b11000101);
            assert_eq!(vec![0b11000101, 0, 0, 0], vec);

            vec = vec![0; 4];

            vec.set_bits_from_usize(2..6, 0b1111);
            assert_eq!(vec![0b00111100, 0, 0, 0], vec);

            vec = vec![0; 4];

            vec.set_bits_from_usize(20..24, 0b1111);
            assert_eq!(vec![0, 0, 0b11110000, 0], vec);

            vec = vec![0; 4];

            vec.set_bits_from_usize(0..12, 0b1001_11001010);
            assert_eq!(vec![0b11001010, 0b00001001, 0, 0], vec);

            vec = vec![0; 4];

            vec.set_bits_from_usize(4..16, 0b1001_11001010);
            assert_eq!(vec![0b10100000, 0b10011100, 0, 0], vec);

            vec = vec![0; 4];

            vec.set_bits_from_usize(3..23, 0b1110_10010111_11001010);
            assert_eq!(vec![0b01010000, 0b10111110, 0b01110100, 0], vec);

            vec = vec![0; 4];

            let expected: Vec<u8> = vec![0b1010_1010, 0b0110_1001, 0b0011_1010, 0b1110_1110];
            vec.set_bits_from_usize(0..32, 0b11101110_00111010_01101001_10101010);
            assert_eq!(expected, vec);
        }
    }
}
