use crate::data_types::scalar_size_in_bytes;
use crate::data_types::{ScalarType, BIT};
use crate::errors::Result;

use std::ops::Not;

pub(super) fn add_u64(val1: u64, val2: u64, modulus: Option<u64>) -> u64 {
    match modulus {
        Some(m) => {
            let val = val1 as u128 + val2 as u128;
            (val % m as u128) as u64
        }
        None => val1.wrapping_add(val2),
    }
}

pub(super) fn add_u128(val1: u128, val2: u128, modulus: Option<u128>) -> u128 {
    let val = (val1).wrapping_add(val2);
    match modulus {
        Some(m) => val % m,
        None => val,
    }
}

pub(super) fn multiply_u64(val1: u64, val2: u64, modulus: Option<u64>) -> u64 {
    match modulus {
        Some(m) => {
            let val = val1 as u128 * val2 as u128;
            (val % m as u128) as u64
        }
        None => val1.wrapping_mul(val2),
    }
}

pub(super) fn multiply_u128(val1: u128, val2: u128, modulus: Option<u128>) -> u128 {
    let val = val1.wrapping_mul(val2);
    match modulus {
        Some(m) => val % m,
        None => val,
    }
}

pub fn add_vectors_u64(vec1: &[u64], vec2: &[u64], modulus: Option<u64>) -> Result<Vec<u64>> {
    if vec1.len() != vec2.len() {
        return Err(runtime_error!(
            "Vectors of different lengths can't be summed"
        ));
    }
    let mut res = vec![];
    for i in 0..vec1.len() {
        res.push(add_u64(vec1[i], vec2[i], modulus));
    }
    Ok(res)
}

pub fn add_vectors_u128(vec1: &[u128], vec2: &[u128], modulus: Option<u128>) -> Result<Vec<u128>> {
    if vec1.len() != vec2.len() {
        return Err(runtime_error!(
            "Vectors of different lengths can't be summed"
        ));
    }
    let mut res = vec![];
    for i in 0..vec1.len() {
        res.push(add_u128(vec1[i], vec2[i], modulus));
    }
    Ok(res)
}

pub fn sum_vector_u64(vec: &[u64], modulus: Option<u64>) -> u64 {
    let mut res = 0;
    for a in vec {
        res = add_u64(res, *a, modulus);
    }
    res
}

pub fn dot_vectors_u64(vec1: &[u64], vec2: &[u64], modulus: Option<u64>) -> Result<u64> {
    if vec1.len() != vec2.len() {
        return Err(runtime_error!(
            "Vectors of different lengths can't be summed"
        ));
    }
    let mut res = 0;
    for i in 0..vec1.len() {
        res = add_u64(res, multiply_u64(vec1[i], vec2[i], modulus), modulus);
    }
    Ok(res)
}

pub fn dot_vectors_u128(vec1: &[u128], vec2: &[u128], modulus: Option<u128>) -> Result<u128> {
    if vec1.len() != vec2.len() {
        return Err(runtime_error!(
            "Vectors of different lengths can't be summed"
        ));
    }
    let mut res = 0;
    for i in 0..vec1.len() {
        res = add_u128(res, multiply_u128(vec1[i], vec2[i], modulus), modulus);
    }
    Ok(res)
}

pub fn subtract_vectors_u64(vec1: &[u64], vec2: &[u64], modulus: Option<u64>) -> Result<Vec<u64>> {
    if vec1.len() != vec2.len() {
        return Err(runtime_error!(
            "Vectors of different lengths can't be subtracted"
        ));
    }
    let mut res = vec![];
    match modulus {
        Some(m) => {
            for i in 0..vec1.len() {
                let mut val = vec1[i] as u128 + (m - vec2[i] % m) as u128;
                val %= m as u128;
                res.push(val as u64);
            }
        }
        None => {
            for i in 0..vec1.len() {
                res.push(vec1[i].wrapping_sub(vec2[i]));
            }
        }
    }
    Ok(res)
}
pub fn subtract_vectors_u128(
    vec1: &[u128],
    vec2: &[u128],
    modulus: Option<u128>,
) -> Result<Vec<u128>> {
    if vec1.len() != vec2.len() {
        return Err(runtime_error!(
            "Vectors of different lengths can't be subtracted"
        ));
    }
    let mut res = vec![];
    for i in 0..vec1.len() {
        let val = u128::wrapping_sub(vec1[i], vec2[i]);
        res.push(val);
    }
    if let Some(m) = modulus {
        Ok(res.iter().map(|x| (x % m)).collect())
    } else {
        Ok(res)
    }
}
pub fn multiply_vectors_u64(vec1: &[u64], vec2: &[u64], modulus: Option<u64>) -> Result<Vec<u64>> {
    if vec1.len() != vec2.len() {
        return Err(runtime_error!(
            "Vectors of different lengths can't be multiplied"
        ));
    }
    let mut res = vec![];
    for i in 0..vec1.len() {
        res.push(multiply_u64(vec1[i], vec2[i], modulus));
    }
    Ok(res)
}
pub fn multiply_vectors_u128(
    vec1: &[u128],
    vec2: &[u128],
    modulus: Option<u128>,
) -> Result<Vec<u128>> {
    if vec1.len() != vec2.len() {
        return Err(runtime_error!(
            "Vectors of different lengths can't be multiplied"
        ));
    }
    let mut res = vec![];
    for i in 0..vec1.len() {
        res.push(multiply_u128(vec1[i], vec2[i], modulus));
    }
    Ok(res)
}

/// Converts any integer of standard type to u64. This is a generic version of the `as u64`
/// command that is not supported by any trait in the standard library and thus cannot
/// support generic input.
fn as_u64<T: TryInto<u64> + Not<Output = T> + Copy>(x: T) -> Result<u64> {
    match x.try_into() {
        Ok(ux) => Ok(ux), // x is a positive integer (try_into passed)
        Err(_) => {
            // x is a negative signed integer (try_into failed)
            // flip the bits of x including the sign bit
            // e.g. x = 10111000 -> 01000111
            let neg_x = !x;
            // now neg_x is a positive integer that should pass through try_into
            // e.g 01000111 -> 0..001000111
            let neg_x_u64 = match neg_x.try_into() {
                Ok(m) => m,
                Err(_) => {
                    return Err(runtime_error!("The integer of this size is not supported"));
                }
            };
            // flip the bits again
            Ok(!neg_x_u64)
        }
    }
}

/// Converts any integer of standard type to u128. This is a generic version of the `as u128`
/// command that is not supported by any trait in the standard library and thus cannot
/// support generic input.
fn as_u128<T: TryInto<u128> + Not<Output = T> + Copy>(x: T) -> Result<u128> {
    match x.try_into() {
        Ok(ux) => Ok(ux), // x is a positive integer (try_into passed)
        Err(_) => {
            // x is a negative signed integer (try_into failed)
            // flip the bits of x including the sign bit
            // e.g. x = 10111000 -> 01000111
            let neg_x = !x;
            //now neg_x is a positive integer that should pass through try_into
            // e.g 01000111 -> 0..001000111
            let neg_x_u128 = match neg_x.try_into() {
                Ok(m) => m,
                Err(_) => {
                    return Err(runtime_error!("The integer of this size is not supported"));
                }
            };
            // flip the bits again
            Ok(!neg_x_u128)
        }
    }
}

/// Bytes are presented in the little-endian order
pub fn vec_to_bytes<T: TryInto<u128> + Not<Output = T> + TryInto<u8> + Copy>(
    x: &[T],
    st: ScalarType,
) -> Result<Vec<u8>> {
    let mut x_bytes = vec![];
    match st {
        BIT => {
            let iter = x.chunks(8);
            for slice in iter.clone() {
                let mut x_byte = 0u8;
                for (i, bit) in slice.iter().enumerate() {
                    let bit_u8: u8 = match (*bit).try_into() {
                        Ok(b) => {
                            if b > 1 {
                                return Err(runtime_error!("Input is not a bit"));
                            } else {
                                b
                            }
                        }
                        Err(_) => {
                            return Err(runtime_error!("Input is not a bit"));
                        }
                    };
                    x_byte += bit_u8 << i;
                }
                x_bytes.push(x_byte);
            }
        }
        _ => {
            let byte_length = scalar_size_in_bytes(st) as usize;
            let x_u128s = vec_as_u128(x)?;
            for x_u128 in x_u128s {
                let x_elem_bytes = x_u128.to_le_bytes();
                for x_elem_byte in x_elem_bytes.iter().take(byte_length) {
                    x_bytes.push(*x_elem_byte);
                }
            }
        }
    }
    Ok(x_bytes)
}

/// Bytes are presented in the little-endian order
pub fn vec_u64_to_bytes<T: TryInto<u64> + Not<Output = T> + TryInto<u8> + Copy>(
    x: &[T],
    st: ScalarType,
) -> Result<Vec<u8>> {
    let mut x_bytes = vec![];
    match st {
        BIT => {
            let iter = x.chunks(8);
            for slice in iter.clone() {
                let mut x_byte = 0u8;
                for (i, bit) in slice.iter().enumerate() {
                    let bit_u8: u8 = match (*bit).try_into() {
                        Ok(b) => {
                            if b > 1 {
                                return Err(runtime_error!("Input is not a bit"));
                            } else {
                                b
                            }
                        }
                        Err(_) => {
                            return Err(runtime_error!("Input is not a bit"));
                        }
                    };
                    x_byte += bit_u8 << i;
                }
                x_bytes.push(x_byte);
            }
        }
        _ => {
            let byte_length = scalar_size_in_bytes(st) as usize;
            let x_u64s = vec_as_u64(x)?;
            for x_u64 in x_u64s {
                let x_elem_bytes = x_u64.to_le_bytes();
                for x_elem_byte in x_elem_bytes.iter().take(byte_length) {
                    x_bytes.push(*x_elem_byte);
                }
            }
        }
    }
    Ok(x_bytes)
}

pub fn vec_as_u64<T: TryInto<u64> + Not<Output = T> + Copy>(x: &[T]) -> Result<Vec<u64>> {
    x.iter().map(|x| as_u64(*x)).collect::<Result<_>>()
}

pub fn vec_as_u128<T: TryInto<u128> + Not<Output = T> + Copy>(x: &[T]) -> Result<Vec<u128>> {
    x.iter().map(|x| as_u128(*x)).collect::<Result<_>>()
}

/// Can return excess zero elements when ScalarType = BIT and
/// the number of bits in bytes is bigger than the actual number of packed bits
pub fn vec_u64_from_bytes(x: &[u8], st: ScalarType) -> Result<Vec<u64>> {
    let mut x_u64s = vec![];
    match st {
        BIT => {
            for byte in x {
                for i in 0..8 {
                    let bit = ((byte >> i) & 1) as u64;
                    x_u64s.push(bit);
                }
            }
        }
        _ => {
            let byte_length = scalar_size_in_bytes(st) as usize;
            // Whether to look at the leading bit when padding to 8 bytes.
            let pad_with_sign_bit = st.is_signed() && byte_length < 8;
            // E.g. 0xFFFFFFFFFFFF0000 if byte_length == 2.
            let sign_mask = match pad_with_sign_bit {
                false => 0,
                true => u64::MAX ^ ((1 << (byte_length * 8)) - 1),
            };
            if x.len() % byte_length != 0 {
                return Err(runtime_error!("Incompatible vector and scalar type"));
            }
            for x_slice in x.chunks_exact(byte_length) {
                let mut res = 0u64;
                for (i, xi) in x_slice.iter().enumerate() {
                    res += (*xi as u64) << (i * 8);
                }
                if pad_with_sign_bit {
                    let sign_bit = res >> (byte_length * 8 - 1);
                    if sign_bit == 1 {
                        res |= sign_mask;
                    }
                }
                x_u64s.push(res);
            }
        }
    }
    Ok(x_u64s)
}

pub fn vec_u128_from_bytes(x: &[u8], st: ScalarType) -> Result<Vec<u128>> {
    let mut x_u128s = vec![];
    match st {
        BIT => {
            for byte in x {
                for i in 0..8 {
                    let bit = ((byte >> i) & 1) as u128;
                    x_u128s.push(bit);
                }
            }
        }
        _ => {
            let byte_length = scalar_size_in_bytes(st) as usize;
            // Whether to look at the leading bit when padding to 8 bytes.
            let pad_with_sign_bit = st.is_signed() && byte_length < 16;
            // E.g. 0xFFFFFFFFFFFF0000 if byte_length == 2.
            let sign_mask = match pad_with_sign_bit {
                false => 0,
                true => u128::MAX ^ ((1 << (byte_length * 8)) - 1),
            };
            if x.len() % byte_length != 0 {
                return Err(runtime_error!("Incompatible vector and scalar type"));
            }
            for x_slice in x.chunks_exact(byte_length) {
                let mut res = 0;
                for (i, xi) in x_slice.iter().enumerate() {
                    res += (*xi as u128) << (i * 8);
                }
                if pad_with_sign_bit {
                    let sign_bit = res >> (byte_length * 8 - 1);
                    if sign_bit == 1 {
                        res |= sign_mask;
                    }
                }
                x_u128s.push(res);
            }
        }
    }
    Ok(x_u128s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::{
        BIT, INT128, INT16, INT32, INT64, INT8, UINT128, UINT16, UINT32, UINT64, UINT8,
    };

    #[test]
    fn test_as_u64_as_u128() {
        //correct input
        for x in [u8::MIN, u8::MAX, 0] {
            assert_eq!(x as u64, as_u64(x).unwrap());
            assert_eq!(x as u128, as_u128(x).unwrap());
        }
        for x in [i8::MIN, i8::MAX, 0] {
            assert_eq!(x as u64, as_u64(x).unwrap());
            assert_eq!(x as u128, as_u128(x).unwrap());
        }
        for x in [u16::MIN, u16::MAX, 0] {
            assert_eq!(x as u64, as_u64(x).unwrap());
            assert_eq!(x as u128, as_u128(x).unwrap());
        }
        for x in [i16::MIN, i16::MAX, 0] {
            assert_eq!(x as u64, as_u64(x).unwrap());
            assert_eq!(x as u128, as_u128(x).unwrap());
        }
        for x in [u32::MIN, u32::MAX, 0] {
            assert_eq!(x as u64, as_u64(x).unwrap());
            assert_eq!(x as u128, as_u128(x).unwrap());
        }
        for x in [i32::MIN, i32::MAX, 0] {
            assert_eq!(x as u64, as_u64(x).unwrap());
            assert_eq!(x as u128, as_u128(x).unwrap());
        }
        for x in [u64::MIN, u64::MAX, 0] {
            assert_eq!({ x }, as_u64(x).unwrap());
            assert_eq!(x as u128, as_u128(x).unwrap());
        }
        for x in [i64::MIN, i64::MAX, 0] {
            assert_eq!(x as u64, as_u64(x).unwrap());
            assert_eq!(x as u128, as_u128(x).unwrap());
        }
        for x in [i128::MIN, i128::MAX, 0] {
            assert_eq!(x as u128, as_u128(x).unwrap());
        }

        //malformed input
        assert!(as_u64(i128::MAX).is_err());
    }

    fn vec_to_bytes_helper<
        T: TryInto<u128> + TryInto<u64> + Not<Output = T> + TryInto<u8> + Copy,
    >(
        ints: &[T],
        bytes: &[u8],
        st: ScalarType,
    ) {
        let ints_bytes = vec_to_bytes(ints, st).unwrap();
        if st.size_in_bits() <= 64 {
            let from_u64_bytes = vec_u64_to_bytes(ints, st).unwrap();
            assert_eq!(ints_bytes, from_u64_bytes);
        }
        assert_eq!(ints_bytes, *bytes);
    }
    #[test]
    fn test_vec_to_bytes() {
        // correct input
        vec_to_bytes_helper(&[0; 1], &[0u8], BIT);
        vec_to_bytes_helper(&[1; 12], &[255u8, 15u8], BIT);
        vec_to_bytes_helper(&[0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1], &[78u8, 9u8], BIT);

        vec_to_bytes_helper(&[0u8, 0u8], &[0u8, 0u8], UINT8);
        vec_to_bytes_helper(&[u8::MAX, 100u8], &[u8::MAX, 100u8], UINT8);

        vec_to_bytes_helper(&[0u16, 0u16], &[0u8; 4], UINT16);
        vec_to_bytes_helper(&[u16::MAX, 100u16], &[u8::MAX, u8::MAX, 100u8, 0u8], UINT16);

        vec_to_bytes_helper(&[0u32, 0u32], &[0u8; 8], UINT32);
        vec_to_bytes_helper(&[u32::MAX, u32::MAX], &[u8::MAX; 8], UINT32);

        vec_to_bytes_helper(&[0u64, 0u64], &[0u8; 16], UINT64);
        vec_to_bytes_helper(&[u64::MAX, u64::MAX], &[u8::MAX; 16], UINT64);

        vec_to_bytes_helper(&[0u128, 0u128], &[0u8; 32], UINT128);
        vec_to_bytes_helper(&[u128::MAX, u128::MAX], &[u8::MAX; 32], UINT128);

        vec_to_bytes_helper(&[0i8, 0i8], &[0u8; 2], INT8);
        vec_to_bytes_helper(&[i8::MIN, i8::MAX], &[128u8, 127u8], INT8);
        vec_to_bytes_helper(&[-100i8, 120i8], &[156u8, 120u8], INT8);

        vec_to_bytes_helper(&[0i16, 0i16], &[0u8; 4], INT16);
        vec_to_bytes_helper(&[i16::MIN, i16::MAX], &[0u8, 128u8, 255u8, 127u8], INT16);
        vec_to_bytes_helper(&[-30_000i16, 20_000i16], &[208u8, 138u8, 32u8, 78u8], INT16);

        vec_to_bytes_helper(&[0i32, 0i32], &[0u8; 8], INT32);
        vec_to_bytes_helper(
            &[i32::MIN, i32::MAX],
            &[0u8, 0u8, 0u8, 128u8, 255u8, 255u8, 255u8, 127u8],
            INT32,
        );
        vec_to_bytes_helper(
            &[-2_000_000_002i32, 1_900_000_017i32],
            &[254u8, 107u8, 202u8, 136u8, 17u8, 179u8, 63u8, 113u8],
            INT32,
        );

        vec_to_bytes_helper(&[0i64, 0i64], &[0u8; 16], INT64);
        vec_to_bytes_helper(
            &[i64::MIN, i64::MAX],
            &[
                0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 128u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8,
                255u8, 127u8,
            ],
            INT64,
        );
        vec_to_bytes_helper(
            &[-9_000_000_000_000_000_808i64, 9_100_000_000_000_000_808i64],
            &[
                216u8, 252u8, 123u8, 29u8, 175u8, 147u8, 25u8, 131u8, 40u8, 3u8, 14u8, 64u8, 201u8,
                177u8, 73u8, 126u8,
            ],
            INT64,
        );

        vec_to_bytes_helper(&[0i128, 0i128], &[0u8; 32], INT128);
        vec_to_bytes_helper(
            &[i128::MIN, i128::MAX],
            &[
                0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 128u8,
                255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8,
                255u8, 255u8, 255u8, 127u8,
            ],
            INT128,
        );
        vec_to_bytes_helper(
            &[
                -170_000_000_000_000_000_000_000_000_000_000_000_808i128,
                170_000_000_000_000_000_000_000_000_000_000_000_808i128,
            ],
            &[
                216u8, 252u8, 255u8, 255u8, 95u8, 95u8, 200u8, 239u8, 150u8, 75u8, 129u8, 204u8,
                225u8, 48u8, 27u8, 128u8, 40u8, 3u8, 0u8, 0u8, 160u8, 160u8, 55u8, 16u8, 105u8,
                180u8, 126u8, 51u8, 30u8, 207u8, 228u8, 127u8,
            ],
            INT128,
        );

        // malformed input
        let e = vec_to_bytes(&[3, 0], BIT);
        assert!(e.is_err());
        let e = vec_to_bytes(&[-1, 0], BIT);
        assert!(e.is_err());
        let e = vec_u64_to_bytes(&[i128::MAX], UINT32);
        assert!(e.is_err());
    }

    fn vec_u64_from_bytes_helper(bytes: &[u8], ints: &[u64], st: ScalarType) {
        let bytes_ints = vec_u64_from_bytes(bytes, st).unwrap();
        assert_eq!(bytes_ints, *ints);
    }

    #[test]
    fn test_vec_u64_from_bytes() {
        // correct input
        vec_u64_from_bytes_helper(&[0u8], &[0; 8], BIT);
        vec_u64_from_bytes_helper(
            &[129u8, 2u8],
            &[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            BIT,
        );

        vec_u64_from_bytes_helper(&[0u8, 0u8], &[0u64, 0u64], UINT8);
        vec_u64_from_bytes_helper(&[255u8, 128u8], &[255u64, 128u64], UINT8);

        vec_u64_from_bytes_helper(&[0u8; 4], &[0u64; 2], UINT16);
        vec_u64_from_bytes_helper(
            &[255u8, 255u8, 10u8, 100u8],
            &[(1u64 << 16) - 1, 25610u64],
            UINT16,
        );

        vec_u64_from_bytes_helper(&[0u8; 8], &[0u64; 2], UINT32);
        vec_u64_from_bytes_helper(
            &[255u8, 255u8, 255u8, 255u8, 128u8, 119u8, 142u8, 6u8],
            &[(1u64 << 32) - 1u64, 110_000_000u64],
            UINT32,
        );

        vec_u64_from_bytes_helper(&[0u8; 16], &[0u64; 2], UINT64);
        vec_u64_from_bytes_helper(
            &[
                255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 1u8, 0u8, 75u8, 205u8,
                106u8, 204u8, 134u8, 1u8,
            ],
            &[u64::MAX, 110_000_000_000_000_001u64],
            UINT64,
        );

        vec_u64_from_bytes_helper(&[0u8, 0u8], &[0u64, 0u64], INT8);
        vec_u64_from_bytes_helper(&[255u8, 128u8], &[-1i8 as u64, -128i8 as u64], INT8);

        vec_u64_from_bytes_helper(&[0u8; 4], &[0u64; 2], INT16);
        vec_u64_from_bytes_helper(
            &[255u8, 255u8, 10u8, 100u8],
            &[-1i16 as u64, 25610u64],
            INT16,
        );
        vec_u64_from_bytes_helper(
            &[156u8, 254u8, 10u8, 100u8],
            &[-356i16 as u64, 25610u64],
            INT16,
        );

        vec_u64_from_bytes_helper(&[0u8; 8], &[0u64; 2], INT32);
        vec_u64_from_bytes_helper(
            &[255u8, 255u8, 255u8, 255u8, 128u8, 119u8, 142u8, 6u8],
            &[-1i32 as u64, 110_000_000u64],
            INT32,
        );
        vec_u64_from_bytes_helper(
            &[155u8, 200u8, 250u8, 185u8, 128u8, 119u8, 142u8, 6u8],
            &[-1_174_746_981i32 as u64, 110_000_000u64],
            INT32,
        );

        vec_u64_from_bytes_helper(&[0u8; 16], &[0u64; 2], INT64);
        vec_u64_from_bytes_helper(
            &[
                255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 1u8, 0u8, 75u8, 205u8,
                106u8, 204u8, 134u8, 1u8,
            ],
            &[u64::MAX, 110_000_000_000_000_001u64],
            INT64,
        );
        vec_u64_from_bytes_helper(
            &[
                206u8, 254u8, 155u8, 156u8, 205u8, 252u8, 200u8, 245u8, 1u8, 0u8, 75u8, 205u8,
                106u8, 204u8, 134u8, 1u8,
            ],
            &[17_710_683_494_660_439_758u64, 110_000_000_000_000_001u64],
            INT64,
        );

        // malformed input
        let e = vec_u64_from_bytes(&[0u8, 0u8, 0u8], UINT16);
        assert!(e.is_err());
    }
    fn vec_u128_from_bytes_helper(bytes: &[u8], ints: &[u128], st: ScalarType) {
        let bytes_ints = vec_u128_from_bytes(bytes, st).unwrap();
        assert_eq!(bytes_ints, *ints);
    }
    #[test]
    fn test_vec_u128_from_bytes() {
        // correct input
        vec_u128_from_bytes_helper(&[0u8], &[0; 8], BIT);
        vec_u128_from_bytes_helper(
            &[129u8, 2u8],
            &vec![1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            BIT,
        );

        vec_u128_from_bytes_helper(&[0u8, 0u8], &[0u128, 0u128], UINT8);
        vec_u128_from_bytes_helper(&[255u8, 128u8], &[255u128, 128u128], UINT8);

        vec_u128_from_bytes_helper(&[0u8; 4], &[0u128; 2], UINT16);
        vec_u128_from_bytes_helper(
            &[255u8, 255u8, 10u8, 100u8],
            &[(1u128 << 16) - 1, 25610u128],
            UINT16,
        );

        vec_u128_from_bytes_helper(&[0u8; 8], &[0u128; 2], UINT32);
        vec_u128_from_bytes_helper(
            &[255u8, 255u8, 255u8, 255u8, 128u8, 119u8, 142u8, 6u8],
            &[(1u128 << 32) - 1u128, 110_000_000u128],
            UINT32,
        );

        vec_u128_from_bytes_helper(&[0u8; 16], &[0u128; 2], UINT64);
        vec_u128_from_bytes_helper(
            &[
                255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 1u8, 0u8, 75u8, 205u8,
                106u8, 204u8, 134u8, 1u8,
            ],
            &[u64::MAX as u128, 110_000_000_000_000_001u128],
            UINT64,
        );
        vec_u128_from_bytes_helper(&[0u8; 32], &[0u128; 2], UINT128);
        vec_u128_from_bytes_helper(
            &[
                255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8,
                255u8, 255u8, 255u8, 255u8, 40u8, 3u8, 0u8, 0u8, 160u8, 160u8, 55u8, 16u8, 105u8,
                180u8, 126u8, 51u8, 30u8, 207u8, 228u8, 127u8,
            ],
            &[
                u128::MAX,
                170_000_000_000_000_000_000_000_000_000_000_000_808u128,
            ],
            UINT128,
        );

        vec_u128_from_bytes_helper(&[0u8, 0u8], &[0u128, 0u128], INT8);
        vec_u128_from_bytes_helper(&[255u8, 128u8], &[-1i8 as u128, -128i8 as u128], INT8);

        vec_u128_from_bytes_helper(&[0u8; 4], &[0u128; 2], INT16);
        vec_u128_from_bytes_helper(
            &[255u8, 255u8, 10u8, 100u8],
            &[-1i16 as u128, 25610u128],
            INT16,
        );
        vec_u128_from_bytes_helper(
            &[156u8, 254u8, 10u8, 100u8],
            &[-356i16 as u128, 25610u128],
            INT16,
        );

        vec_u128_from_bytes_helper(&[0u8; 8], &[0u128; 2], INT32);
        vec_u128_from_bytes_helper(
            &[255u8, 255u8, 255u8, 255u8, 128u8, 119u8, 142u8, 6u8],
            &[-1i32 as u128, 110_000_000u128],
            INT32,
        );
        vec_u128_from_bytes_helper(
            &[155u8, 200u8, 250u8, 185u8, 128u8, 119u8, 142u8, 6u8],
            &[-1_174_746_981i32 as u128, 110_000_000u128],
            INT32,
        );

        vec_u128_from_bytes_helper(&[0u8; 16], &[0u128; 2], INT64);
        vec_u128_from_bytes_helper(
            &[
                255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 1u8, 0u8, 75u8, 205u8,
                106u8, 204u8, 134u8, 1u8,
            ],
            &[-1i64 as u128, 110_000_000_000_000_001u128],
            INT64,
        );
        vec_u128_from_bytes_helper(
            &[
                206u8, 254u8, 155u8, 156u8, 205u8, 252u8, 200u8, 245u8, 1u8, 0u8, 75u8, 205u8,
                106u8, 204u8, 134u8, 1u8,
            ],
            &[
                -736_060_579_049_111_858_i64 as u128,
                110_000_000_000_000_001u128,
            ],
            INT64,
        );

        vec_u128_from_bytes_helper(&[0u8; 32], &[0u128; 2], INT128);
        vec_u128_from_bytes_helper(
            &[
                255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8,
                255u8, 255u8, 255u8, 255u8, 40u8, 3u8, 0u8, 0u8, 160u8, 160u8, 55u8, 16u8, 105u8,
                180u8, 126u8, 51u8, 30u8, 207u8, 228u8, 127u8,
            ],
            &[
                -1i128 as u128,
                170_000_000_000_000_000_000_000_000_000_000_000_808u128,
            ],
            INT128,
        );
        vec_u128_from_bytes_helper(
            &[
                216u8, 252u8, 255u8, 255u8, 95u8, 95u8, 200u8, 239u8, 150u8, 75u8, 129u8, 204u8,
                225u8, 48u8, 27u8, 128u8, 40u8, 3u8, 0u8, 0u8, 160u8, 160u8, 55u8, 16u8, 105u8,
                180u8, 126u8, 51u8, 30u8, 207u8, 228u8, 127u8,
            ],
            &[
                -170_000_000_000_000_000_000_000_000_000_000_000_808i128 as u128,
                170_000_000_000_000_000_000_000_000_000_000_000_808u128,
            ],
            INT128,
        );

        // malformed input
        let e = vec_u64_from_bytes(&[0u8, 0u8, 0u8], UINT16);
        assert!(e.is_err());
    }
}
