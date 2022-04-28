use std::ops::Not;

use crate::bytes::{add_vectors_u64, subtract_vectors_u64};
use crate::data_types::{array_type, scalar_size_in_bytes, ScalarType, UINT8};
use crate::data_values::Value;
use crate::errors::Result;
use crate::random::PRNG;

/// Utility function for preparing secret-shared data.
///
/// TODO: in the future, we need to make secret-sharing more generic.
///
/// # Arguments
///
/// * `prng` - PRNG object (for randomness).
/// * `data` - array to secret-share.
/// * `scalar_type` - scalar type to use.
///
/// # Returns
///
/// Vector of shares.
pub fn share_vector<T: TryInto<u64> + Not<Output = T> + TryInto<u8> + Copy>(
    prng: &mut PRNG,
    data: &[T],
    scalar_type: ScalarType,
) -> Result<Vec<Value>> {
    let n = data.len();
    let n_bytes = n * scalar_size_in_bytes(scalar_type.clone()) as usize;

    // first share (r0) is pseudo-random
    let r0_bytes = prng.get_random_bytes(n_bytes)?;
    let r0 = Value::from_bytes(r0_bytes)
        .to_flattened_array_u64(array_type(vec![n as u64], scalar_type.clone()))?;
    // second share (r1) is pseudo-random
    let r1_bytes = prng.get_random_bytes(n_bytes)?;
    let r1 = Value::from_bytes(r1_bytes)
        .to_flattened_array_u64(array_type(vec![n as u64], scalar_type.clone()))?;
    // third share (r2) is r2 = data - (r0 + r1)
    let r0r1 = add_vectors_u64(&r0, &r1, scalar_type.get_modulus())?;
    let data_u64 = Value::from_flattened_array(data, scalar_type.clone())?
        .to_flattened_array_u64(array_type(vec![n as u64], scalar_type.clone()))?;
    let r2 = subtract_vectors_u64(&data_u64, &r0r1, scalar_type.get_modulus())?;

    let shares = vec![
        Value::from_flattened_array(&r0, scalar_type.clone())?,
        Value::from_flattened_array(&r1, scalar_type.clone())?,
        Value::from_flattened_array(&r2, scalar_type)?,
    ];

    let mut garbage = vec![];
    for _ in 0..3 {
        garbage.push(Value::from_flattened_array(
            &prng.get_random_bytes(n_bytes)?,
            UINT8,
        )?);
    }

    // convert the shares to a value
    Ok(vec![
        Value::from_vector(vec![
            shares[0].clone(),
            shares[1].clone(),
            garbage[2].clone(),
        ]),
        Value::from_vector(vec![
            garbage[0].clone(),
            shares[1].clone(),
            shares[2].clone(),
        ]),
        Value::from_vector(vec![
            shares[0].clone(),
            garbage[1].clone(),
            shares[2].clone(),
        ]),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::UINT32;

    #[test]
    fn test_simple_share() {
        || -> Result<()> {
            let data = vec![12, 34, 56];
            let mut prng = PRNG::new(None)?;
            let shares = share_vector(&mut prng, &data, UINT32)?;
            let shares0 = shares[0].to_vector()?;
            let shares1 = shares[1].to_vector()?;
            let shares2 = shares[2].to_vector()?;
            assert_eq!(shares0[0], shares2[0]);
            assert_eq!(shares0[1], shares1[1]);
            assert_eq!(shares1[2], shares2[2]);
            let t = array_type(vec![3], UINT32);
            let v0 = shares0[0].to_flattened_array_u64(t.clone())?;
            let v1 = shares1[1].to_flattened_array_u64(t.clone())?;
            let v2 = shares2[2].to_flattened_array_u64(t.clone())?;
            let new_data = add_vectors_u64(
                &add_vectors_u64(&v0, &v1, UINT32.get_modulus())?,
                &v2,
                UINT32.get_modulus(),
            )?;
            assert_eq!(new_data, data);
            Ok(())
        }()
        .unwrap();
    }
}
