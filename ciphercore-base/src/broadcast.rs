use crate::data_types::{array_type, is_valid_shape, ArrayShape, Type};
use crate::errors::Result;
use std::cmp::max;

pub(super) fn broadcast_shapes(s1: ArrayShape, s2: ArrayShape) -> Result<ArrayShape> {
    let result_length = max(s1.len(), s2.len());
    let offset1 = result_length - s1.len();
    let offset2 = result_length - s2.len();
    let mut result = vec![];
    for i in 0..result_length {
        let mut value1 = 1;
        if i >= offset1 {
            value1 = s1[i - offset1];
        }
        let mut value2 = 1;
        if i >= offset2 {
            value2 = s2[i - offset2];
        }
        if value1 > 1 && value2 > 1 && value1 != value2 {
            return Err(runtime_error!(
                "Invalid broadcast: shapes {:?} and {:?}",
                s1,
                s2
            ));
        }
        result.push(max(value1, value2));
    }
    Ok(result)
}

fn broadcast_pair(t1: Type, t2: Type) -> Result<Type> {
    if t1.get_scalar_type() != t2.get_scalar_type() {
        return Err(runtime_error!("Scalar types mismatch"));
    }
    if t1.is_scalar() {
        return Ok(t2);
    }
    if t2.is_scalar() {
        return Ok(t1);
    }
    let scalar_type = t1.get_scalar_type();
    Ok(array_type(
        broadcast_shapes(t1.get_shape(), t2.get_shape())?,
        scalar_type,
    ))
}

pub(super) fn broadcast_arrays(element_types: Vec<Type>) -> Result<Type> {
    if element_types.is_empty() {
        return Err(runtime_error!("Can't broadcast the empty sequence"));
    }
    for x in &element_types {
        if !x.is_scalar() && !x.is_array() {
            return Err(runtime_error!("Can broadcast only scalars and arrays"));
        }
        if x.is_array() && !is_valid_shape(x.get_shape()) {
            return Err(runtime_error!("Invalid shape"));
        }
    }
    let mut result = element_types[0].clone();
    for item in element_types.iter().skip(1) {
        result = broadcast_pair(result, item.clone())?;
    }
    Ok(result)
}

pub fn index_to_number(index: &[u64], shape: &[u64]) -> u64 {
    let mut num = 0;
    for (i, d) in shape.iter().enumerate() {
        // mod d makes sure that input indices do not exceed dimensions given by the shape
        num = num * d + (index[i] % d);
    }
    num
}

pub fn number_to_index(num: u64, shape: &[u64]) -> Vec<u64> {
    let mut num_left = num;
    let mut index = vec![];
    let mut radix: u64 = shape.iter().product();
    for d in shape {
        radix /= d;
        let digit = num_left / radix;
        index.push(digit);
        num_left %= radix;
    }
    index
}

#[cfg(tests)]
mod tests {
    use super::*;
    use crate::data_types::{scalar_type, BIT, UINT8};

    #[test]
    fn test_malformed() {
        let e1 = broadcast_arrays(vec![]);
        assert!(e1.is_err());
        let e2 = broadcast_arrays(vec![tuple_type(vec![])]);
        assert!(e2.is_err());
        let e3 = broadcast_arrays(vec![scalar_type(BIT), tuple_type(vec![])]);
        assert!(e3.is_err());
        let e4 = broadcast_arrays(vec![scalar_type(BIT), array_type(vec![10, 10], UINT8)]);
        assert!(e4.is_err());
        let e5 = broadcast_arrays(vec![
            array_type(vec![10, 10], BIT),
            array_type(vec![10, 10], UINT8),
        ]);
        assert!(e5.is_err());
        let e6 = broadcast_arrays(vec![array_type(vec![], BIT)]);
        assert!(e6.is_err());
        let e7 = broadcast_arrays(vec![array_type(vec![7, 0, 3], BIT)]);
        assert!(e7.is_err());
        let e8 = broadcast_arrays(vec![scalar_type(BIT), array_type(vec![], BIT)]);
        assert!(e8.is_err());
        let e9 = broadcast_arrays(vec![scalar_type(BIT), array_type(vec![7, 0, 3], BIT)]);
        assert!(e9.is_err());
        let e10 = broadcast_arrays(vec![array_type(vec![3], BIT), array_type(vec![7], BIT)]);
        assert!(e10.is_err());
        let e11 = broadcast_arrays(vec![array_type(vec![7, 3], BIT), array_type(vec![7], BIT)]);
        assert!(e11.is_err());
    }

    #[test]
    fn test_valid() {
        assert_eq!(broadcast_arrays(vec![scalar_type(BIT)]), scalar_type(BIT));
        assert_eq!(
            broadcast_arrays(vec![scalar_type(UINT8)]),
            scalar_type(UINT8)
        );
        assert_eq!(
            broadcast_arrays(vec![scalar_type(BIT), scalar_type(BIT)]),
            scalar_type(BIT)
        );
        assert_eq!(
            broadcast_arrays(vec![
                scalar_type(UINT8),
                scalar_type(UINT8),
                scalar_type(UINT8)
            ]),
            scalar_type(UINT8)
        );
        assert_eq!(
            broadcast_arrays(vec![array_type(vec![10, 10], BIT)]),
            array_type(vec![10, 10], BIT)
        );
        assert_eq!(
            broadcast_arrays(vec![array_type(vec![10, 10], BIT), scalar_type(BIT)]),
            array_type(vec![10, 10], BIT)
        );
        assert_eq!(
            broadcast_arrays(vec![scalar_type(BIT), array_type(vec![10, 10], BIT)]),
            array_type(vec![10, 10], BIT)
        );
        assert_eq!(
            broadcast_arrays(vec![
                array_type(vec![10, 10], BIT),
                array_type(vec![10, 10], BIT)
            ]),
            array_type(vec![10, 10], BIT)
        );
        assert_eq!(
            broadcast_arrays(vec![
                array_type(vec![10, 10], BIT),
                array_type(vec![10, 1], BIT)
            ]),
            array_type(vec![10, 10], BIT)
        );
        assert_eq!(
            broadcast_arrays(vec![
                array_type(vec![1, 10], BIT),
                array_type(vec![10, 1], BIT)
            ]),
            array_type(vec![10, 10], BIT)
        );
        assert_eq!(
            broadcast_arrays(vec![
                array_type(vec![10], BIT),
                array_type(vec![10, 1], BIT)
            ]),
            array_type(vec![10, 10], BIT)
        );
        assert_eq!(
            broadcast_arrays(vec![array_type(vec![9], BIT), array_type(vec![3, 1], BIT)]),
            array_type(vec![3, 9], BIT)
        );
        assert_eq!(
            broadcast_arrays(vec![array_type(vec![9], BIT), array_type(vec![3, 9], BIT)]),
            array_type(vec![3, 9], BIT)
        );
        assert_eq!(
            broadcast_arrays(vec![array_type(vec![3, 9], BIT), array_type(vec![9], BIT)]),
            array_type(vec![3, 9], BIT)
        );
        assert_eq!(
            broadcast_arrays(vec![array_type(vec![3, 1], BIT), array_type(vec![9], BIT)]),
            array_type(vec![3, 9], BIT)
        );
    }
}
