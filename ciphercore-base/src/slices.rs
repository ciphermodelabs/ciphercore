use crate::data_types::ArrayShape;
use crate::errors::Result;
use crate::graphs::{Slice, SliceElement};

use std::iter::repeat;

pub(super) fn get_slice_shape(shape: ArrayShape, slice: Slice) -> Result<ArrayShape> {
    let clean_slice = get_clean_slice(shape.clone(), slice)?;
    let mut result_shape = vec![];
    for i in 0..shape.len() {
        if i < clean_slice.len() {
            if let Some(s) = get_slice_shape_1d(shape[i], clean_slice[i].clone())? {
                result_shape.push(s);
            }
        } else {
            result_shape.push(shape[i]);
        }
    }
    Ok(result_shape)
}

/// Assumes that `slice` is a correct slice, in particular that
/// `get_slice_shape(shape, slice)` does not return an error.
pub fn slice_index(shape: ArrayShape, slice: Slice, index: ArrayShape) -> Result<ArrayShape> {
    let clean_slice = get_clean_slice(shape.clone(), slice)?;
    let mut result_index: Vec<u64> = vec![];
    let mut j = 0;
    for i in 0..shape.len() {
        if i < clean_slice.len() {
            match clean_slice[i] {
                SliceElement::SingleIndex(ind) => {
                    let real_ind = if ind >= 0 { ind } else { ind + shape[i] as i64 };
                    if real_ind < 0 {
                        panic!("Should not be here!");
                    }
                    result_index.push(real_ind as u64);
                }
                SliceElement::SubArray(_, _, _) => {
                    if j >= index.len() {
                        return Err(runtime_error!("Index is too short"));
                    }
                    result_index.push(slice_1d_index(shape[i], clean_slice[i].clone(), index[j])?);
                    j += 1;
                }
                SliceElement::Ellipsis => {
                    panic!("Should not be here!");
                }
            }
        } else {
            if j >= index.len() {
                return Err(runtime_error!("Index is too short"));
            }
            result_index.push(index[j]);
            j += 1;
        }
    }
    if j == 0 && index.len() == 1 && index[0] == 0 {
        return Ok(result_index);
    }
    if j != index.len() {
        return Err(runtime_error!("Index is too long"));
    }
    Ok(result_index)
}

fn slice_1d_index(dimension: u64, slice_element: SliceElement, index: u64) -> Result<u64> {
    let (begin, _, step) = normalize_subarray(dimension, slice_element)?;
    let result = begin + step * (index as i64);
    if result < 0 {
        Err(runtime_error!("Negative index"))
    } else {
        Ok(result as u64)
    }
}

#[doc(hidden)]
pub fn get_clean_slice(shape: ArrayShape, slice: Slice) -> Result<Slice> {
    let mut num_ellipsis = 0;
    for x in &slice {
        if *x == SliceElement::Ellipsis {
            num_ellipsis += 1;
        }
    }
    if num_ellipsis > 1 {
        return Err(runtime_error!("Multiple Ellipsis in the slice"));
    }
    let mut clean_slice = vec![];
    for x in &slice {
        if *x == SliceElement::Ellipsis {
            let padding = shape.len() as i64 - slice.len() as i64 + 1;
            if padding < 0 {
                return Err(runtime_error!(
                    "Ellipsis corresponds to a negative number of entries"
                ));
            }
            clean_slice
                .extend(repeat(SliceElement::SubArray(None, None, None)).take(padding as usize));
        } else {
            clean_slice.push(x.clone());
        }
    }
    if clean_slice.len() > shape.len() {
        return Err(runtime_error!("Slice is too long"));
    }
    Ok(clean_slice)
}

fn normalize_subarray(dimension: u64, slice_element: SliceElement) -> Result<(i64, i64, i64)> {
    if let SliceElement::SubArray(begin, end, step) = slice_element {
        let step = step.unwrap_or(1);
        if step == 0 {
            return Err(runtime_error!("Slice step can't be zero"));
        }
        let mut begin = begin.unwrap_or(if step > 0 { 0 } else { dimension as i64 - 1 });
        if begin < 0 {
            begin += dimension as i64;
        }
        let end = end
            .map(|x| if x >= 0 { x } else { x + dimension as i64 })
            .unwrap_or(if step > 0 { dimension as i64 } else { -1 });
        Ok((begin, end, step))
    } else {
        panic!("Should not be here!");
    }
}

fn get_slice_shape_1d(dimension: u64, slice_element: SliceElement) -> Result<Option<u64>> {
    match slice_element {
        SliceElement::SingleIndex(mut ind) => {
            if ind < 0 {
                ind += dimension as i64;
            }
            if ind < 0 || ind >= dimension as i64 {
                Err(runtime_error!("Slice is out of bounds (SingleIndex)"))
            } else {
                Ok(None)
            }
        }
        SliceElement::SubArray(_, _, _) => {
            let (begin, end, step) = normalize_subarray(dimension, slice_element)?;
            let mut current = begin;
            let mut counter = 0;
            loop {
                if (step > 0 && current >= end) || (step < 0 && current <= end) {
                    break;
                }
                if current < 0 || current >= dimension as i64 {
                    return Err(runtime_error!(
                        "Slicing index is out of bounds: {} not in [{}, {})",
                        current,
                        0,
                        dimension
                    ));
                }
                counter += 1;
                current += step;
            }
            if counter == 0 {
                return Err(runtime_error!("Empty slice"));
            }
            Ok(Some(counter))
        }
        SliceElement::Ellipsis => {
            panic!("Should not be here!");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphs::SliceElement;

    #[test]
    fn test_slicing() {
        assert_eq!(
            get_slice_shape(vec![10, 20, 30], vec![]).unwrap(),
            vec![10, 20, 30]
        );
        assert_eq!(
            get_slice_shape(vec![10, 20, 30], vec![SliceElement::SingleIndex(0)]).unwrap(),
            vec![20, 30]
        );
        assert_eq!(
            get_slice_shape(vec![10, 20, 30], vec![SliceElement::SingleIndex(5)]).unwrap(),
            vec![20, 30]
        );
        assert_eq!(
            get_slice_shape(vec![10, 20, 30], vec![SliceElement::SingleIndex(9)]).unwrap(),
            vec![20, 30]
        );
        assert_eq!(
            get_slice_shape(vec![10, 20, 30], vec![SliceElement::SingleIndex(-10)]).unwrap(),
            vec![20, 30]
        );
        assert!(get_slice_shape(vec![10, 20, 30], vec![SliceElement::SingleIndex(-11)]).is_err());
        assert!(get_slice_shape(vec![10, 20, 30], vec![SliceElement::SingleIndex(-12)]).is_err());
        assert!(get_slice_shape(vec![10, 20, 30], vec![SliceElement::SingleIndex(-100)]).is_err());
        assert!(get_slice_shape(vec![10, 20, 30], vec![SliceElement::SingleIndex(10)]).is_err());
        assert!(get_slice_shape(vec![10, 20, 30], vec![SliceElement::SingleIndex(11)]).is_err());
        assert!(get_slice_shape(vec![10, 20, 30], vec![SliceElement::SingleIndex(100)]).is_err());
        assert_eq!(
            get_slice_shape(
                vec![10, 20, 30],
                vec![SliceElement::SubArray(None, None, None)]
            )
            .unwrap(),
            vec![10, 20, 30]
        );
        assert_eq!(
            get_slice_shape(
                vec![10, 20, 30],
                vec![SliceElement::SubArray(Some(3), None, None)]
            )
            .unwrap(),
            vec![7, 20, 30]
        );
        assert_eq!(
            get_slice_shape(
                vec![10, 20, 30],
                vec![SliceElement::SubArray(Some(9), None, None)]
            )
            .unwrap(),
            vec![1, 20, 30]
        );
        assert_eq!(
            get_slice_shape(
                vec![10, 20, 30],
                vec![SliceElement::SubArray(Some(-1), None, None)]
            )
            .unwrap(),
            vec![1, 20, 30]
        );
        assert_eq!(
            get_slice_shape(
                vec![10, 20, 30],
                vec![SliceElement::SubArray(Some(-10), None, None)]
            )
            .unwrap(),
            vec![10, 20, 30]
        );
        assert!(get_slice_shape(
            vec![10, 20, 30],
            vec![SliceElement::SubArray(Some(10), None, None)]
        )
        .is_err());
        assert!(get_slice_shape(
            vec![10, 20, 30],
            vec![SliceElement::SubArray(Some(-11), None, None)]
        )
        .is_err());
        assert_eq!(
            get_slice_shape(
                vec![10, 20, 30],
                vec![SliceElement::SubArray(Some(3), Some(13), Some(5))]
            )
            .unwrap(),
            vec![2, 20, 30]
        );
        assert_eq!(
            get_slice_shape(
                vec![10, 20, 30],
                vec![SliceElement::SubArray(Some(9), Some(13), Some(5))]
            )
            .unwrap(),
            vec![1, 20, 30]
        );
        assert_eq!(
            get_slice_shape(
                vec![10, 20, 30],
                vec![SliceElement::SubArray(None, None, Some(-1))]
            )
            .unwrap(),
            vec![10, 20, 30]
        );
        assert_eq!(
            get_slice_shape(
                vec![10, 20, 30],
                vec![SliceElement::SubArray(Some(3), None, Some(-1))]
            )
            .unwrap(),
            vec![4, 20, 30]
        );
        assert!(get_slice_shape(
            vec![10, 20, 30],
            vec![SliceElement::SubArray(None, None, Some(0))]
        )
        .is_err());
        assert!(get_slice_shape(
            vec![10, 20, 30],
            vec![
                SliceElement::SubArray(None, None, None),
                SliceElement::SubArray(None, None, None),
                SliceElement::SubArray(None, None, None),
                SliceElement::SubArray(None, None, None)
            ]
        )
        .is_err());
        assert_eq!(
            get_slice_shape(
                vec![10, 20, 30],
                vec![SliceElement::SubArray(Some(3), Some(-1), None)]
            )
            .unwrap(),
            vec![6, 20, 30]
        );
        assert_eq!(
            get_slice_shape(
                vec![10, 20, 30],
                vec![
                    SliceElement::SubArray(Some(3), Some(-1), None),
                    SliceElement::SingleIndex(3)
                ]
            )
            .unwrap(),
            vec![6, 30]
        );
        assert_eq!(
            get_slice_shape(
                vec![10, 20, 30],
                vec![
                    SliceElement::SingleIndex(3),
                    SliceElement::SubArray(Some(3), Some(-1), None)
                ]
            )
            .unwrap(),
            vec![16, 30]
        );
        assert_eq!(
            get_slice_shape(vec![10], vec![SliceElement::SingleIndex(3)]).unwrap(),
            Vec::<u64>::new()
        );
        assert_eq!(
            get_slice_shape(vec![10, 20, 30], vec![SliceElement::Ellipsis]).unwrap(),
            vec![10, 20, 30]
        );
        assert!(get_slice_shape(
            vec![10, 20, 30],
            vec![SliceElement::Ellipsis, SliceElement::Ellipsis]
        )
        .is_err());
        assert_eq!(
            get_slice_shape(
                vec![10, 20, 30],
                vec![SliceElement::Ellipsis, SliceElement::SingleIndex(3)]
            )
            .unwrap(),
            vec![10, 20]
        );
        assert_eq!(
            get_slice_shape(
                vec![10, 20, 30],
                vec![SliceElement::SingleIndex(3), SliceElement::Ellipsis]
            )
            .unwrap(),
            vec![20, 30]
        );
        assert_eq!(
            get_slice_shape(
                vec![10, 20, 30],
                vec![
                    SliceElement::SingleIndex(3),
                    SliceElement::Ellipsis,
                    SliceElement::SubArray(Some(0), Some(5), Some(3)),
                    SliceElement::SubArray(Some(0), Some(5), Some(4))
                ]
            )
            .unwrap(),
            vec![2, 2]
        );
        assert!(get_slice_shape(
            vec![10, 20, 30],
            vec![
                SliceElement::SingleIndex(3),
                SliceElement::Ellipsis,
                SliceElement::SingleIndex(7),
                SliceElement::SubArray(Some(0), Some(5), Some(3)),
                SliceElement::SubArray(Some(0), Some(5), Some(4))
            ]
        )
        .is_err());
    }

    #[test]
    fn test_slice_index() {
        assert_eq!(
            slice_index(
                vec![10, 20, 30],
                vec![
                    SliceElement::SingleIndex(3),
                    SliceElement::Ellipsis,
                    SliceElement::SubArray(Some(0), Some(5), Some(3)),
                    SliceElement::SubArray(Some(0), Some(5), Some(4))
                ],
                vec![1, 1]
            )
            .unwrap(),
            vec![3, 3, 4]
        );
        assert!(slice_index(
            vec![10, 20, 30],
            vec![
                SliceElement::SingleIndex(3),
                SliceElement::Ellipsis,
                SliceElement::SubArray(Some(0), Some(5), Some(3)),
                SliceElement::SubArray(Some(0), Some(5), Some(4))
            ],
            vec![]
        )
        .is_err());
        assert!(slice_index(vec![10, 20, 30], vec![], vec![]).is_err());
        assert!(slice_index(vec![10, 20, 30], vec![], vec![0, 1, 2, 3]).is_err());
        assert!(slice_index(
            vec![10, 20, 30],
            vec![SliceElement::SubArray(None, None, Some(-1))],
            vec![100, 0, 0]
        )
        .is_err());
        assert_eq!(
            slice_index(vec![10, 20, 30], vec![], vec![0, 1, 2]).unwrap(),
            vec![0, 1, 2]
        );
        assert!(slice_index(
            vec![10, 20, 30],
            vec![
                SliceElement::SingleIndex(1),
                SliceElement::SingleIndex(2),
                SliceElement::SingleIndex(3)
            ],
            vec![123]
        )
        .is_err());
        assert_eq!(
            slice_index(
                vec![10, 20, 30],
                vec![
                    SliceElement::SingleIndex(1),
                    SliceElement::SingleIndex(2),
                    SliceElement::SingleIndex(3)
                ],
                vec![0]
            )
            .unwrap(),
            vec![1, 2, 3]
        );
    }
}
