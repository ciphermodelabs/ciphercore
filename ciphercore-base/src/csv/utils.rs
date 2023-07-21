use crate::data_types::Type;
use crate::errors::Result;
use crate::runtime_error;
use crate::typed_value::TypedValue;

pub struct Column {
    pub name: String,
    pub data: Vec<Option<String>>,
}

pub fn output_string_column(
    name: String,
    body: TypedValue,
    len: TypedValue,
    mask: &[u8],
) -> Result<Column> {
    let n = match len.t.get_shape().as_slice() {
        &[n] => n as usize,
        _ => return Err(runtime_error!("len type {:?}", len.t)),
    };
    let max_len = match body.t.get_shape().as_slice() {
        &[rows, max_len] => {
            if rows as usize != n {
                return Err(runtime_error!("len {:?} vs body {:?}", body.t, len.t));
            }
            max_len as usize
        }
        _ => return Err(runtime_error!("body type {:?}", body.t)),
    };
    if mask.len() != n {
        return Err(runtime_error!("mask.len() != n: {} vs {}", mask.len(), n));
    }
    let len = len.value.to_flattened_array_u16(len.t)?;
    let mut data = vec![];
    body.value.access_bytes(|bytes| {
        for i in 0..n {
            data.push(if mask[i] == 1 {
                let offset = i * max_len;
                let b = bytes[offset..offset + len[i] as usize].to_vec();
                Some(String::from_utf8(b).map_err(|e| runtime_error!("UTF8 error: {e}"))?)
            } else {
                None
            });
        }
        Ok(())
    })?;
    Ok(Column { name, data })
}

fn map_or_mask<T>(
    values: Vec<T>,
    mask: &[u8],
    f: impl Fn(T) -> String,
) -> Result<Vec<Option<String>>> {
    let mut data = vec![];
    for (value, &mask) in values.into_iter().zip(mask.iter()) {
        data.push(if mask == 0 { None } else { Some(f(value)) });
    }
    Ok(data)
}

pub fn output_int_column(name: String, value: TypedValue, mask: &[u8]) -> Result<Column> {
    validate_shapes(&value, mask)?;
    let data = if value.t.get_scalar_type().is_signed() {
        let values = value.value.to_flattened_array_i64(value.t)?;
        map_or_mask(values, mask, |v| v.to_string())?
    } else {
        let values = value.value.to_flattened_array_u64(value.t)?;
        map_or_mask(values, mask, |v| v.to_string())?
    };
    Ok(Column { name, data })
}

pub fn output_float_column(
    name: String,
    value: TypedValue,
    fractional_bits: usize,
    float_decimal_places: usize,
    mask: &[u8],
) -> Result<Column> {
    validate_shapes(&value, mask)?;
    let data = if value.t.get_scalar_type().is_signed() {
        let values = value.value.to_flattened_array_i64(value.t)?;
        map_or_mask(values, mask, |v| {
            format!(
                "{:.prec$}",
                v as f64 / (1 << fractional_bits) as f64,
                prec = float_decimal_places
            )
        })?
    } else {
        let values = value.value.to_flattened_array_u64(value.t)?;
        map_or_mask(values, mask, |v| {
            format!(
                "{:.prec$}",
                v as f64 / (1 << fractional_bits) as f64,
                prec = float_decimal_places
            )
        })?
    };
    Ok(Column { name, data })
}

pub fn output_bool_column(name: String, value: TypedValue, mask: &[u8]) -> Result<Column> {
    validate_shapes(&value, mask)?;
    let values = value.value.to_flattened_array_u8(value.t)?;
    let data = map_or_mask(values, mask, |v| {
        if v == 0 {
            "false".to_string()
        } else {
            "true".to_string()
        }
    })?;
    Ok(Column { name, data })
}

fn validate_shapes(value: &TypedValue, mask: &[u8]) -> Result<()> {
    let n = match value.t.get_shape().as_slice() {
        &[n] => n as usize,
        _ => return Err(runtime_error!("value type {:?}", value.t)),
    };
    if mask.len() != n {
        return Err(runtime_error!("mask.len() != n: {} vs {}", mask.len(), n));
    }
    Ok(())
}

pub fn output_table(columns: &Vec<Column>) -> Result<Vec<Vec<String>>> {
    if columns.is_empty() {
        return Err(runtime_error!("write_rows: empty columns list"));
    }
    let n = columns[0].data.len();
    let mut table = vec![];
    for i in 0..n {
        let row_iter = columns.iter().map(|column| &column.data[i]);
        if row_iter.clone().all(|cell| cell.is_none()) {
            // Skip empty rows.
            continue;
        }
        table.push(
            row_iter
                .map(|cell| match cell {
                    None => "".to_owned(),
                    Some(val) => val.clone(),
                })
                .collect::<Vec<String>>(),
        );
    }
    Ok(table)
}

pub fn write_table(
    mut columns: Vec<Column>,
    sort_columns: bool,
    sort_rows: bool,
) -> Result<Vec<u8>> {
    if sort_columns {
        columns.sort_by(|c1, c2| c1.name.cmp(&c2.name));
    }
    let mut table = output_table(&columns)?;
    if sort_rows {
        table.sort();
    }
    let header = columns.into_iter().map(|column| column.name).collect();
    write_to_csv(header, table)
}

fn write_to_csv(header: Vec<String>, table: Vec<Vec<String>>) -> Result<Vec<u8>> {
    let mut wtr = csv::Writer::from_writer(vec![]);
    wtr.write_record(header)?;
    for row in table {
        wtr.write_record(row)?;
    }
    wtr.into_inner()
        .map_err(|err| runtime_error!("Error: {}", err))
}

pub fn unpack_named_tuple(value: TypedValue) -> Result<Vec<(String, TypedValue)>> {
    let name_and_types = match value.t.clone() {
        Type::NamedTuple(elements) => elements,
        t => return Err(runtime_error!("Expected NamedTuple, got {:?}", t)),
    };
    let values = value.value.to_vector()?;
    if name_and_types.len() != values.len() {
        return Err(runtime_error!("Inconsistent data"));
    }
    let mut result = vec![];
    for ((name, t), value) in name_and_types.into_iter().zip(values.into_iter()) {
        result.push((
            name,
            TypedValue {
                value,
                t: t.as_ref().clone(),
                name: None,
            },
        ));
    }
    Ok(result)
}

pub fn unpack_tuple(value: TypedValue) -> Result<Vec<TypedValue>> {
    let types = match value.t.clone() {
        Type::Tuple(elements) => elements,
        t => return Err(runtime_error!("Expected Tuple, got {:?}", t)),
    };
    let values = value.value.to_vector()?;
    if types.len() != values.len() {
        return Err(runtime_error!("Inconsistent data"));
    }
    let mut result = vec![];
    for (t, value) in types.into_iter().zip(values.into_iter()) {
        result.push(TypedValue {
            value,
            t: t.as_ref().clone(),
            name: None,
        });
    }
    Ok(result)
}

pub fn unpack_pair(value: TypedValue) -> Result<(TypedValue, TypedValue)> {
    let values = unpack_tuple(value)?;
    match values.as_slice() {
        [first, second] => Ok((first.clone(), second.clone())),
        _ => Err(runtime_error!("Expected tuple of size 2")),
    }
}

pub fn extract_data_mask_pair(value: TypedValue) -> Result<(TypedValue, Vec<u8>)> {
    let (data, mask) = unpack_pair(value)?;
    let mask = mask.value.to_flattened_array_u8(mask.t)?;
    Ok((data, mask))
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::{
        csv::test_utils::assert_table_eq,
        data_types::{BIT, INT64, UINT8},
        typed_value_operations::TypedValueArrayOperations,
    };

    #[test]
    fn test_output_csv() -> Result<()> {
        let c1 = output_int_column(
            "d".into(),
            TypedValue::from_ndarray(array![1, 2, 3, 4].into_dyn(), INT64)?,
            &[1, 1, 1, 0],
        )?;
        let c2 = output_float_column(
            "c".into(),
            TypedValue::from_ndarray(array![128, 256, 512, 1024].into_dyn(), INT64)?,
            10,
            3,
            &[0, 1, 1, 1],
        )?;
        let c3 = output_bool_column(
            "b".into(),
            TypedValue::from_ndarray(array![1, 0, 0, 1].into_dyn(), BIT)?,
            &[1, 0, 1, 1],
        )?;
        let c4 = output_string_column(
            "a".into(),
            TypedValue::from_ndarray(
                array![[65, 66, 0], [70, 0, 0], [75, 76, 77], [80, 81, 0]].into_dyn(),
                UINT8,
            )?,
            TypedValue::from_ndarray(array![2, 1, 3, 2].into_dyn(), INT64)?,
            &[1, 1, 1, 1],
        )?;
        assert_table_eq(
            write_table(vec![c1, c2, c3, c4], false, false)?,
            vec!["d", "c", "b", "a"],
            vec![
                vec!["1", "", "true", "AB"],
                vec!["2", "0.250", "", "F"],
                vec!["3", "0.500", "false", "KLM"],
                vec!["", "1.000", "true", "PQ"],
            ],
        )
    }

    #[tokio::test]
    async fn test_output_sorted_csv() -> Result<()> {
        let c1 = output_int_column(
            "salary".into(),
            TypedValue::from_ndarray(array![1000, 2000, 3000, 4000].into_dyn(), INT64)?,
            &[1, 1, 1, 1],
        )?;
        let c2 = output_int_column(
            "age".into(),
            TypedValue::from_ndarray(array![10, 20, 100, 30].into_dyn(), INT64)?,
            &[1, 1, 1, 0],
        )?;
        assert_table_eq(
            write_table(vec![c1, c2], true, true)?,
            vec!["age", "salary"],
            vec![
                vec!["", "4000"],
                vec!["10", "1000"],
                vec!["100", "3000"],
                vec!["20", "2000"],
            ],
        )
    }
}
