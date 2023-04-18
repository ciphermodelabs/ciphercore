use crate::{
    data_types::{
        array_type, get_named_types, get_types_vector, tuple_type, ArrayShape, ScalarType, Type,
        BIT,
    },
    errors::Result,
    type_inference::NULL_HEADER,
};
use std::collections::HashMap;

#[derive(Clone)]
pub(crate) struct ColumnType {
    mask_t: Option<Type>,
    data_t: Type,
}

impl ColumnType {
    pub(crate) fn new(column_t: Type, has_column_mask: bool, header: &str) -> Result<Self> {
        let (mask_t, data_t) = if has_column_mask && header != NULL_HEADER {
            if !column_t.is_tuple() {
                return Err(runtime_error!(
                    "Column should contain a tuple, got: {column_t:?}"
                ));
            }
            match get_types_vector(column_t.clone())?.as_slice() {
                [mask_t, data_t] => (Some((**mask_t).clone()), (**data_t).clone()),
                _ => {
                    return Err(runtime_error!(
                        "Column should contain a tuple with two arrays, given {}",
                        column_t
                    ));
                }
            }
        } else {
            (None, column_t.clone())
        };
        if !data_t.is_array() {
            if header == NULL_HEADER {
                return Err(runtime_error!(
                    "Null column should be a binary array, got {data_t}"
                ));
            }
            return Err(runtime_error!(
                "Column should have an array with data, got: {column_t:?}"
            ));
        }
        if let Some(mask_t_resolved) = mask_t.clone() {
            let num_col_entries = data_t.get_shape()[0];
            ColumnType::check_column_mask_type(mask_t_resolved, num_col_entries, header)?;
        }
        if header == NULL_HEADER && data_t.get_scalar_type() != BIT {
            return Err(runtime_error!(
                "Null column should be binary, got {data_t:?}"
            ));
        }
        Ok(ColumnType { mask_t, data_t })
    }

    fn check_column_mask_type(
        binary_mask_t: Type,
        expected_num_entries: u64,
        header: &str,
    ) -> Result<()> {
        if binary_mask_t.get_scalar_type() != BIT {
            return Err(runtime_error!(
                "{header} column mask should be binary, got {binary_mask_t:?}"
            ));
        }
        if binary_mask_t.get_shape() != vec![expected_num_entries] {
            return Err(runtime_error!(
                "{header} column mask should have shape {:?}",
                vec![expected_num_entries]
            ));
        }
        Ok(())
    }

    pub(crate) fn get_num_entries(&self) -> u64 {
        self.get_data_shape()[0]
    }

    pub(crate) fn clone_with_number_of_entries(&self, new_num_entries: u64) -> ColumnType {
        let mut shape = self.get_data_shape();
        shape[0] = new_num_entries;
        let st = self.get_scalar_type();
        let data_t = array_type(shape, st);
        let mut mask_t = None;
        if self.mask_t.is_some() {
            mask_t = Some(array_type(vec![new_num_entries], BIT));
        }
        ColumnType { mask_t, data_t }
    }

    pub(crate) fn get_data_shape(&self) -> ArrayShape {
        self.data_t.get_shape()
    }

    pub(crate) fn get_scalar_type(&self) -> ScalarType {
        self.data_t.get_scalar_type()
    }

    pub(crate) fn get_data_type(&self) -> Type {
        self.data_t.clone()
    }

    pub(crate) fn get_mask_type(&self) -> Result<Type> {
        if let Some(mask_t) = self.mask_t.clone() {
            return Ok(mask_t);
        }
        Err(runtime_error!("Column has no mask"))
    }

    pub(crate) fn get_row_size_in_elements(&self) -> usize {
        self.get_data_shape().iter().skip(1).product::<u64>() as usize
    }

    #[allow(dead_code)]
    pub(crate) fn add_mask(&mut self) {
        self.mask_t = Some(array_type(vec![self.get_num_entries()], BIT));
    }
}

impl From<ColumnType> for Type {
    fn from(column_t: ColumnType) -> Self {
        if let Some(mask_t) = column_t.mask_t {
            return tuple_type(vec![mask_t, column_t.data_t]);
        }
        column_t.data_t
    }
}

pub(crate) fn check_table_and_extract_column_types(
    t: Type,
    has_null_column: bool,
    has_column_masks: bool,
) -> Result<(HashMap<String, ColumnType>, u64)> {
    let v = get_named_types(&t)?;

    if has_null_column && v.len() < 2 {
        return Err(runtime_error!("Named tuple should contain at least two columns, one of which must be the null column. Got: {v:?}"));
    }
    if !has_null_column && v.is_empty() {
        return Err(runtime_error!(
            "Named tuple should contain at least one column."
        ));
    }
    let mut num_rows = 0;
    let mut contains_null = false;
    let mut all_headers: HashMap<String, ColumnType> = HashMap::new();
    for (h, sub_t) in v {
        let column_type = ColumnType::new((**sub_t).clone(), has_column_masks, h)?;
        let num_entries = column_type.get_num_entries();
        if num_rows == 0 {
            num_rows = num_entries
        }
        if num_rows != num_entries {
            return Err(runtime_error!(
                "Number of entries should be the same in each column: {num_rows} vs {num_entries}"
            ));
        }
        if h == NULL_HEADER && has_null_column {
            let null_shape = column_type.get_data_shape();
            let expected_shape = vec![num_rows];
            if null_shape != expected_shape {
                return Err(runtime_error!(
                    "Null column has shape {null_shape:?}, should be {expected_shape:?}"
                ));
            }
            contains_null = true;
        }
        all_headers.insert(h.clone(), column_type);
    }
    if !contains_null && has_null_column {
        return Err(runtime_error!("Named tuple should contain the null column"));
    }
    Ok((all_headers, num_rows))
}
