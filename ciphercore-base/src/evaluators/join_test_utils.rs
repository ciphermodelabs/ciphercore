use std::collections::HashMap;

use crate::{
    data_types::{array_type, named_tuple_type, tuple_type, ScalarType, Type, BIT},
    data_values::Value,
    errors::Result,
    graphs::JoinType,
    join_utils::ColumnType,
    type_inference::NULL_HEADER,
};

// Helper structures and functions for testing
#[derive(Clone)]
pub(crate) struct ColumnInfo {
    pub header: String,
    t: ColumnType,
    mask: Option<Vec<u64>>,
    data: Vec<u64>,
}
impl ColumnInfo {
    pub(crate) fn get_value(&self) -> Result<Value> {
        let data_value = Value::from_flattened_array(&self.data, self.t.get_scalar_type())?;
        if let Some(mask_arr) = self.mask.clone() {
            let mask_value = Value::from_flattened_array(&mask_arr, BIT)?;
            return Ok(Value::from_vector(vec![mask_value, data_value]));
        }
        Ok(data_value)
    }

    fn has_mask(&self) -> bool {
        self.mask.is_some()
    }
}

pub(crate) fn column_info(
    header: &str,
    shape: &[u64],
    st: ScalarType,
    data: &[u64],
) -> Result<ColumnInfo> {
    let t = ColumnType::new(array_type(shape.to_vec(), st), false, header)?;
    Ok(ColumnInfo {
        header: header.to_owned(),
        t,
        mask: None,
        data: data.to_vec(),
    })
}

pub(crate) fn column_info_with_mask(
    header: &str,
    shape: &[u64],
    st: ScalarType,
    mask: Option<&[u64]>,
    data: &[u64],
) -> Result<ColumnInfo> {
    if header == NULL_HEADER && mask.is_some() {
        panic!("Null column shouldn't have a mask");
    }
    if header != NULL_HEADER && mask.is_none() {
        panic!("Column should have a mask");
    }
    let resolved_mask = mask.map(|mask_arr| mask_arr.to_vec());
    let binary_t = array_type(vec![shape[0]], BIT);
    let column_t = if header == NULL_HEADER {
        binary_t
    } else {
        tuple_type(vec![binary_t, array_type(shape.to_vec(), st)])
    };
    let t = ColumnType::new(column_t, true, header)?;
    Ok(ColumnInfo {
        header: header.to_owned(),
        t,
        mask: resolved_mask,
        data: data.to_vec(),
    })
}

pub(crate) type SetInfo = Vec<ColumnInfo>;

pub(crate) trait SetHelpers {
    fn get_type(&self) -> Type;
    fn get_value(&self) -> Result<Value>;
}

impl SetHelpers for SetInfo {
    fn get_type(&self) -> Type {
        let mut v = vec![];
        for col in self.iter() {
            v.push((col.header.clone(), col.t.clone().into()));
        }
        named_tuple_type(v)
    }
    fn get_value(&self) -> Result<Value> {
        let mut v = vec![];
        for col in self.iter() {
            v.push(col.get_value()?);
        }
        Ok(Value::from_vector(v))
    }
}

pub(crate) type ExpectedSetInfo = Vec<(String, Option<Vec<u64>>, Vec<u64>)>;

pub(crate) fn expected_set_info(expected_columns: Vec<(&str, &[u64])>) -> ExpectedSetInfo {
    let mut v = vec![];
    for (header, data) in expected_columns {
        v.push((header.to_owned(), None, data.to_vec()));
    }
    v
}

type ExpectedColumn<'a> = (&'a str, Option<&'a [u64]>, &'a [u64]);

pub(crate) fn expected_set_info_with_mask(
    expected_columns: Vec<ExpectedColumn>,
) -> ExpectedSetInfo {
    let mut v = vec![];
    for (header, mask, data) in expected_columns {
        if header == NULL_HEADER && mask.is_some() {
            panic!("Null column shouldn't have a column mask");
        }
        let resolved_mask = mask.map(|mask_arr| mask_arr.to_vec());
        v.push((header.to_owned(), resolved_mask, data.to_vec()));
    }
    v
}

#[derive(Clone)]
pub(crate) struct JoinTestInfo {
    pub set0: SetInfo,
    pub set1: SetInfo,
    pub headers: HashMap<String, String>,
    pub expected: HashMap<JoinType, ExpectedSetInfo>,
    pub has_column_masks: bool,
}

pub(crate) fn join_info(
    set0: SetInfo,
    set1: SetInfo,
    headers: Vec<(&str, &str)>,
    expected: HashMap<JoinType, ExpectedSetInfo>,
) -> JoinTestInfo {
    let mut hmap = HashMap::new();
    for (h0, h1) in headers {
        hmap.insert(h0.to_owned(), h1.to_owned());
    }
    let mut has_column_masks = false;
    for col in &set0 {
        if col.has_mask() {
            has_column_masks = true;
            break;
        }
    }
    JoinTestInfo {
        set0,
        set1,
        headers: hmap,
        expected,
        has_column_masks,
    }
}
