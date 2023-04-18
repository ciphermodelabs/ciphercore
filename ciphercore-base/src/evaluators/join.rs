use std::collections::HashMap;

use crate::{
    data_types::{get_named_types, named_tuple_type, HeadersTypes, Type, BIT},
    data_values::Value,
    errors::Result,
    graphs::JoinType,
    join_utils::ColumnType,
    type_inference::NULL_HEADER,
    typed_value::TypedValue,
};

struct ColumnsMap {
    // column header -> column array index
    header_index_map: HashMap<String, usize>,
    // Null column
    null_column: Vec<u8>,
    // arrays containing masks of all columns
    columns_masks: Vec<Vec<u8>>,
    // arrays containing data of all columns
    columns_data: Vec<Vec<u128>>,
    // number of scalar elements per row of each column
    elements_per_row: Vec<usize>,
}

impl ColumnsMap {
    fn get_row_size(&self, header: &str) -> usize {
        self.elements_per_row[self.header_index_map[header]]
    }
    fn get_data(&self, header: &str) -> Vec<u128> {
        self.columns_data[self.header_index_map[header]].clone()
    }
    fn has_column_masks(&self) -> bool {
        !self.columns_masks.is_empty()
    }
    fn get_mask_entry(&self, header: &str, row_index: usize) -> u8 {
        self.columns_masks[self.header_index_map[header]][row_index]
    }
    fn get_num_rows(&self) -> usize {
        self.null_column.len()
    }
    // Checks that the row with a given index has 0 in the null column or contains entries with the zero mask in given columns
    fn row_has_empty_entries(&self, row_index: usize, headers: &[String]) -> bool {
        if self.null_column[row_index] == 0 {
            return true;
        }
        if !self.has_column_masks() {
            return false;
        }
        for h in headers {
            if self.get_mask_entry(h, row_index) == 0 {
                return true;
            }
        }
        false
    }

    // Merges given columns of the row with a given index.
    fn get_flattened_row(&self, row_index: usize, headers: &[String]) -> Vec<u128> {
        let mut row = vec![];
        for header in headers {
            let row_data = self.get_data(header);
            let row_size = self.get_row_size(header);
            row.extend(row_data[row_index * row_size..(row_index + 1) * row_size].to_vec());
        }
        row
    }

    fn get_entry(&self, row_index: usize, header: &str) -> Vec<u128> {
        self.get_flattened_row(row_index, &[header.to_owned()])
    }
}

fn extract_columns(
    set: &TypedValue,
    headers_types: &HeadersTypes,
    has_column_masks: bool,
) -> Result<ColumnsMap> {
    let mut header_index_map = HashMap::new();
    let mut columns_data = vec![];
    let mut columns_masks = vec![];
    let mut elements_per_row = vec![];
    let mut null_column = vec![];
    let set_columns = set.value.to_vector()?;
    let mut ind = 0;
    for (i, (header, sub_t)) in headers_types.iter().enumerate() {
        if header == NULL_HEADER {
            null_column = set_columns[i].to_flattened_array_u8((**sub_t).clone())?;
            continue;
        }
        let column_type = ColumnType::new((**sub_t).clone(), has_column_masks, header)?;
        if has_column_masks {
            let column_mask_data = set_columns[i].to_vector()?;
            let mask = column_mask_data[0].clone();
            let data = column_mask_data[1].clone();
            let column_mask = mask.to_flattened_array_u8(column_type.get_mask_type()?)?;
            let column_data = data.to_flattened_array_u128(column_type.get_data_type())?;
            columns_masks.push(column_mask);
            columns_data.push(column_data);
        } else {
            let column_data =
                set_columns[i].to_flattened_array_u128(column_type.get_data_type())?;
            columns_data.push(column_data);
        }
        elements_per_row.push(column_type.get_row_size_in_elements());
        header_index_map.insert((*header).clone(), ind);
        ind += 1;
    }

    Ok(ColumnsMap {
        header_index_map,
        null_column,
        columns_masks,
        columns_data,
        elements_per_row,
    })
}

impl ColumnsMap {
    fn init_result_columns(join_input: &JoinInput) -> Result<Self> {
        let has_column_masks = join_input.set0.has_column_masks();

        let num_columns = join_input.result_headers_types.len() - 1;
        let mut header_index_map: HashMap<String, usize> = HashMap::new();
        let mut elements_per_row = vec![];
        let mut ind = 0;
        for (h, sub_t) in join_input.result_headers_types.iter() {
            if h == NULL_HEADER {
                continue;
            }
            let column_type = ColumnType::new((**sub_t).clone(), has_column_masks, h)?;
            elements_per_row.push(column_type.get_row_size_in_elements());
            header_index_map.insert((*h).clone(), ind);
            ind += 1;
        }

        let columns_masks = if has_column_masks {
            vec![vec![]; num_columns]
        } else {
            vec![]
        };

        Ok(ColumnsMap {
            header_index_map,
            null_column: vec![],
            columns_masks,
            columns_data: vec![vec![]; num_columns],
            elements_per_row,
        })
    }

    fn append_zero_row(&mut self) {
        // Add 0 to the null column
        self.null_column.push(0);
        // Add zero data columns
        for (col_i, column_data) in self.columns_data.iter_mut().enumerate() {
            column_data.extend(vec![0; self.elements_per_row[col_i]]);
            if !self.columns_masks.is_empty() {
                self.columns_masks[col_i].push(0);
            }
        }
    }

    fn append_zero_entry(&mut self, header: &str) {
        let row_size = self.get_row_size(header);
        let col_ind = self.header_index_map[header];
        self.columns_data[col_ind].extend(vec![0; row_size]);
        if !self.columns_masks.is_empty() {
            self.columns_masks[col_ind].push(0);
        }
    }

    // Add new elements from a column with a given header to another
    fn copy_entry_from_column(
        &mut self,
        target_header: &str,
        source_header: &str,
        source_columns: &ColumnsMap,
        row_index: usize,
    ) {
        if target_header == NULL_HEADER && source_header == NULL_HEADER {
            self.null_column.push(source_columns.null_column[row_index]);
            return;
        }
        let target_index = self.header_index_map[target_header];
        if source_columns.has_column_masks() {
            let source_mask_entry = source_columns.get_mask_entry(source_header, row_index);
            if source_mask_entry == 1 {
                self.columns_masks[target_index].push(source_mask_entry);
                let source_entry = source_columns.get_entry(row_index, source_header);
                self.columns_data[target_index].extend(source_entry);
            } else {
                self.append_zero_entry(target_header);
            }
        } else {
            let source_entry = source_columns.get_entry(row_index, source_header);
            self.columns_data[target_index].extend(source_entry);
        }
    }

    fn to_value(&self, join_input: JoinInput) -> Result<Value> {
        // Collect all columns
        let mut res_value_vec = vec![];
        for (h, t) in join_input.result_headers_types.iter() {
            if h == NULL_HEADER {
                res_value_vec.push(Value::from_flattened_array(&self.null_column, BIT)?);
                continue;
            }
            let col_ind = self.header_index_map[h];
            let column_t = ColumnType::new((**t).clone(), self.has_column_masks(), h)?;
            let data_val = Value::from_flattened_array(
                &self.columns_data[col_ind],
                column_t.get_scalar_type(),
            )?;
            if self.has_column_masks() {
                let mask_val = Value::from_flattened_array(&self.columns_masks[col_ind], BIT)?;
                res_value_vec.push(Value::from_vector(vec![mask_val, data_val]));
            } else {
                res_value_vec.push(data_val);
            }
        }
        Ok(Value::from_vector(res_value_vec))
    }
}

// (row_hash_key, row_index)
type KeyColumnsMap = HashMap<Vec<u128>, usize>;

// Creates a hashmap of the input rows with keys corresponding to the content of key columns
fn get_hashmap_from_key_columns(
    headers_values: &ColumnsMap,
    key_headers: &[String],
) -> Result<KeyColumnsMap> {
    let mut key_data_hashmap = HashMap::new();
    for i in 0..headers_values.get_num_rows() {
        if headers_values.row_has_empty_entries(i, key_headers) {
            continue;
        }
        let row_key = headers_values.get_flattened_row(i, key_headers);
        key_data_hashmap.insert(row_key, i);
    }
    Ok(key_data_hashmap)
}

// Computes the inner join of given sets.
fn get_inner_join_columns(join_input: &JoinInput) -> Result<ColumnsMap> {
    let mut result_columns = ColumnsMap::init_result_columns(join_input)?;

    // Check row by row that the content of the key columns of the first set intersects with the second set
    for row_ind0 in 0..join_input.number_of_rows_set_0() {
        // If the null column entry is zero or there are empty entries, just insert a zero row
        if join_input.row_has_empty_entries(row_ind0) {
            result_columns.append_zero_row();
            continue;
        }
        // Merge the key columns of the first set
        let row_key = join_input
            .set0
            .get_flattened_row(row_ind0, &join_input.key_headers0);
        // Check that the row key of the first set is contained in the second set
        if join_input.key_data_hashmap1.contains_key(&row_key) {
            // Add the columns of the first set first
            for (header0, _) in join_input.headers_types0 {
                result_columns.copy_entry_from_column(header0, header0, &join_input.set0, row_ind0);
            }
            // Extract the corresponding row of the second set
            let row_ind1 = join_input.key_data_hashmap1[&row_key];
            for header1 in &join_input.nonkey_headers1 {
                result_columns.copy_entry_from_column(header1, header1, &join_input.set1, row_ind1);
            }
        } else {
            result_columns.append_zero_row();
        }
    }
    Ok(result_columns)
}

// Preprocessed input of join algorithms.
// Note that it contains a hashmap of the rows of the second set hashed by their key columns, which simplifies computation of the set intersection
struct JoinInput<'a> {
    // Columns of the first set
    set0: ColumnsMap,
    // Columns of the second set
    set1: ColumnsMap,
    // Headers and types of the first set
    headers_types0: &'a HeadersTypes,
    // Key headers of the first set
    key_headers0: Vec<String>,
    // Header of the second set without NULL_HEADER and key headers
    nonkey_headers1: Vec<String>,
    // Map of key headers: first set header -> second set header
    key_headers_map: HashMap<String, String>,
    // Keys of the second set with the corresponding rows kept in a hashmap
    key_data_hashmap1: KeyColumnsMap,
    // Headers and types of the result
    result_headers_types: &'a HeadersTypes,
    // Headers of non-key columns having the same name in both sets
    // WARNING: these are used in union join only while computing full join! Keep it empty for other types of joins!
    same_headers: Vec<String>,
}

impl<'a> JoinInput<'a> {
    fn number_of_rows_set_0(&self) -> usize {
        self.set0.get_num_rows()
    }
    fn row_has_empty_entries(&self, row_index: usize) -> bool {
        self.set0
            .row_has_empty_entries(row_index, &self.key_headers0)
    }
}

// Computes the left join of given sets.
// These sets and necessary precomputations should be contained in the join input.
fn get_left_join_columns(join_input: &JoinInput) -> Result<ColumnsMap> {
    let mut result_columns = ColumnsMap::init_result_columns(join_input)?;

    // Check row by row that the content of the key columns of the first set intersects with the second one.
    // If yes, append the row of the first set with the columns of the second one.
    // If no, append it with zeros.
    for row_ind0 in 0..join_input.number_of_rows_set_0() {
        // If the null column entry is zero, just insert a zero row
        if join_input.set0.null_column[row_ind0] == 0 {
            result_columns.append_zero_row();
            continue;
        }
        // Add the columns of the first set first
        for (header0, _) in join_input.headers_types0.iter() {
            result_columns.copy_entry_from_column(header0, header0, &join_input.set0, row_ind0);
        }
        // If there are empty entries in the key columns, append the current row with zeros
        if join_input.row_has_empty_entries(row_ind0) {
            for header1 in &join_input.nonkey_headers1 {
                result_columns.append_zero_entry(header1);
            }
            continue;
        }
        // Merge the key columns of the first set
        let row_key = join_input
            .set0
            .get_flattened_row(row_ind0, &join_input.key_headers0);
        // Check that the row key of the first set is contained in the second set
        if join_input.key_data_hashmap1.contains_key(&row_key) {
            // Extract the corresponding row of the second set and attach to the row of the first set
            let row_ind1 = join_input.key_data_hashmap1[&row_key];
            for header1 in &join_input.nonkey_headers1 {
                result_columns.copy_entry_from_column(header1, header1, &join_input.set1, row_ind1);
            }
        } else {
            // Append the current row with zeros
            for header1 in &join_input.nonkey_headers1 {
                result_columns.append_zero_entry(header1);
            }
        }
    }
    Ok(result_columns)
}

// Computes the union join of given sets.
// These sets and necessary precomputations should be contained in the join input.
// In contrast to the SQL union, this operation does not require that input datasets have respective columns of the same type.
// This means that columns of both datasets are included and filled with zeros where no data can be retrieved.
// Namely, the rows of the second set in the union join will contain zeros in non-key columns of the first set and vice versa.
fn get_union_columns(join_input: &JoinInput) -> Result<ColumnsMap> {
    let mut result_columns = ColumnsMap::init_result_columns(join_input)?;

    // Add all the rows of the first set that don't belong to the inner join
    for row_ind0 in 0..join_input.set0.get_num_rows() {
        // If the null bit is zero, just insert a zero row
        if join_input.set0.null_column[row_ind0] == 0 {
            result_columns.append_zero_row();
            continue;
        }

        if join_input.row_has_empty_entries(row_ind0) {
            // Extract the rows of the first set
            for (header0, _) in join_input.headers_types0.iter() {
                result_columns.copy_entry_from_column(header0, header0, &join_input.set0, row_ind0);
            }
            // Append the current row with zeros in the remaining unique columns of the second set
            for h in &join_input.nonkey_headers1 {
                if !join_input.same_headers.contains(h) {
                    result_columns.append_zero_entry(h);
                }
            }
            continue;
        }
        // Merge the key columns of the first set
        let row_key = join_input
            .set0
            .get_flattened_row(row_ind0, &join_input.key_headers0);
        // Check that the row key of the first set is contained in the second set
        if join_input.key_data_hashmap1.contains_key(&row_key) {
            // Fill the current row with zeros
            result_columns.append_zero_row();
        } else {
            // Extract the rows of the first set
            for (header0, _) in join_input.headers_types0.iter() {
                result_columns.copy_entry_from_column(header0, header0, &join_input.set0, row_ind0);
            }
            // Append the current row with zeros in the remaining unique columns of the second set
            for h in &join_input.nonkey_headers1 {
                if !join_input.same_headers.contains(h) {
                    result_columns.append_zero_entry(h);
                }
            }
        }
    }

    // Add all the rows of the second set
    for row_ind1 in 0..join_input.set1.get_num_rows() {
        if join_input.set1.null_column[row_ind1] == 0 {
            result_columns.append_zero_row();
            continue;
        }
        for (h, _) in join_input.result_headers_types {
            if h == NULL_HEADER {
                // the null bit will be always 1 here
                result_columns.null_column.push(1);
            } else if join_input.key_headers0.contains(h) {
                // extract the content of a key column of the second set
                let header1 = &join_input.key_headers_map[h];
                result_columns.copy_entry_from_column(h, header1, &join_input.set1, row_ind1);
            } else if join_input.nonkey_headers1.contains(h) {
                // non-key columns of the second set
                result_columns.copy_entry_from_column(h, h, &join_input.set1, row_ind1);
            } else {
                // non-key columns of the first set. Fill them with zeros.
                result_columns.append_zero_entry(h);
            }
        }
    }
    Ok(result_columns)
}

fn get_number_of_rows(set_t: &Type) -> Result<u64> {
    let headers_types = get_named_types(set_t)?;
    if headers_types.is_empty() {
        return Err(runtime_error!("The set is empty"));
    }
    let first_column_t = ColumnType::new((*headers_types[0].1).clone(), true, &headers_types[0].0)?;
    Ok(first_column_t.get_num_entries())
}

fn evaluate_full_join(
    set0: TypedValue,
    set1: TypedValue,
    has_column_masks: bool,
    headers: &HashMap<String, String>,
    res_t: Type,
) -> Result<Value> {
    // First, evaluate the left join of set1 and set0.
    let num_rows1 = get_number_of_rows(&set1.t)?;
    let headers_types0 = get_named_types(&set0.t)?;
    // Resulting type of this left join. The columns of set1 go first, then the remaining columns of set0.
    let left_join_t = {
        let mut result_types_vec = vec![];
        let headers_types1 = get_named_types(&set1.t)?;
        for (h, sub_t) in headers_types1 {
            result_types_vec.push((h.clone(), (**sub_t).clone()));
        }
        for (h, sub_t) in headers_types0 {
            if h != NULL_HEADER && !headers.contains_key(h) {
                let column_t = ColumnType::new((**sub_t).clone(), has_column_masks, h)?;
                let res_t = column_t.clone_with_number_of_entries(num_rows1);
                result_types_vec.push((h.clone(), res_t.into()));
            }
        }

        named_tuple_type(result_types_vec)
    };
    // The headers of the left join are the reversed input headers.
    let mut left_join_headers = HashMap::new();
    for (h0, h1) in headers {
        left_join_headers.insert(h1.clone(), h0.clone());
    }

    let left_join_value = evaluate_join(
        JoinType::Left,
        set1,
        set0.clone(),
        has_column_masks,
        &left_join_headers,
        left_join_t.clone(),
    )?;
    let left_join = TypedValue {
        value: left_join_value,
        t: left_join_t,
        name: None,
    };
    // Then, evaluate the union of set0 and the left join result
    evaluate_join(
        JoinType::Union,
        set0,
        left_join,
        has_column_masks,
        headers,
        res_t,
    )
}

pub(crate) fn evaluate_join(
    join_t: JoinType,
    set0_tv: TypedValue,
    set1_tv: TypedValue,
    has_column_masks: bool,
    headers: &HashMap<String, String>,
    res_t: Type,
) -> Result<Value> {
    if join_t == JoinType::Full {
        return evaluate_full_join(set0_tv, set1_tv, has_column_masks, headers, res_t);
    }
    let headers_types1 = get_named_types(&set1_tv.t)?;
    // Extract columns of the second set
    let set1 = extract_columns(&set1_tv, headers_types1, has_column_masks)?;

    let (key_headers0, key_headers1): (Vec<_>, Vec<_>) = headers.clone().into_iter().unzip();
    // Key columns of the second set are merged and added to the hash map along with the corresponding rows
    let key_data_hashmap1 = get_hashmap_from_key_columns(&set1, &key_headers1)?;

    let headers_types0 = get_named_types(&set0_tv.t)?;
    // Extract columns of the first set
    let set0 = extract_columns(&set0_tv, headers_types0, has_column_masks)?;

    let result_headers_types = get_named_types(&res_t)?;

    let mut nonkey_headers1 = vec![];
    for (h, _) in headers_types1 {
        if h != NULL_HEADER && !key_headers1.contains(h) {
            nonkey_headers1.push((*h).clone());
        }
    }
    let mut same_headers = vec![];
    for (h, _) in headers_types0 {
        if nonkey_headers1.contains(h) {
            same_headers.push((*h).clone());
        }
    }

    let join_input = JoinInput {
        set0,
        set1,
        headers_types0,
        key_headers0,
        nonkey_headers1,
        key_headers_map: headers.clone(),
        key_data_hashmap1,
        result_headers_types,
        same_headers,
    };

    let res_columns = match join_t {
        JoinType::Inner => get_inner_join_columns(&join_input)?,
        JoinType::Left => get_left_join_columns(&join_input)?,
        JoinType::Union => get_union_columns(&join_input)?,
        JoinType::Full => {
            // Full join is computed via left and union joins
            return Err(runtime_error!("Shouldn't be here"));
        }
    };
    res_columns.to_value(join_input)
}
