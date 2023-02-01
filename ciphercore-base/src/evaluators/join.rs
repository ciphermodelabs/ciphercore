use std::{collections::HashMap, sync::Arc};

use crate::{
    data_types::Type, data_values::Value, errors::Result, graphs::JoinType,
    type_inference::NULL_HEADER,
};

// column header -> column array type
type HeadersTypes = Vec<(String, Arc<Type>)>;

fn get_named_types(t: Type) -> HeadersTypes {
    if let Type::NamedTuple(v) = t {
        v
    } else {
        panic!("Can't get named types. Input type must be NamedTuple.")
    }
}

struct ColumnsMap {
    // column header -> column array
    header_index_map: HashMap<String, usize>,
    // arrays containing data of all columns
    columns_data: Vec<Vec<u64>>,
    // number of scalar elements per row of each column
    elements_per_row: Vec<usize>,
}

impl ColumnsMap {
    fn get_row_size(&self, header: &str) -> usize {
        self.elements_per_row[self.header_index_map[header]]
    }
    fn get_data(&self, header: &str) -> Vec<u64> {
        self.columns_data[self.header_index_map[header]].clone()
    }
}

fn extract_columns(set_value: Value, headers_types: &HeadersTypes) -> Result<ColumnsMap> {
    let mut header_index_map = HashMap::new();
    let mut columns_data = vec![];
    let mut elements_per_row = vec![];
    let set_columns = set_value.to_vector()?;
    for (i, (header, column_t)) in headers_types.iter().enumerate() {
        let column_array = set_columns[i].to_flattened_array_u64((**column_t).clone())?;
        columns_data.push(column_array);
        let column_shape = column_t.get_shape();
        elements_per_row.push(column_shape.iter().skip(1).product::<u64>() as usize);
        header_index_map.insert((*header).clone(), i);
    }

    Ok(ColumnsMap {
        header_index_map,
        columns_data,
        elements_per_row,
    })
}

fn append_zero_row(columns: &mut [Vec<u64>], join_input: &JoinInput) {
    // Add columns of the first set
    for (col_i, (header0, _)) in join_input.headers_types0.iter().enumerate() {
        let row_size = join_input.columns0.get_row_size(header0);
        columns[col_i].extend(vec![0; row_size]);
    }
    // Add remaining columns of the second set
    for (col_i, column) in columns
        .iter_mut()
        .enumerate()
        .take(join_input.result_headers_types.len())
        .skip(join_input.headers_types0.len())
    {
        let header = join_input.result_headers_types[col_i].0.clone();
        let row_size = join_input.columns1.get_row_size(&header);
        column.extend(vec![0; row_size]);
    }
}

// Merges given columns of the row with a given index
fn flatten_row(row_index: usize, headers: &[String], headers_values: &ColumnsMap) -> Vec<u64> {
    let mut row = vec![];
    for header in headers {
        let row_data = headers_values.get_data(header);
        let row_size = headers_values.get_row_size(header);
        row.extend(row_data[row_index * row_size..(row_index + 1) * row_size].to_vec());
    }
    row
}

// Add new elements from a column with a given header to another
fn extend_column(
    target_column: &mut Vec<u64>,
    source_header: &str,
    columns: &ColumnsMap,
    row_index: usize,
) {
    let source_column = columns.get_data(source_header);
    let row_size = columns.get_row_size(source_header);
    target_column.extend(source_column[row_index * row_size..(row_index + 1) * row_size].to_vec());
}

type KeyColumnsMap = HashMap<Vec<u64>, Vec<Vec<u64>>>;

// Creates a hashmap of the input rows with keys corresponding to the content of key columns
fn get_hashmap_from_key_columns(
    headers_values: &ColumnsMap,
    headers_types: &HeadersTypes,
    key_headers: &[String],
) -> Result<KeyColumnsMap> {
    // Extract the null column of the second set
    let null_column = headers_values.get_data(NULL_HEADER);
    // Key columns of the second set are merged and added to the hash map along with the corresponding rows
    let mut key_data_hashmap = HashMap::new();
    for (i, null_bit) in null_column.iter().enumerate() {
        if *null_bit == 0 {
            continue;
        }
        let row_key = flatten_row(i, key_headers, headers_values);
        let mut row = vec![];
        for (header, _) in headers_types {
            if !key_headers.contains(header) && header != NULL_HEADER {
                let row_data = headers_values.get_data(header);
                let row_size = headers_values.get_row_size(header);
                row.push(row_data[i * row_size..(i + 1) * row_size].to_vec());
            }
        }
        key_data_hashmap.insert(row_key, row);
    }
    Ok(key_data_hashmap)
}

// Computes the inner join of given sets.
// These sets and necessary precomputations should be contained in the join input.
fn get_inner_join_columns(join_input: &JoinInput) -> Vec<Vec<u64>> {
    // The resulting null column is derived from the null column of the first set
    let mut res_null_column = join_input.columns0.get_data(NULL_HEADER);
    // Merge key columns of the first set and check if the corresponding rows belong to the second set
    let num_res_columns = join_input.result_headers_types.len();
    let mut res_columns = vec![vec![]; num_res_columns];

    // Check row by row that the content of the key columns of the first set intersects with the second set
    for (i, null_bit) in res_null_column.iter_mut().enumerate() {
        // If the null bit is zero, just insert a zero row
        if *null_bit == 0 {
            append_zero_row(&mut res_columns, join_input);
            continue;
        }
        // Merge the key columns of the first set
        let row_key = flatten_row(i, &join_input.key_headers0, &join_input.columns0);
        // Check that the row key of the first set is contained in the second set
        if join_input.key_data_hashmap1.contains_key(&row_key) {
            // Add the columns of the first set first
            for (col_i, (header0, _)) in join_input.headers_types0.iter().enumerate() {
                extend_column(&mut res_columns[col_i], header0, &join_input.columns0, i);
            }
            // Extract the corresponding row of the second set
            let row_data1 = join_input.key_data_hashmap1[&row_key].clone();
            for col_i in 0..row_data1.len() {
                res_columns[join_input.headers_types0.len() + col_i]
                    .extend(row_data1[col_i].clone());
            }
        } else {
            *null_bit = 0;
            append_zero_row(&mut res_columns, join_input);
        }
    }
    res_columns
}

// Preprocessed input of join algorithms.
// Note that it contains a hashmap of the rows of the second set hashed by their key columns, which simplifies computation of the set intersection
struct JoinInput {
    // Columns of the first set
    columns0: ColumnsMap,
    // Columns of the second set
    columns1: ColumnsMap,
    // Headers and types of the first set
    headers_types0: HeadersTypes,
    // Key headers of the first set
    key_headers0: Vec<String>,
    // Map of key headers: first set header -> second set header
    key_headers_map: HashMap<String, String>,
    // Keys of the second set with the corresponding rows kept in a hashmap
    key_data_hashmap1: HashMap<Vec<u64>, Vec<Vec<u64>>>,
    // Headers and types of the result
    result_headers_types: HeadersTypes,
}

// Computes the left join of given sets.
// These sets and necessary precomputations should be contained in the join input.
fn get_left_join_columns(join_input: &JoinInput) -> Vec<Vec<u64>> {
    // The resulting null column is derived from the null column of the first set
    let mut res_null_column = join_input.columns0.get_data(NULL_HEADER);
    // Merge key columns of the first set and check if the corresponding rows belong to the second set
    let num_res_columns = join_input.result_headers_types.len();
    let mut res_columns = vec![vec![]; num_res_columns];

    // Check row by row that the content of the key columns of the first set intersects with the second one.
    // If yes, append the row of the first set with the columns of the second one.
    // If no, append it with zeros.
    for (i, null_bit) in res_null_column.iter_mut().enumerate() {
        // If the null bit is zero, just insert a zero row
        if *null_bit == 0 {
            append_zero_row(&mut res_columns, join_input);
            continue;
        }
        // Add the columns of the first set first
        for (col_i, (header0, _)) in join_input.headers_types0.iter().enumerate() {
            extend_column(&mut res_columns[col_i], header0, &join_input.columns0, i);
        }
        // Merge the key columns of the first set
        let row_key = flatten_row(i, &join_input.key_headers0, &join_input.columns0);
        // Check that the row key of the first set is contained in the second set
        if join_input.key_data_hashmap1.contains_key(&row_key) {
            // Extract the corresponding row of the second set and attach to the row of the first set
            let row_data1 = join_input.key_data_hashmap1[&row_key].clone();
            for col_i in 0..row_data1.len() {
                res_columns[join_input.headers_types0.len() + col_i]
                    .extend(row_data1[col_i].clone());
            }
        } else {
            // Append the current row with zeros
            for (col_i, column) in res_columns
                .iter_mut()
                .enumerate()
                .skip(join_input.headers_types0.len())
            {
                let header = &join_input.result_headers_types[col_i].0;
                let row_size = join_input.columns1.get_row_size(header);
                column.extend(vec![0; row_size]);
            }
        }
    }
    res_columns
}

// Computes the union join of given sets.
// These sets and necessary precomputations should be contained in the join input.
// In contrast to the SQL union, this operation does not require that input datasets have respective columns of the same type.
// This means that columns of both datasets are included and filled with zeros where no data can be retrieved.
// Namely, the rows of the second set in the union join will contain zeros in non-key columns of the first set and vice versa.
fn get_union_columns(join_input: &JoinInput) -> Vec<Vec<u64>> {
    // The first part of the resulting null column is derived from the null column of the first set
    let mut x_null_column = join_input.columns0.get_data(NULL_HEADER);
    // Merge key columns of the first set and check if the corresponding rows belong to the second set
    let num_res_columns = join_input.result_headers_types.len();
    let mut res_columns = vec![vec![]; num_res_columns];

    // Add all the rows of the first set that don't belong to the inner join
    for (i, null_bit) in x_null_column.iter_mut().enumerate() {
        // If the null bit is zero, just insert a zero row
        if *null_bit == 0 {
            append_zero_row(&mut res_columns, join_input);
            continue;
        }
        // Merge the key columns of the first set
        let row_key = flatten_row(i, &join_input.key_headers0, &join_input.columns0);
        // Check that the row key of the first set is contained in the second set
        if join_input.key_data_hashmap1.contains_key(&row_key) {
            // Fill the current row with zeros
            *null_bit = 0;
            append_zero_row(&mut res_columns, join_input);
        } else {
            // Extract the rows of the first set
            for (col_i, (header0, _)) in join_input.headers_types0.iter().enumerate() {
                extend_column(&mut res_columns[col_i], header0, &join_input.columns0, i);
            }
            // Append the current row with zeros in the remaining columns of the second set
            for (col_i, column) in res_columns
                .iter_mut()
                .enumerate()
                .skip(join_input.headers_types0.len())
            {
                let header = &join_input.result_headers_types[col_i].0;
                let row_size = join_input.columns1.get_row_size(header);
                column.extend(vec![0; row_size]);
            }
        }
    }

    // The second part of the resulting null column is derived from the null column of the second set
    let y_null_column = join_input.columns1.get_data(NULL_HEADER);
    // Add all the rows of the second set
    for (i, null_bit) in y_null_column.iter().enumerate() {
        if *null_bit == 0 {
            append_zero_row(&mut res_columns, join_input);
            continue;
        }
        for (col_i, column) in res_columns.iter_mut().enumerate() {
            let header = &join_input.result_headers_types[col_i].0;
            if header == NULL_HEADER {
                // the null bit will be always 1 here
                column.push(*null_bit);
            } else if join_input.key_headers0.contains(header) {
                // extract the content of a key column of the second set
                let y_header = &join_input.key_headers_map[header];
                extend_column(column, y_header, &join_input.columns1, i);
            } else if col_i < join_input.headers_types0.len() {
                // non-key columns of the first set. Fill them with zeros.
                let row_size = join_input.columns0.get_row_size(header);
                column.extend(vec![0; row_size]);
            } else {
                // non-key columns of the second set
                extend_column(column, header, &join_input.columns1, i);
            }
        }
    }
    res_columns
}

pub(crate) fn evaluate_join(
    join_t: JoinType,
    set0: Value,
    set1: Value,
    set0_t: Type,
    set1_t: Type,
    headers: &HashMap<String, String>,
    res_t: Type,
) -> Result<Value> {
    let headers_types1 = get_named_types(set1_t);
    // Extract columns of the second set
    let columns1 = extract_columns(set1, &headers_types1)?;

    let (key_headers0, key_headers1): (Vec<_>, Vec<_>) = headers.clone().into_iter().unzip();
    // Key columns of the second set are merged and added to the hash map along with the corresponding rows
    let key_data_hashmap1 =
        get_hashmap_from_key_columns(&columns1, &headers_types1, &key_headers1)?;

    let headers_types0 = get_named_types(set0_t);
    // Extract columns of the first set
    let columns0 = extract_columns(set0, &headers_types0)?;

    let result_headers_types = get_named_types(res_t);

    let join_input = JoinInput {
        columns0,
        columns1,
        headers_types0,
        key_headers0,
        key_headers_map: headers.clone(),
        key_data_hashmap1,
        result_headers_types,
    };

    let res_columns = match join_t {
        JoinType::Inner => get_inner_join_columns(&join_input),
        JoinType::Left => get_left_join_columns(&join_input),
        JoinType::Union => get_union_columns(&join_input),
    };
    // Collect all columns
    let mut res_value_vec = vec![];
    for (i, (_, t)) in join_input.result_headers_types.iter().enumerate() {
        res_value_vec.push(Value::from_flattened_array(
            &res_columns[i],
            t.get_scalar_type(),
        )?);
    }
    Ok(Value::from_vector(res_value_vec))
}
