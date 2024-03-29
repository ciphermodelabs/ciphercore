use std::collections::HashMap;

use crate::custom_ops::{CustomOperation, CustomOperationBody};
use crate::data_types::{
    array_type, get_types_vector, named_tuple_type, tuple_type, Type, BIT, UINT64,
};
use crate::errors::Result;
use crate::graphs::util::simple_context;
use crate::graphs::{Context, Graph, JoinType, Node, NodeAnnotation, SliceElement};
use crate::join_utils::ColumnType;
use crate::ops::comparisons::Equal;
use crate::ops::utils::{custom_reduce, extend_with_zeros, zeros_like};
use crate::type_inference::NULL_HEADER;

use serde::{Deserialize, Serialize};

use super::low_mc::{LowMC, LowMCBlockSize, LOW_MC_KEY_SIZE};
use super::mpc_arithmetic::{add_mpc, gemm_mpc, mixed_multiply_mpc, multiply_mpc, subtract_mpc};
use super::mpc_compiler::{
    check_private_tuple, get_node_shares, get_zero_shares, IOStatus, KEY_LENGTH, PARTIES,
};
use super::utils::{convert_main_graph_to_mpc, get_column, select_node};

// Keeps information about column types of tables.
#[derive(Clone)]
struct ColumnTypesInfo {
    headers_types: Vec<(String, ColumnType)>,
    headers_map: HashMap<String, usize>,
    // To simplify Join, we need to flatten set to named tuples of arrays.
    // When column masks are present, we generate new headers for these masks, which can be accessed here as column_header -> mask_header
    mask_headers_map: HashMap<String, String>,
}

impl ColumnTypesInfo {
    fn new(headers_types: Vec<(String, ColumnType)>) -> Self {
        let mut mask_headers_map = HashMap::new();
        let mut headers_map = HashMap::new();
        for (i, (h, column_t)) in headers_types.iter().enumerate() {
            if column_t.has_mask() {
                // Mask names should be unique
                mask_headers_map.insert(h.into(), format!("{h}-mask-cvbphkieac",));
            }
            headers_map.insert(h.into(), i);
        }
        ColumnTypesInfo {
            headers_types,
            headers_map,
            mask_headers_map,
        }
    }
    fn get_num_entries(&self) -> u64 {
        self.headers_types[0].1.get_num_entries()
    }

    fn get_mask_header(&self, header: &str) -> Result<String> {
        if self.mask_headers_map.is_empty() {
            return Err(runtime_error!(
                "Headers for column masks are not yet generated"
            ));
        }
        Ok(self.mask_headers_map[header].clone())
    }

    fn get_column_type(&self, header: &str) -> ColumnType {
        self.headers_types[self.headers_map[header]].1.clone()
    }

    fn get_plaintext_type(&self) -> Type {
        let mut named_tuple_vec = vec![];
        for (h, col_t) in &self.headers_types {
            named_tuple_vec.push((h.clone(), col_t.clone().into()));
        }
        named_tuple_type(named_tuple_vec)
    }
}

const PRF_OUTPUT_SIZE: u64 = 80;

fn is_type_shared(t: &Type) -> bool {
    t.is_tuple()
}

fn generate_shared_random_array(t: Type, prf_keys: &[Node]) -> Result<Node> {
    let mut shares = vec![];
    for key in prf_keys {
        shares.push(key.prf(0, t.clone())?);
    }
    prf_keys[0].get_graph().create_tuple(shares)
}

fn get_column_type(column: &Node) -> Result<Type> {
    let t = column.get_type()?;
    if is_type_shared(&t) {
        Ok((*get_types_vector(t)?[0]).clone())
    } else {
        Ok(t)
    }
}

fn is_shared(column: &Node) -> Result<bool> {
    Ok(is_type_shared(&column.get_type()?))
}

fn reshape_shared_array(a: Node, new_t: Type) -> Result<Node> {
    if is_shared(&a)? {
        let mut shares = vec![];
        for share_id in 0..PARTIES as u64 {
            shares.push(a.tuple_get(share_id)?.reshape(new_t.clone())?);
        }
        a.get_graph().create_tuple(shares)
    } else {
        a.reshape(new_t)
    }
}

fn merge_mask_data(mask: Node, data: Node) -> Result<Node> {
    let g = mask.get_graph();
    if !is_shared(&mask)? {
        return g.create_tuple(vec![mask, data]);
    }
    let mut res_shares = vec![];
    for share_id in 0..PARTIES as u64 {
        res_shares
            .push(g.create_tuple(vec![mask.tuple_get(share_id)?, data.tuple_get(share_id)?])?);
    }
    g.create_tuple(res_shares)
}

fn concatenate_mpc(arrays: &[Node], axis: u64) -> Result<Node> {
    if arrays.is_empty() {
        panic!("Can't concatenate empty vector of nodes");
    }
    if arrays.len() == 1 {
        return Ok(arrays[0].clone());
    }
    let g = arrays[0].get_graph();
    let mut contains_private = false;
    for arr in arrays {
        contains_private |= is_shared(arr)?;
    }
    if !contains_private {
        return g.concatenate(arrays.to_vec(), axis);
    }
    let mut res_shares = vec![];
    for share_id in 0..PARTIES as u64 {
        let mut share_vec = vec![];
        for arr in arrays {
            if is_shared(arr)? {
                share_vec.push(arr.tuple_get(share_id)?);
            } else {
                // if node is public, fake its sharing, i.e., create a sharing (node, 0, 0).
                if share_id == 0 {
                    share_vec.push(arr.clone())
                } else {
                    let zeros = zeros_like(arr.clone())?;
                    share_vec.push(zeros);
                }
            }
        }
        res_shares.push(g.concatenate(share_vec, axis)?);
    }
    g.create_tuple(res_shares)
}

fn reveal_array_from_3_out_of_3(a: Node, party_id: u64) -> Result<Node> {
    // Share with party_id ID belongs to the given party.
    // The missing shares (when PARTIES = 3) are the shares with IDs equal to party_id - 1 and party_id + 1.
    let next_id = (party_id + 1) % PARTIES as u64;
    let previous_id = (party_id + PARTIES as u64 - 1) % PARTIES as u64;

    let missing_share1 = a
        .tuple_get(previous_id)?
        .nop()?
        .add_annotation(NodeAnnotation::Send(previous_id, party_id))?;

    let missing_share2 = a
        .tuple_get(next_id)?
        .nop()?
        .add_annotation(NodeAnnotation::Send(next_id, party_id))?;

    a.tuple_get(party_id)?
        .add(missing_share1)?
        .add(missing_share2)
}

fn sum_named_columns(a: Node, b: Node) -> Result<Node> {
    let header_types = a.get_type()?.get_named_types()?;
    let mut result_columns = vec![];
    for (header, _) in header_types {
        let c = a
            .named_tuple_get(header.clone())?
            .add(b.named_tuple_get(header.clone())?)?;
        result_columns.push((header, c));
    }
    a.get_graph().create_named_tuple(result_columns)
}

fn random_pad_columns(columns: Node, num_extra_rows: u64, prf_keys: &[Node]) -> Result<Node> {
    let graph = columns.get_graph();
    let header_types = {
        let tuple_types_vec = get_types_vector(columns.get_type()?)?;
        tuple_types_vec[0].get_named_types()?
    };
    // Null header should contain zeros to avoid false positives while comparing the rows
    let null_header_shares = get_zero_shares(
        graph.clone(),
        graph.create_tuple(prf_keys.to_vec())?,
        array_type(vec![num_extra_rows], BIT),
    )?;
    let mut shares = vec![];
    for (share_id, prf_key) in prf_keys.iter().enumerate() {
        let data_share = columns.tuple_get(share_id as u64)?;
        let mut result_columns = vec![];
        for (header, t) in header_types.clone() {
            let column = data_share.named_tuple_get(header.clone())?;
            let padded_column = if header == NULL_HEADER {
                // Merge input rows and extra rows
                graph.concatenate(vec![column, null_header_shares[share_id].clone()], 0)?
            } else {
                let mut extra_rows_shape = t.get_shape();
                extra_rows_shape[0] = num_extra_rows;
                let st = t.get_scalar_type();
                let extra_rows = prf_key.prf(0, array_type(extra_rows_shape.clone(), st))?;
                // Merge input rows and extra rows
                graph.concatenate(vec![column, extra_rows], 0)?
            };
            result_columns.push((header, padded_column));
        }
        let share = graph.create_named_tuple(result_columns)?;
        shares.push(share);
    }
    graph.create_tuple(shares)
}

fn zero_pad_column(
    column: Node,
    num_extra_rows: u64,
    in_front: bool,
    prf_keys: Node,
) -> Result<Node> {
    let g = column.get_graph();
    let t = get_column_type(&column)?;
    if is_shared(&column)? {
        let mut extra_rows_shape = t.get_shape();
        extra_rows_shape[0] = num_extra_rows;
        let st = t.get_scalar_type();
        let zero_rows = get_zero_shares(g.clone(), prf_keys, array_type(extra_rows_shape, st))?;

        let mut shares = vec![];
        for (share_id, zero_rows_share) in zero_rows.iter().cloned().enumerate() {
            let column_share = column.tuple_get(share_id as u64)?;
            // Merge input rows and extra rows
            let share = if in_front {
                g.concatenate(vec![zero_rows_share, column_share], 0)?
            } else {
                g.concatenate(vec![column_share, zero_rows_share], 0)?
            };
            shares.push(share);
        }
        g.create_tuple(shares)
    } else {
        let mut extra_rows_shape = t.get_shape();
        extra_rows_shape[0] = num_extra_rows;
        let st = t.get_scalar_type();
        let zero_rows = g.zeros(array_type(extra_rows_shape, st))?;
        if in_front {
            g.concatenate(vec![zero_rows, column], 0)
        } else {
            g.concatenate(vec![column, zero_rows], 0)
        }
    }
}

fn get_equality_graph(
    context: Context,
    type1: Type,
    type2: Type,
    key_header: String,
    is_input1_private: bool,
    is_input2_private: bool,
) -> Result<Graph> {
    let eq_context = simple_context(|g| {
        let i0 = g.input(type1)?;
        let i1 = g.input(type2)?;

        let key_columns_0 = i0.named_tuple_get(key_header.clone())?;
        let key_columns_1 = i1.named_tuple_get(key_header)?;

        let eq_bits = g.custom_op(
            CustomOperation::new(Equal {}),
            vec![key_columns_0, key_columns_1],
        )?;

        let null_0 = i0.named_tuple_get(NULL_HEADER.to_owned())?;
        let null_1 = i1.named_tuple_get(NULL_HEADER.to_owned())?;

        null_0.multiply(null_1)?.multiply(eq_bits)
    })?;

    convert_main_graph_to_mpc(
        eq_context,
        context,
        vec![is_input1_private, is_input2_private],
    )
}

// Extracts the null column and the column masks of key headers, and computes their product.
// The Boolean output indicates whether PRF keys are needed.
fn get_oprf_mask_graph(
    context: Context,
    column_types_info: &ColumnTypesInfo,
    key_headers: &[String],
    is_input_private: bool,
    has_column_masks: bool,
) -> Result<(Graph, bool)> {
    let oprf_null_column_context = simple_context(|g| {
        let i = g.input(column_types_info.get_plaintext_type())?;
        if !has_column_masks {
            return i.named_tuple_get(NULL_HEADER.to_string());
        }
        let mut masks_vec = vec![i.named_tuple_get(NULL_HEADER.into())?];
        for h in key_headers {
            let mask = i.named_tuple_get(h.into())?.tuple_get(0)?;
            masks_vec.push(mask);
        }
        let masks = g.stack(masks_vec, vec![key_headers.len() as u64 + 1])?;
        custom_reduce(masks, |first, second| first.multiply(second))
    })?;

    let g = convert_main_graph_to_mpc(oprf_null_column_context, context, vec![is_input_private])?;
    // If input is private and thera column masks, we have to resort to MPC multiplication of secret shared values.
    // Therefore, we need PRF keys.
    let needs_prf_keys = is_input_private && has_column_masks;
    Ok((g, needs_prf_keys))
}

fn get_select_graph(
    context: Context,
    column_header_types: Vec<(String, Type)>,
    num_entries: u64,
    key_header: String,
) -> Result<Graph> {
    let select_context = simple_context(|g| {
        let data_t = named_tuple_type(column_header_types.clone());
        let data_columns = g.input(data_t)?;

        let mask_t = array_type(vec![num_entries], BIT);
        let mask = g.input(mask_t)?;

        let mut result_columns = vec![];
        for (header, t) in column_header_types {
            if header == NULL_HEADER || header == key_header {
                continue;
            }
            let column = data_columns.named_tuple_get(header.clone())?;
            let column_shape = t.get_shape();
            // Reshape the mask to multiply row-wise
            let mut mask_shape = vec![num_entries];
            if column_shape.len() > 1 {
                mask_shape.extend(vec![1; column_shape.len() - 1]);
            }
            let column_mask = mask.reshape(array_type(mask_shape, BIT))?;
            // Multiply the column by the mask
            let result_column = if t.get_scalar_type() == BIT {
                column.multiply(column_mask)?
            } else {
                column.mixed_multiply(column_mask)?
            };

            result_columns.push((header, result_column));
        }
        g.create_named_tuple(result_columns)
    })?;

    convert_main_graph_to_mpc(select_context, context, vec![true, true])
}

fn get_lowmc_graph(context: Context, input_t: Type, key_t: Type) -> Result<Graph> {
    let lowmc_context = simple_context(|g| {
        // Compute OPRF of hashed key columns in both sets
        // Set the parameters of the LowMC block cipher serving here as PRF.
        // TODO: these parameters can be further optimized with great caution.
        // See `low_mc.rs` for guidelines.
        let block_size = match PRF_OUTPUT_SIZE {
            80 => LowMCBlockSize::SIZE80,
            128 => LowMCBlockSize::SIZE128,
            _ => {
                return Err(runtime_error!("LowMC doesn't support this block size"));
            }
        };
        let low_mc_op = CustomOperation::new(LowMC {
            s_boxes_per_round: 16,
            rounds: 11,
            block_size,
        });

        let input_data = g.input(input_t)?;
        let key = g.input(key_t)?;

        g.custom_op(low_mc_op, vec![input_data, key])
    })?;

    convert_main_graph_to_mpc(lowmc_context, context, vec![true, true])
}

// Convert key columns to binary and merge them for each input database
fn get_merging_graph(
    context: Context,
    column_types: ColumnTypesInfo,
    key_headers: &[String],
    is_private: bool,
) -> Result<Graph> {
    let mut headers_map = HashMap::new();
    for (h, t) in &column_types.headers_types {
        headers_map.insert((*h).clone(), (*t).clone());
    }

    let merging_context = simple_context(|g| {
        let data = g.input(column_types.get_plaintext_type())?;

        let num_entries = column_types.get_num_entries();

        let mut bit_columns = vec![];
        for header in key_headers {
            let t = headers_map.get(header).unwrap();

            let mut column = data.named_tuple_get((*header).clone())?;
            if t.has_mask() {
                column = column.tuple_get(1)?;
            }
            let mut bit_column = if t.get_scalar_type() != BIT {
                column.a2b()?
            } else {
                column
            };
            // Flatten all the bits per entry
            let flattened_shape = vec![num_entries, t.get_row_size_in_bits()];
            bit_column = bit_column.reshape(array_type(flattened_shape, BIT))?;
            // Pull out bits to simplify merging of columns
            bit_columns.push(bit_column);
        }
        // Merge key columns
        let merged_columns = if bit_columns.len() > 1 {
            g.concatenate(bit_columns, 1)?
        } else {
            bit_columns[0].clone()
        };

        Ok(merged_columns)
    })?;

    convert_main_graph_to_mpc(merging_context, context, vec![is_private])
}

// Multiply the column by the mask/null column
// Assumes that column mask is binary and it has the same number of rows as the column
fn apply_mask(column: Node, column_mask: Node, prf_keys: Node) -> Result<Node> {
    let t = get_column_type(&column)?;
    let column_shape = t.get_shape();
    // Reshape the mask to multiply row-wise
    let mut mask_shape = vec![column_shape[0]];
    if column_shape.len() > 1 {
        mask_shape.extend(vec![1; column_shape.len() - 1]);
    }
    let column_mask = reshape_shared_array(column_mask, array_type(mask_shape, BIT))?;

    if t.get_scalar_type() == BIT {
        multiply_mpc(column, column_mask, prf_keys, true)
    } else {
        mixed_multiply_mpc(column, column_mask, prf_keys)
    }
}

// Share a column if it is public
fn share_column(column: Node, prf_keys: Node) -> Result<Node> {
    if is_shared(&column)? {
        Ok(column)
    } else {
        let g = column.get_graph();
        let column_shares = get_node_shares(
            g.clone(),
            prf_keys,
            column.get_type()?,
            Some((column, IOStatus::Public)),
        )?;
        g.create_tuple(column_shares)
    }
}

enum Columns {
    Public(Vec<(String, Node)>),
    Shared(Vec<Vec<(String, Node)>>),
}

impl Columns {
    fn new_shared() -> Self {
        Columns::Shared(vec![vec![]; 3])
    }

    fn new_public() -> Self {
        Columns::Public(vec![])
    }

    // Attach a given shared column to shares of columns
    fn attach_column(&mut self, header: &str, column: Node) -> Result<()> {
        match self {
            Columns::Public(columns) => {
                if is_shared(&column)? {
                    return Err(runtime_error!("New columns should be public"));
                }
                columns.push((header.to_owned(), column));
            }
            Columns::Shared(shares) => {
                if is_shared(&column)? {
                    for (share_id, share) in shares.iter_mut().enumerate() {
                        share.push((header.to_owned(), column.tuple_get(share_id as u64)?));
                    }
                } else {
                    // If column is public, fake its sharing as (column, 0, 0).
                    let zero_share = zeros_like(column.clone())?;
                    for (share_id, share) in shares.iter_mut().enumerate() {
                        if share_id == 0 {
                            share.push((header.to_owned(), column.clone()));
                        } else {
                            share.push((header.to_owned(), zero_share.clone()));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn extract_and_attach(&mut self, header: &str, columns_shares: &DataShares) -> Result<()> {
        if columns_shares.column_has_mask(header) {
            self.attach_column(
                &columns_shares.get_mask_header(header)?,
                columns_shares.get_mask_column(header)?,
            )?;
        }
        self.attach_column(header, columns_shares.get_data_column(header)?)?;
        Ok(())
    }

    fn is_empty(&self) -> bool {
        match self {
            Columns::Public(v) => v.is_empty(),
            Columns::Shared(v) => v[0].is_empty(),
        }
    }

    fn into_node(self) -> Result<Node> {
        if self.is_empty() {
            return Err(runtime_error!("Can't turn empty shares into a node"));
        }
        match self {
            Columns::Public(columns) => {
                let g = columns[0].1.get_graph();
                g.create_named_tuple(columns)
            }
            Columns::Shared(shares) => {
                let g = shares[0][0].1.get_graph();
                let mut result_shares = vec![];
                for share_vec in shares {
                    let share = g.create_named_tuple(share_vec.clone())?;
                    result_shares.push(share);
                }
                g.create_tuple(result_shares)
            }
        }
    }
}

struct DataShares {
    // shares of named tuples
    shares: Vec<Node>,
    types_info: ColumnTypesInfo,
}

fn extract_shares(data_node: Node, types_info: ColumnTypesInfo) -> Result<DataShares> {
    let mut shares = vec![];
    if is_shared(&data_node)? {
        for share_id in 0..PARTIES as u64 {
            shares.push(data_node.tuple_get(share_id)?);
        }
    } else {
        shares.push(data_node);
    }
    Ok(DataShares { shares, types_info })
}

impl DataShares {
    fn is_public(&self) -> bool {
        self.shares.len() == 1
    }
    fn get_graph(&self) -> Graph {
        self.shares[0].get_graph()
    }

    fn get_data_column(&self, header: &str) -> Result<Node> {
        let column_t = self.types_info.get_column_type(header);
        let get_data_node = |column_node: Node| -> Result<Node> {
            if column_t.has_mask() {
                column_node.tuple_get(1)
            } else {
                Ok(column_node)
            }
        };
        if self.is_public() {
            let column_node = self.shares[0].named_tuple_get(header.into())?;
            get_data_node(column_node)
        } else {
            let mut shares_vec = vec![];
            for share_id in 0..PARTIES {
                let column_share = self.shares[share_id].named_tuple_get(header.into())?;
                shares_vec.push(get_data_node(column_share)?);
            }
            self.get_graph().create_tuple(shares_vec)
        }
    }

    fn get_mask_column(&self, header: &str) -> Result<Node> {
        let column_t = self.types_info.get_column_type(header);
        if !column_t.has_mask() {
            return Err(runtime_error!("Column has no mask"));
        }
        if self.is_public() {
            self.shares[0].named_tuple_get(header.into())?.tuple_get(0)
        } else {
            let mut shares_vec = vec![];
            for share_id in 0..PARTIES {
                let column_share = self.shares[share_id].named_tuple_get(header.into())?;
                shares_vec.push(column_share.tuple_get(0)?);
            }
            self.get_graph().create_tuple(shares_vec)
        }
    }

    fn get_mask_header(&self, header: &str) -> Result<String> {
        self.types_info.get_mask_header(header)
    }

    fn column_has_mask(&self, header: &str) -> bool {
        self.types_info.get_column_type(header).has_mask()
    }
}

/// Adds a node returning a join of a given type on given databases along given column keys.
///
/// Databases are represented as named tuples of integer arrays.
/// Each database should contain a special binary column named "null" that contains bits indicating whether the corresponding row has a zero content after previous operations (0 if yes).
/// Non-key column names must be unique in both databases.
///
/// A join of these named tuples is another named tuple containing a join of both input databases.
/// Namely, it contains only the database rows whose values match in given key columns according to the join type.
/// The content of non-key columns is attached to these rows from both sets.
///
/// The protocol follows the description of the Join protocol from <https://eprint.iacr.org/2019/518.pdf>.
/// Let X be the first database and Y be the second one.
/// 1. Key columns of both sets are converted to binary and merged row-wise.
/// 2. If the bitsize of merged entries is bigger than the block size of the LowMC block cipher, hash them via multiplication by a random matrix obliviously generated by all parties.
/// 3. Compute the oblivious pseudo random function (OPRF) on the merged columns of both sets using the LowMC block cipher with a random key obliviously generated by all parties.
/// This operation returns random string on entries with zero values in the "null" column, i.e.
///
/// OPRF(S) = (PRF(key columns of S) - R) * S_null_column XOR R where R is a random matrix obliviously  generated by all parties.
///
/// 4. OPRF(X) is revealed to party 2.
/// 5. OPRF(Y) is revealed to party 1.
/// 6. Parties 1 and 2 sample 3 hash functions that they will use for hashing using their common PRF key (key 2 in the multiplication PRF key triple).
/// 7. Party 1 computes a Cuckoo hash map from OPRF(Y) using the above hash functions and randomizes it to a permutation.
/// 8. All parties attach merged key columns of Y to Y and get Y'.
/// 9. All parties pad Y' with obliviously sampled random strings such that the number of entries in Y' is equal to the length of the Cuckoo map created in step 7.
/// 10. Parties 0 and 1 convert 2-out-of-3 shares of Y' to 2-out-of-2 shares.
/// 11. Parties 0 and 1 create a Cuckoo table of Y by applying the above Cuckoo permutation to the 2-out-of-2 shares of Y' using the Permutation protocol (PermutationMPC).
/// The Cuckoo table will be shared between parties 2 (share 0) and 1 (share 1).
/// 12. Party 2 computes a simple hash map of OPRF(X) using the hash functions generated in step 6.
/// 13. For each simple hash map h, parties 2 and 1 perform the Switching protocol (SwitchingMPC) to get 2-out-of-2 shares of Y_h, which is an arrangement of several Cuckoo table elements such that elements of the intersection are located at the same positions as elements of X belonging to the intersection.
/// As a result, Parties 2 and 0 have 2-out-of-2 shares of Y_h.
/// 14. All parties convert the 2-out-of-2 shares of each Y_h to 2-out-of-3 shares.
/// 15. Compare X with all Y_h row-wise and select the rows of Y_h that match rows in X according to the join type.
/// 16. Combine the selected rows along the columns of X and Y.
///
/// # Custom operation arguments
///
/// - a named tuple containing the first database
/// - a named tuple containing the second database
/// - a tuple of PRF keys for multiplication
///
/// # Custom operation returns
///
/// Node containing a named tuple containing the join of both databases
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct JoinMPC {
    // Type of join
    pub join_t: JoinType,
    // Instead of HashMap, Vector is used to support the Hash trait
    pub headers: Vec<(String, String)>,
    // Indicates whether non-null columns have masks
    pub has_column_masks: bool,
}

impl JoinMPC {
    // FullJoin (X, Y) = Union(X, LeftJoin(Y, X))
    fn create_full_join_graph(
        &self,
        context: Context,
        data_x_t: Type,
        data_y_t: Type,
        prf_t: Type,
    ) -> Result<Graph> {
        let g = context.create_graph()?;
        let data_x = g.input(data_x_t)?;
        let data_y = g.input(data_y_t)?;
        let prf_keys = g.input(prf_t)?;
        // The headers of the left join are the reversed input headers.
        let mut left_join_headers = vec![];
        for (h0, h1) in &self.headers {
            left_join_headers.push((h1.clone(), h0.clone()));
        }
        let left_join = g.custom_op(
            CustomOperation::new(JoinMPC {
                join_t: JoinType::Left,
                headers: left_join_headers,
                has_column_masks: self.has_column_masks,
            }),
            vec![data_y, data_x.clone(), prf_keys.clone()],
        )?;
        let res = g.custom_op(
            CustomOperation::new(JoinMPC {
                join_t: JoinType::Union,
                headers: self.headers.clone(),
                has_column_masks: self.has_column_masks,
            }),
            vec![data_x, left_join, prf_keys],
        )?;
        res.set_as_output()?;
        g.finalize()
    }
}

fn check_and_extract_dataset_parameters(
    t: Type,
    is_private: bool,
    has_mask_columns: bool,
) -> Result<ColumnTypesInfo> {
    let headers_types = if is_private {
        if !is_type_shared(&t) {
            return Err(runtime_error!("Private database must be a tuple of shares"));
        }

        let t_vec = get_types_vector(t)?;

        check_private_tuple(t_vec.clone())?;
        t_vec[0].get_named_types()?
    } else {
        t.get_named_types()?
    };

    let mut column_header_types = vec![];
    for (h, t) in headers_types {
        column_header_types.push((h.clone(), ColumnType::new(t.clone(), has_mask_columns, &h)?));
    }

    let column_types_info = ColumnTypesInfo::new(column_header_types);

    Ok(column_types_info)
}

#[typetag::serde]
impl CustomOperationBody for JoinMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        if argument_types.len() == 2 {
            if argument_types[0].is_named_tuple() && argument_types[1].is_named_tuple() {
                let g = context.create_graph()?;
                let set0 = g.input(argument_types[0].clone())?;
                let set1 = g.input(argument_types[1].clone())?;
                let mut headers = HashMap::new();
                for (h0, h1) in &self.headers {
                    headers.insert((*h0).clone(), (*h1).clone());
                }
                if self.has_column_masks {
                    set0.join_with_column_masks(set1, self.join_t, headers)?
                        .set_as_output()?;
                } else {
                    set0.join(set1, self.join_t, headers)?.set_as_output()?;
                }
                g.finalize()?;
                return Ok(g);
            } else {
                // Panics since:
                // - the user has no direct access to this function.
                // - the MPC compiler should pass correct arguments
                // and this panic should never happen.
                panic!("Inconsistency with type checker");
            }
        }
        if argument_types.len() != 3 {
            return Err(runtime_error!("Join protocol should have 3 inputs"));
        }

        let data_x_t = argument_types[0].clone();
        let data_y_t = argument_types[1].clone();
        let prf_t = argument_types[2].clone();

        let is_x_private = is_type_shared(&data_x_t);
        let is_y_private = is_type_shared(&data_y_t);

        let column_header_types_x = check_and_extract_dataset_parameters(
            data_x_t.clone(),
            is_x_private,
            self.has_column_masks,
        )?;
        let num_entries_x = column_header_types_x.get_num_entries();
        let column_header_types_y = check_and_extract_dataset_parameters(
            data_y_t.clone(),
            is_y_private,
            self.has_column_masks,
        )?;
        let num_entries_y = column_header_types_y.get_num_entries();

        // FullJoin (X, Y) = Union(set0, LeftJoin(Y, X))
        if self.join_t == JoinType::Full {
            return self.create_full_join_graph(context, data_x_t, data_y_t, prf_t);
        }

        // Name of the "key" column containing bits of compared columns
        // To avoid a collision with input headers, the key header is the join of all input headers
        let mut all_headers: Vec<String> = column_header_types_x
            .headers_types
            .iter()
            .map(|v| v.0.clone())
            .collect();
        let headers_y: Vec<String> = column_header_types_y
            .headers_types
            .iter()
            .map(|v| v.0.clone())
            .collect();
        all_headers.extend(headers_y);
        // This reduces the collision probability with a non-key header
        all_headers.push("merged-key-columns-750a7826-a23c-11ed-b543-9f91976d37c4".to_owned());
        let key_header = all_headers.join("-");

        let mut headers_map = HashMap::new();
        for (h0, h1) in &self.headers {
            headers_map.insert((*h0).clone(), (*h1).clone());
        }

        let mut key_headers_x = vec![];
        let mut key_headers_y = vec![];
        for (h_x, h_y) in &self.headers {
            key_headers_x.push((*h_x).clone());
            key_headers_y.push((*h_y).clone());
        }

        // When computing full join, we call union join on datasets with repeating non-key headers.
        let mut same_non_key_headers = vec![];
        if self.join_t == JoinType::Union {
            for (h, _) in &column_header_types_x.headers_types {
                if h != NULL_HEADER
                    && !key_headers_y.contains(h)
                    && column_header_types_y.headers_map.contains_key(h)
                {
                    same_non_key_headers.push((*h).clone());
                }
            }
        }

        // Compute the bit length of one entry containing only key columns.
        // This value is the same for both input sets.
        // In addition, checks whether non-binary key columns are present.
        // This defines the merging graphs below need PRF keys.
        let mut key_columns_entry_bitlength = 0;
        let mut is_a2b_needed = false;
        for (header, t) in &column_header_types_x.headers_types {
            if key_headers_x.contains(header) {
                let column_entry_bitlength = t.get_row_size_in_bits();
                key_columns_entry_bitlength += column_entry_bitlength;
                if t.get_scalar_type() != BIT {
                    is_a2b_needed = true;
                }
            }
        }
        let prf_needed_to_merge_x = is_x_private && is_a2b_needed;
        let prf_needed_to_merge_y = is_y_private && is_a2b_needed;
        // Graph that merges the key columns of the dataset X
        let merging_g_x = get_merging_graph(
            context.clone(),
            column_header_types_x.clone(),
            &key_headers_x,
            is_x_private,
        )?;
        // Graph that merges the key columns of the dataset Y
        let merging_g_y = get_merging_graph(
            context.clone(),
            column_header_types_y.clone(),
            &key_headers_y,
            is_y_private,
        )?;

        // Graph that computes the product of the null column and key column masks of a given dataset.
        // This is needed by OPRF to randomize encoding of rows with empty entries in key columns.
        let (oprf_mask_g_x, oprf_mask_x_needs_prf) = get_oprf_mask_graph(
            context.clone(),
            &column_header_types_x,
            &key_headers_x,
            is_x_private,
            self.has_column_masks,
        )?;
        let (oprf_mask_g_y, oprf_mask_y_needs_prf) = get_oprf_mask_graph(
            context.clone(),
            &column_header_types_y,
            &key_headers_y,
            is_y_private,
            self.has_column_masks,
        )?;

        // Graph that computes LowMC on the dataset X
        let lowmc_g_x = get_lowmc_graph(
            context.clone(),
            array_type(vec![num_entries_x, PRF_OUTPUT_SIZE], BIT),
            array_type(vec![LOW_MC_KEY_SIZE], BIT),
        )?;
        // Graph that computes LowMC on the dataset Y
        let lowmc_g_y = get_lowmc_graph(
            context.clone(),
            array_type(vec![num_entries_y, PRF_OUTPUT_SIZE], BIT),
            array_type(vec![LOW_MC_KEY_SIZE], BIT),
        )?;
        // Graph that compares null and merged key columns of X and Y contained in several named tuples called Y_h.
        // For inner and left join, Y_h contains full rows of Y such that they can be later extracted and attached to the result.
        // In union join, the rows of Y can be extracted directly from Y.
        let mut y_h_types = vec![(
            key_header.clone(),
            array_type(vec![num_entries_x, key_columns_entry_bitlength], BIT),
        )];
        match self.join_t {
            JoinType::Inner | JoinType::Left => {
                for (header, t) in &column_header_types_y.headers_types {
                    let column_t = t.clone_with_number_of_entries(num_entries_x);
                    // extract mask and data
                    if column_t.has_mask() {
                        y_h_types.push((
                            column_header_types_y.get_mask_header(header)?,
                            column_t.get_mask_type()?,
                        ));
                    }
                    y_h_types.push(((*header).clone(), column_t.get_data_type()));
                }
            }
            JoinType::Union => {
                y_h_types.push((NULL_HEADER.to_owned(), array_type(vec![num_entries_x], BIT)));
            }
            JoinType::Full => {
                panic!("Shouldn't be here");
            }
        }

        let y_h_type = named_tuple_type(y_h_types.clone());
        let merged_key_columns_x_type = named_tuple_type(vec![
            (NULL_HEADER.to_owned(), array_type(vec![num_entries_x], BIT)),
            (
                key_header.clone(),
                array_type(vec![num_entries_x, key_columns_entry_bitlength], BIT),
            ),
        ]);
        let eq_g = get_equality_graph(
            context.clone(),
            y_h_type,
            merged_key_columns_x_type,
            key_header.clone(),
            true,
            is_x_private,
        )?;
        // Graph that selects rows of Y_h according to the given mask
        let select_g_y = get_select_graph(
            context.clone(),
            y_h_types,
            num_entries_x,
            key_header.clone(),
        )?;

        // Main graph computing PSI
        let g = context.create_graph()?;

        let data_x = g.input(data_x_t)?;
        let data_y = g.input(data_y_t)?;
        let prf_keys = g.input(prf_t)?;

        // Extract input shares
        let data_x_shares = extract_shares(data_x.clone(), column_header_types_x.clone())?;
        let data_y_shares = extract_shares(data_y.clone(), column_header_types_y.clone())?;

        // Extract PRF keys
        let mut prf_keys_vec = vec![];
        for key_id in 0..PARTIES as u64 {
            prf_keys_vec.push(prf_keys.tuple_get(key_id)?);
        }

        // 1. Key columns of both sets are converted to binary and merged row-wise.
        let merged_columns_x = g.call(
            merging_g_x,
            if prf_needed_to_merge_x {
                vec![prf_keys.clone(), data_x.clone()]
            } else {
                vec![data_x.clone()]
            },
        )?;
        let merged_columns_y = g.call(
            merging_g_y,
            if prf_needed_to_merge_y {
                vec![prf_keys.clone(), data_y.clone()]
            } else {
                vec![data_y.clone()]
            },
        )?;

        // 2. If the bitsize of merged entries is bigger than the block size of the LowMC block cipher, hash them via multiplication by a random matrix obliviously generated by all parties.
        //  - Generate a random matrix shared by all the parties
        let random_hash_matrix = generate_shared_random_array(
            array_type(vec![PRF_OUTPUT_SIZE, key_columns_entry_bitlength], BIT),
            &prf_keys_vec,
        )?;

        // 3. Compute the oblivious pseudo random function (OPRF) on the merged columns of both sets using the LowMC block cipher with a random key obliviously generated by all parties.
        // This operation returns random string on entries with zero values in the "null" column, i.e.
        //
        // OPRF(S) = (PRF(key columns of S) - R) * S_null_column XOR R where R is a random matrix obliviously  generated by all parties.
        let oprf_key =
            generate_shared_random_array(array_type(vec![LOW_MC_KEY_SIZE], BIT), &prf_keys_vec)?;

        let compute_oprf = |merged_columns: Node,
                            null_column: Node,
                            lowmc_graph: Graph,
                            num_entries: u64|
         -> Result<Node> {
            let hashed_columns = gemm_mpc(
                merged_columns,
                random_hash_matrix.clone(),
                false,
                true,
                prf_keys.clone(),
                true,
            )?;

            let oprf_set = g.call(
                lowmc_graph,
                vec![prf_keys.clone(), hashed_columns, oprf_key.clone()],
            )?;
            let r = generate_shared_random_array(
                array_type(vec![num_entries, PRF_OUTPUT_SIZE], BIT),
                &prf_keys_vec,
            )?;
            add_mpc(
                multiply_mpc(
                    subtract_mpc(oprf_set, r.clone())?,
                    reshape_shared_array(null_column, array_type(vec![num_entries, 1], BIT))?,
                    prf_keys.clone(),
                    false, // Don't reshare this multiply as the results are later revealed
                )?,
                r,
            )
        };

        // Compute OPRF(X) = (PRF(key columns of X) - R_X) * X_OPRF_mask XOR R_X where R_X is a random matrix generated by all parties
        let oprf_mask_x = if oprf_mask_x_needs_prf {
            g.call(oprf_mask_g_x, vec![prf_keys.clone(), data_x])?
        } else {
            g.call(oprf_mask_g_x, vec![data_x])?
        };
        let oprf_set_x = compute_oprf(
            merged_columns_x.clone(),
            oprf_mask_x,
            lowmc_g_x,
            num_entries_x,
        )?;

        // Compute OPRF(Y) = (PRF(key columns of Y) - R_Y) * Y_OPRF_mask XOR R_Y where R_Y is a random matrix generated by all parties
        let oprf_mask_y = if oprf_mask_y_needs_prf {
            g.call(oprf_mask_g_y, vec![prf_keys.clone(), data_y])?
        } else {
            g.call(oprf_mask_g_y, vec![data_y])?
        };
        let oprf_set_y = compute_oprf(
            merged_columns_y.clone(),
            oprf_mask_y,
            lowmc_g_y,
            num_entries_y,
        )?;

        // 4. Reveal OPRF(X) to party 2
        let revealed_oprf_set_x = reveal_array_from_3_out_of_3(oprf_set_x, 2)?;
        // 5. Reveal OPRF(Y) to party 1
        let revealed_oprf_set_y = reveal_array_from_3_out_of_3(oprf_set_y, 1)?;

        // 6. Parties 1 and 2 generate random matrices for hashing of shape [3, m, LOW_MC_BLOCK_SIZE],
        // where m = ceil(log(max(num_entries_y, num_entries_x)+1).
        // TODO: quantify probability of success of Cuckoo hashing with these parameters
        let max_num_entries = num_entries_y.max(num_entries_x);
        let log_num_cuckoo_entries = if max_num_entries >= 512 {
            // here we use the empirical results from the paper, see Appendix B, https://eprint.iacr.org/2018/579.pdf.
            // This should result in failure probability smaller than 2^(-40).
            ((max_num_entries as f64).log2() + 1f64).ceil() as u64
        } else {
            // The approximation from the paper fails here.
            // Therefore, we assume the conservative failure probability equal to the probability that there is a pair of elements with all hashes equal.
            // This event is dominant over other failure events such as the existense of 3 elements hashed to only 2 values, 4 elements hashed to 3 etc.
            ((max_num_entries as f64).log2() + 7f64).ceil() as u64
        };
        let num_hash_functions = 3;
        let hash_matrices = prf_keys_vec[2].prf(
            0,
            array_type(
                vec![num_hash_functions, log_num_cuckoo_entries, PRF_OUTPUT_SIZE],
                BIT,
            ),
        )?;

        // 7. Party 1 computes a Cuckoo hash map from OPRF(Y) and randomizes it to a permutation
        let cuckoo_map = revealed_oprf_set_y.cuckoo_hash(hash_matrices.clone())?;
        let cuckoo_permutation = cuckoo_map.cuckoo_to_permutation()?;

        // 8. Attach the merged key columns to Y
        // HACK: If Y is public, we create fake shares containing zeros such that the next operation generating random padding can accept it
        let mut extended_y_shares = Columns::new_shared();
        extended_y_shares.attach_column(&key_header, merged_columns_y)?;
        match self.join_t {
            JoinType::Inner | JoinType::Left => {
                // Attach all the columns of Y as they're later extracted from Y_h
                for (header, _) in &column_header_types_y.headers_types {
                    extended_y_shares.extract_and_attach(header, &data_y_shares)?;
                }
            }
            JoinType::Union => {
                // The resulting rows are extracted directly from Y, so only the null column should be attached to Y_h
                extended_y_shares.extract_and_attach(NULL_HEADER, &data_y_shares)?;
            }
            JoinType::Full => {
                panic!("Shouldn't be here");
            }
        }
        let extended_y = extended_y_shares.into_node()?;

        // 9. Pad columns of Y with random data such that the number of entries is equal to the cuckoo table size
        let padded_shares_y = {
            let num_extra_rows = (1 << log_num_cuckoo_entries) - num_entries_y;
            random_pad_columns(extended_y, num_extra_rows, &prf_keys_vec)?
        };

        // 10. Switch from 2-out-of-3 shares of dataset Y to 2-out-of-2 shares owned by parties 0 and 1
        let data_y_2of2shares = {
            // Share of party 0 is the sum of its 2-out-of-3 shares
            let party0_share =
                sum_named_columns(padded_shares_y.tuple_get(0)?, padded_shares_y.tuple_get(1)?)?;
            // Share of party 1 is the third 2-out-of-3 share
            // Share of party 1 goes first to support the contract of the consecutive PermutationMPC operation, which demands that the first share and a permutation is owned by the same party.
            g.create_tuple(vec![padded_shares_y.tuple_get(2)?, party0_share])?
        };

        // 11. Create a Cuckoo table of Y by applying the above Cuckoo permutation to the shares of Y.
        // The Cuckoo table will be shared between parties 1 (share 0) and 2 (share 1).
        let mut cuckoo_table = g.custom_op(
            CustomOperation::new(PermutationMPC {
                programmer_id: 1,
                sender_id: 0,
            }),
            vec![data_y_2of2shares, cuckoo_permutation, prf_keys.clone()],
        )?;

        // 12. Party 2 computes a simple hash map from OPRF(X) for each of 3 hash functions
        let simple_hash_map = g.custom_op(
            CustomOperation::new(SimpleHash {}),
            vec![revealed_oprf_set_x, hash_matrices],
        )?;

        // 13. For each simple hash map h, parties 2 and 1 perform the switching protocol to get 2-out-of-2 shares of Y_h, which is an arrangement of several Cuckoo table elements such that elements of the intersection are located at the same positions as elements of X belonging to the intersection.
        // As a result, Parties 2 and 0 have 2-out-of-2 shares of Y_h

        // Repack the Cuckoo table such that party 2 has share 0 and party has share 1
        // This is necessary by the contract of SwitchingMPC that requires the first share to be given by Programmer (party 2 having the switching map)
        cuckoo_table =
            g.create_tuple(vec![cuckoo_table.tuple_get(1)?, cuckoo_table.tuple_get(0)?])?;

        let mut all_y_h = vec![];
        for h in 0..num_hash_functions {
            let switch_map = simple_hash_map.get(vec![h])?;
            let switched_cuckoo = g.custom_op(
                CustomOperation::new(SwitchingMPC {
                    sender_id: 1,
                    programmer_id: 2,
                }),
                vec![cuckoo_table.clone(), switch_map, prf_keys.clone()],
            )?;
            all_y_h.push(switched_cuckoo);
        }

        // 14. Convert the 2-out-of-2 shares of Y_h to 2-out-of-3 shares
        let y_h_shares = {
            let mut res = vec![];
            // Tuple of two named tuples
            let y_h_t = all_y_h[0].get_type()?;
            // One named tuple corresponding to one 2-out-of-2 share
            let share_2outof2_t = (*get_types_vector(y_h_t)?[0]).clone();
            for y_h in all_y_h {
                // Parties generate a 3-out-of-3 sharing of zero
                let zero_shares =
                    get_zero_shares(g.clone(), prf_keys.clone(), share_2outof2_t.clone())?;
                // Party 2 adds its input share (share 0 of Y_h) to the zero share and sends the result to Party 1
                let third_share = sum_named_columns(y_h.tuple_get(0)?, zero_shares[2].clone())?
                    .nop()?
                    .add_annotation(NodeAnnotation::Send(2, 1))?;
                // Party 0 adds its input share ((share 1 of Y_h)) to the zero share and sends the result to Party 2
                let first_share = sum_named_columns(y_h.tuple_get(1)?, zero_shares[0].clone())?
                    .nop()?
                    .add_annotation(NodeAnnotation::Send(0, 2))?;
                // Party 1 sends its share of zero to Party 0
                let second_share = zero_shares[1]
                    .nop()?
                    .add_annotation(NodeAnnotation::Send(1, 0))?;
                // Create 2-out-of-3 shares of one Y_h
                let y_h_share = g.create_tuple(vec![first_share, second_share, third_share])?;
                res.push(y_h_share);
            }
            res
        };

        // 15. Compare X with all Y_h and select the rows of Y_h that match rows in X according to the join type.

        // Attach the null column to the merged key columns of X.
        let mut null_merged_columns_x_shares = if is_x_private {
            Columns::new_shared()
        } else {
            Columns::new_public()
        };
        let null_x = data_x_shares.get_data_column(NULL_HEADER)?;
        null_merged_columns_x_shares.attach_column(NULL_HEADER, null_x.clone())?;
        null_merged_columns_x_shares.attach_column(&key_header, merged_columns_x)?;
        let null_merged_columns_x = null_merged_columns_x_shares.into_node()?;

        // Compare first Y_h with key columns of X
        let mut match_bits = g.call(
            eq_g.clone(),
            vec![
                prf_keys.clone(),
                y_h_shares[0].clone(),
                null_merged_columns_x.clone(),
            ],
        )?;
        // There is no need to select columns from Y_h for union join
        let mut selected_columns_y = match self.join_t {
            JoinType::Inner | JoinType::Left => {
                let node = g.call(
                    select_g_y.clone(),
                    vec![prf_keys.clone(), y_h_shares[0].clone(), match_bits.clone()],
                )?;
                let mut node_vec = vec![];
                for share_id in 0..PARTIES as u64 {
                    node_vec.push(node.tuple_get(share_id)?);
                }
                node_vec
            }
            JoinType::Union => vec![],
            JoinType::Full => {
                panic!("Shouldn't be here");
            }
        };
        for shares in y_h_shares.iter().skip(1) {
            // Compare elements of Y_h and X
            let eq_bits = g.call(
                eq_g.clone(),
                vec![
                    prf_keys.clone(),
                    (*shares).clone(),
                    null_merged_columns_x.clone(),
                ],
            )?;
            // Compute selection bits.
            // Selection bits must satisfy the following rules:
            // - if the current match bit is 0 (the corresponding entry of X hasn't been matched) and the corresponding equality bit is 1 (matching occurred in this iteration), then the corresponding selection bit should be 1;
            // - in other cases, the selection bit must be 0.
            // This can be computed as select_bit = eq_bit AND match_bits XOR eq_bit.
            let select_bits = add_mpc(
                multiply_mpc(eq_bits.clone(), match_bits.clone(), prf_keys.clone(), true)?,
                eq_bits.clone(),
            )?;

            match self.join_t {
                JoinType::Inner | JoinType::Left => {
                    // Select rows of Y_h
                    let selected_rows = g.call(
                        select_g_y.clone(),
                        vec![prf_keys.clone(), (*shares).clone(), select_bits.clone()],
                    )?;
                    // Sum named tuples
                    selected_columns_y = {
                        let mut columns_shares = vec![];
                        for (share_id, selected_column) in selected_columns_y.iter().enumerate() {
                            let share = sum_named_columns(
                                selected_rows.tuple_get(share_id as u64)?,
                                selected_column.clone(),
                            )?;
                            columns_shares.push(share);
                        }
                        columns_shares
                    };
                }
                JoinType::Union => (),
                JoinType::Full => {
                    panic!("Shouldn't be here");
                }
            }
            // match bits are equal to OR of eq_bits over all Y_h
            // Namely, we have new match_bits = match_bits OR eq_bits = match_bits * eq_bits + match_bits + eq_bits = match_bits + select_bits
            match_bits = add_mpc(match_bits, select_bits)?;
        }

        // 16. Combine the selected rows along the columns of X and Y
        let mut result_shares = Columns::new_shared();
        // Compute the null column and attach to the result
        let null_y = data_y_shares.get_data_column(NULL_HEADER)?;
        let res_null_column = match self.join_t {
            JoinType::Inner => {
                // The resulting null column is equal to match bits indicating intersection elements.
                result_shares.attach_column(NULL_HEADER, match_bits.clone())?;
                match_bits
            }
            JoinType::Left => {
                // The resulting null column is equal to that of X.
                // Note that X might be public, so we might need to share it.
                let null_x_shared = share_column(null_x, prf_keys.clone())?;
                result_shares.attach_column(NULL_HEADER, null_x_shared.clone())?;
                null_x_shared
            }
            JoinType::Union => {
                // The resulting null column should consist of two parts:
                // - the null column of X with zeros in the intersection rows.
                //   Note that if null_x[i] = 0, then match_bits[i] = 0 with overwhelming probability.
                //   Thus, res_null[i] = match_bits[i] + null_x[i] satisfies the above condition.
                // - the null column of Y
                let res_null_half_x = add_mpc(match_bits, null_x)?;
                let res_null_half_y = share_column(null_y.clone(), prf_keys.clone())?;

                // Concatenate both halfs of the resulting null column
                let res_null = concatenate_mpc(&[res_null_half_x, res_null_half_y], 0)?;
                result_shares.attach_column(NULL_HEADER, res_null.clone())?;
                res_null
            }
            JoinType::Full => {
                panic!("Shouldn't be here");
            }
        };

        // Add resulting data from the columns of X
        // This closure handles both data and mask columns
        let get_x_mask_or_data = |header_x: &String, is_mask: bool| -> Result<Node> {
            let get_column = |header: &String, data_shares: &DataShares| -> Result<Node> {
                if is_mask {
                    data_shares.get_mask_column(header)
                } else {
                    data_shares.get_data_column(header)
                }
            };

            let mut column = match self.join_t {
                JoinType::Inner | JoinType::Left => get_column(header_x, &data_x_shares)?,
                JoinType::Union => {
                    // For the union, we need to concatenate rows of X and Y appropriately
                    if key_headers_x.contains(header_x) {
                        // If the column is key, concatenate it with the related column of Y
                        let column_x = get_column(header_x, &data_x_shares)?;
                        let header_y = &headers_map[header_x];
                        let column_y = get_column(header_y, &data_y_shares)?;
                        concatenate_mpc(&[column_x, column_y], 0)?
                    } else if same_non_key_headers.contains(header_x) {
                        // If the header is non-key and appears in both sets, concatenate the related columns
                        let column_x = get_column(header_x, &data_x_shares)?;
                        let column_y = get_column(header_x, &data_y_shares)?;
                        concatenate_mpc(&[column_x, column_y], 0)?
                    } else {
                        // If the column is non-key and unique, append it with zeros
                        let column = get_column(header_x, &data_x_shares)?;
                        zero_pad_column(column, num_entries_y, false, prf_keys.clone())?
                    }
                }
                JoinType::Full => {
                    panic!("Shouldn't be here");
                }
            };
            // Mask out rows
            column = apply_mask(column, res_null_column.clone(), prf_keys.clone())?;
            // X might be public, so we might need to share it.
            share_column(column, prf_keys.clone())
        };

        for (header_x, column_t) in &column_header_types_x.headers_types {
            // the null column is already added, the merged key column shouldn't show up in the result
            if header_x == NULL_HEADER || header_x == &key_header {
                continue;
            }
            let data_column = get_x_mask_or_data(header_x, false)?;
            if column_t.has_mask() {
                let mask_column = get_x_mask_or_data(header_x, true)?;
                let data_column = apply_mask(data_column, mask_column.clone(), prf_keys.clone())?;
                result_shares
                    .attach_column(header_x, merge_mask_data(mask_column, data_column)?)?;
            } else {
                result_shares.attach_column(header_x, data_column)?;
            }
        }
        // Attach non-key columns of Y
        // This closure handles both data and mask columns
        let get_y_mask_or_data = |header: &str, is_mask: bool| -> Result<Node> {
            match self.join_t {
                // For inner and left joins, columns of Y are precomputed from Y_h's
                JoinType::Inner | JoinType::Left => {
                    if is_mask {
                        let mask_header = column_header_types_y.get_mask_header(header)?;
                        get_column(&selected_columns_y, mask_header)
                    } else {
                        get_column(&selected_columns_y, header.into())
                    }
                }
                JoinType::Union => {
                    // For union, we need to prepend columns of Y by zeros
                    let mut column = if is_mask {
                        data_y_shares.get_mask_column(header)?
                    } else {
                        data_y_shares.get_data_column(header)?
                    };
                    column = apply_mask(column, null_y.clone(), prf_keys.clone())?;
                    // Prepend the column with zeros
                    column = zero_pad_column(column, num_entries_x, true, prf_keys.clone())?;
                    // Y might be public, so we might need to share it
                    share_column(column, prf_keys.clone())
                }
                JoinType::Full => {
                    panic!("Shouldn't be here");
                }
            }
        };

        for (header_y, column_t) in &column_header_types_y.headers_types {
            // If the current column has been already attached to the result, ignore it
            if key_headers_y.contains(header_y)
                || NULL_HEADER == header_y
                || same_non_key_headers.contains(header_y)
            {
                continue;
            }
            let data_column = get_y_mask_or_data(header_y, false)?;
            if column_t.has_mask() {
                let mask_column = get_y_mask_or_data(header_y, true)?;
                let data_column = apply_mask(data_column, mask_column.clone(), prf_keys.clone())?;
                result_shares
                    .attach_column(header_y, merge_mask_data(mask_column, data_column)?)?;
            } else {
                result_shares.attach_column(header_y, data_column)?;
            }
        }

        // Collect the resulting columns
        let result = result_shares.into_node()?;
        result.set_as_output()?;

        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!(
            "Join(type:{:?},keys:{:?},has-column-masks:{})",
            self.join_t, self.headers, self.has_column_masks
        )
    }
}

/// Adds a node returning hash values of an input array of binary strings using provided hash functions.
///
/// Hash functions are defined as an array of binary matrices.
/// The hash of an input string is a product of one of these matrices and this string.
/// Hence, the last dimension of these matrices should coincide with the length of input strings.
///
/// If the input array has shape `[..., n, b]` and hash matrices are given as an `[h, m, b]`-array,
/// then the hash map is an array of shape `[..., h, 2^m]`.
/// The hash table element with index `[..., h, i]` is equal to `j` if the `[..., i]`-th `b`-bit input string is hashed to `j` by the `h`-th hash function.
///
/// When used within a PSI protocol, the hash functions should be the same as those used for Cuckoo hashing.    
///
/// **WARNING**: this function should not be used before MPC compilation.
///
/// # Custom operation arguments
///
/// - input array of binary strings of shape [..., n, b]
/// - random binary [h, m, b]-matrix.
///
/// # Custom operation returns
///
/// hash table of shape [..., h, 2^m] containing UINT64 elements
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
struct SimpleHash;

#[typetag::serde]
impl CustomOperationBody for SimpleHash {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        if argument_types.len() != 2 {
            // Panics since:
            // - the user has no direct access to this function.
            // - the MPC compiler should pass the correct number of arguments
            // and this panic should never happen.
            return Err(runtime_error!("SimpleHash should have 2 inputs."));
        }

        let input_type = argument_types[0].clone();
        let hash_type = argument_types[1].clone();

        if !matches!(input_type, Type::Array(_, BIT)) {
            return Err(runtime_error!(
                "SimpleHash can't be applied to a non-binary arrays"
            ));
        }
        let input_shape = input_type.get_shape();
        if input_shape.len() < 2 {
            return Err(runtime_error!(
                "Input shape must have at least 2 dimensions"
            ));
        }
        if !matches!(hash_type, Type::Array(_, BIT)) {
            return Err(runtime_error!(
                "SimpleHash needs a binary array as a hash matrix"
            ));
        }
        let hash_shape = hash_type.get_shape();
        if hash_shape.len() != 3 {
            return Err(runtime_error!("Hash array should have 3 dimensions"));
        }
        if hash_shape[1] > 63 {
            return Err(runtime_error!(
                "Hash map is too big. Decrease the number of rows of hash matrices"
            ));
        }
        let input_element_length = input_shape[input_shape.len() - 1];
        if hash_shape[2] != input_element_length {
            return Err(runtime_error!(
                "Hash matrix accepts bitstrings of length {}, but input strings are of length {}",
                hash_shape[2],
                input_element_length
            ));
        }

        let g = context.create_graph()?;

        let input_array = g.input(input_type.clone())?;
        let hash_matrices = g.input(hash_type.clone())?;

        let hash_shape = hash_type.get_shape();

        // For each subarray and for each hash function, the output hash map contains hashes of input bit strings
        let input_shape = input_type.get_shape();
        let mut single_hash_table_shape = input_shape[0..input_shape.len() - 1].to_vec();
        single_hash_table_shape.push(hash_shape[1]);

        // Multiply hash matrices of shape [h, m, b] by input strings of shape [..., n, b].
        // In Einstein notation, ...nb, hmb -> ...hnm.

        let mut extended_shape = input_type.get_shape();
        extended_shape.insert(extended_shape.len() - 1, 1);

        // Change the shape of hash_matrices from [h, m, b] to [h*m, b]
        let hash_matrices_for_sum = hash_matrices.reshape(array_type(
            vec![hash_shape[0] * hash_shape[1], hash_shape[2]],
            BIT,
        ))?;
        // The result shape is [..., n, h*m]
        let mut hash_tables = input_array.gemm(hash_matrices_for_sum, false, true)?;

        // Reshape to [..., n, h, m]
        let mut split_by_hash_shape = input_shape[0..input_shape.len() - 1].to_vec();
        split_by_hash_shape.extend_from_slice(&hash_shape[0..2]);
        hash_tables = hash_tables.reshape(array_type(split_by_hash_shape.clone(), BIT))?;

        // Transpose to [..., h, n, m]
        let len_output_shape = split_by_hash_shape.len() as u64;
        let mut permuted_axes: Vec<u64> = (0..len_output_shape).collect();
        permuted_axes[len_output_shape as usize - 3] = len_output_shape - 2;
        permuted_axes[len_output_shape as usize - 2] = len_output_shape - 3;
        hash_tables = hash_tables.permute_axes(permuted_axes)?;

        let num_zeros = 64 - hash_shape[1];
        let res = extend_with_zeros(&g, hash_tables, num_zeros, false)?;

        res.b2a(UINT64)?.set_as_output()?;

        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        "SimpleHash".to_owned()
    }
}

// Checks inputs of permutation, duplication and switching network maps and returns the number of entries and a vector of column types.
fn check_and_extract_map_input_parameters(
    argument_types: &[Type],
    sender_id: u64,
    programmer_id: u64,
) -> Result<(u64, Vec<(String, ColumnType)>)> {
    if argument_types.len() != 3 {
        return Err(runtime_error!("This map should have 3 input types"));
    }
    let shares_t = argument_types[0].clone();
    if !is_type_shared(&shares_t) {
        return Err(runtime_error!("Input shares must be a tuple of 2 elements"));
    }
    let shares_type_vector = get_types_vector(shares_t)?;
    if shares_type_vector.len() != 2 {
        return Err(runtime_error!(
            "There should be only 2 shares in the input tuple"
        ));
    }
    let share_t = (*shares_type_vector[0]).clone();
    if share_t != (*shares_type_vector[1]).clone() {
        return Err(runtime_error!("Input shares must be of the same type"));
    }
    if !share_t.is_named_tuple() {
        return Err(runtime_error!("Each share must be a named tuple"));
    }
    let headers_types = share_t.get_named_types()?;
    let mut headers_column_types = vec![];
    let mut num_rows = 0;
    for (h, t) in &headers_types {
        let column_type = ColumnType::new(t.clone(), false, h)?;
        let num_column_entries = column_type.get_num_entries();
        if num_rows == 0 {
            num_rows = num_column_entries;
        }
        if num_rows != num_column_entries {
            return Err(runtime_error!(
                "Number of entries should be the same in all columns"
            ));
        }
        headers_column_types.push((h.clone(), column_type));
    }

    let prf_t = argument_types[2].clone();
    let expected_key_type = tuple_type(vec![array_type(vec![KEY_LENGTH], BIT); 3]);
    if prf_t != expected_key_type {
        return Err(runtime_error!(
            "PRF key type should be a tuple of 3 binary arrays of length {}",
            KEY_LENGTH
        ));
    }
    if sender_id >= PARTIES as u64 {
        return Err(runtime_error!("Sender ID is incorrect"));
    }
    if programmer_id >= PARTIES as u64 {
        return Err(runtime_error!("Programmer ID is incorrect"));
    }
    if sender_id == programmer_id {
        return Err(runtime_error!(
            "Programmer ID should be different from the Sender ID"
        ));
    }

    Ok((num_rows, headers_column_types))
}

fn get_receiver_id(sender_id: u64, programmer_id: u64) -> u64 {
    // This is correct only if PARTIES = 3.
    PARTIES as u64 - sender_id - programmer_id
}

// Get the prf key unknown to a given party.
// In case of 3 parties, this key is also a common key for the other two parties.
// Party k knows keys prf_keys[k] and prf_keys[(k+1)%3], but has no clue about prf_keys[(k-1)%3].
fn get_hidden_prf_key(prf_keys: Node, party_id: u64) -> Result<Node> {
    let key_index = ((party_id as usize + PARTIES - 1) % PARTIES) as u64;
    prf_keys.tuple_get(key_index)
}

/// Adds a node that permutes an array shared between Sender and Programmer using a permutation known to Programmer.
/// The output shares are returned only to Receiver and Programmer.
///
/// Input shares are assumed to be a tuple of 2-out-of-2 shares.
/// Each share must be a named tuple containing integer or binary arrays.
/// So databases converted to such named tuples are handled column-wise.
///
/// The protocol follows the Permute protocol from <https://eprint.iacr.org/2019/518.pdf>.
/// Assume that Sender and Programmer have shares `X_s` and `X_p`, respectively.
/// 1. Programmer creates a random composition of its permutation `perm = perm_r * perm_s`,
/// where `perm_r` and `perm_s` are random permutations sent to Receiver and Sender.
/// 2. Programmer and Sender generate a random mask S of the same type as one input share.
/// 3. Programmer and Receiver generate a random mask T of the same type as one input share.
/// 4. Sender computes `B = perm_s(X_s) - S` and sends it to Receiver
/// 5. Receiver computes its share of the output `Y_r = perm_r(B) - T`.
/// 6. Programmer computes its share of the output `Y_p = perm_r(S) + T + perm(X_p)`.
///
/// **WARNING**: this function should not be used before MPC compilation.
///
/// # Custom operation arguments
///
/// - tuple of 2-out-of-2 shares owned by Sender and Programmer
/// - permutation array known to Programmer
/// - tuple of 3 PRF keys used for multiplication
///
/// # Custom operation returns
///
/// Tuple of permuted 2-out-of-2 shares known to Receiver and Programmer
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
struct PermutationMPC {
    pub sender_id: u64,
    pub programmer_id: u64, // The receiver ID is defined automatically
}

#[typetag::serde]
impl CustomOperationBody for PermutationMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        // Check permutation and input types
        let (num_entries, column_header_types) = check_and_extract_map_input_parameters(
            &argument_types,
            self.sender_id,
            self.programmer_id,
        )?;
        // Check that the permutation map is of the correct form
        let permutation_t = argument_types[1].clone();
        if !permutation_t.is_array() {
            return Err(runtime_error!("Permutation map must be an array"));
        }
        if permutation_t.get_shape()[0] > num_entries {
            return Err(runtime_error!(
                "Permutation map length can't be bigger than the number of entries"
            ));
        }

        let shares_t = argument_types[0].clone();
        let prf_t = argument_types[2].clone();

        let sender_id = self.sender_id;
        let programmer_id = self.programmer_id;
        let receiver_id = get_receiver_id(sender_id, programmer_id);

        let g = context.create_graph()?;

        let shares = g.input(shares_t)?;
        let permutation = g.input(permutation_t)?;

        let mut sender_perm = g.random_permutation(num_entries)?;
        let inverse_sender_perm = sender_perm.inverse_permutation()?;
        // Composition permutation(inverse_sender_perm())
        let mut receiver_perm = inverse_sender_perm.gather(permutation.clone(), 0)?;

        // Programmer sends permutations to Sender and Receiver
        sender_perm = sender_perm
            .nop()?
            .add_annotation(NodeAnnotation::Send(programmer_id, sender_id))?;
        receiver_perm = receiver_perm
            .nop()?
            .add_annotation(NodeAnnotation::Send(programmer_id, receiver_id))?;

        // Generate randomness between Sender and Programmer, Programmer and Receiver (PRF keys are needed)
        let prf_keys = g.input(prf_t)?;

        // Choose PRF keys known to Sender and Programmer, Programmer and Receiver.
        // If key is known to parties A and B, then it must be unknown to party C.
        let prf_key_s_p = get_hidden_prf_key(prf_keys.clone(), receiver_id)?;
        let prf_key_p_r = get_hidden_prf_key(prf_keys, sender_id)?;

        let sender_share = shares.tuple_get(1)?;
        let programmer_share = shares.tuple_get(0)?;
        let mut receiver_columns = vec![];
        let mut programmer_columns = vec![];
        for (column_header, _) in column_header_types {
            // Permute the column share of Sender and mask it
            // Select a column
            let sender_share_column = sender_share.named_tuple_get(column_header.clone())?;
            // Permute the column
            let sender_share_column_permuted =
                sender_share_column.gather(sender_perm.clone(), 0)?;
            // Generate a random column mask known to Sender and Programmer
            let sender_column_mask = g.prf(
                prf_key_s_p.clone(),
                0,
                sender_share_column_permuted.get_type()?,
            )?;
            // Mask the column
            let mut sender_share_column_masked =
                sender_share_column_permuted.subtract(sender_column_mask.clone())?;
            // Send the result to Receiver
            sender_share_column_masked = sender_share_column_masked
                .nop()?
                .add_annotation(NodeAnnotation::Send(sender_id, receiver_id))?;
            // Compute the column share of Receiver
            // Permute Sender's masked share
            let mut receiver_result_column =
                sender_share_column_masked.gather(receiver_perm.clone(), 0)?;
            // Generate a random column mask known to Receiver and Programmer
            let receiver_mask =
                g.prf(prf_key_p_r.clone(), 0, receiver_result_column.get_type()?)?;
            // Mask the column
            receiver_result_column = receiver_result_column.subtract(receiver_mask.clone())?;
            // Compute the share of Programmer
            // Select a column
            let programmer_share_column =
                programmer_share.named_tuple_get(column_header.clone())?;
            // Permute Sender's mask (which is known to Programmer) and its input share
            // Then, sum these together with Receiver's mask
            let programmer_result_column = sender_column_mask
                .gather(receiver_perm.clone(), 0)?
                .add(receiver_mask)?
                .add(programmer_share_column.gather(permutation.clone(), 0)?)?;

            receiver_columns.push((column_header.clone(), receiver_result_column));
            programmer_columns.push((column_header, programmer_result_column));
        }
        let receiver_result_share = g.create_named_tuple(receiver_columns)?;
        let programmer_result_share = g.create_named_tuple(programmer_columns)?;

        g.create_tuple(vec![programmer_result_share, receiver_result_share])?
            .set_as_output()?;

        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!(
            "Permutation(sender:{},programming:{})",
            self.sender_id, self.programmer_id
        )
    }
}

/// Adds a node that duplicates some elements of an array shared between Sender and Programmer using a duplication map known to Programmer.
/// The output shares are returned only to Receiver and Programmer.
///
/// A duplication map is a tuple of two one-dimensional arrays of length `n`.
/// The first array contains indices from `[0,n]` in the increasing order with possible repetitions.
/// The second array contains only zeros and ones.
/// If its i-th element is zero, it means that the duplication map doesn't change the i-th element of an array it acts upon.
/// If map's i-th element is one, then the map copies the previous element of the result.
/// This rules can be summarized by the following equation
///
/// `duplication_indices[i] = duplication_bits[i] * duplication_indices[i-1] + (1 - duplication_bits[i]) * i`
///
/// Input shares are assumed to be a tuple of 2-out-of-2 shares.
/// Each share must be a named tuple containing integer or binary arrays.
/// So databases converted to such named tuples are handled column-wise.
///
/// The protocol follows the Duplicate protocol from <https://eprint.iacr.org/2019/518.pdf>.
/// For each column header, the following steps are performed.
/// 1. Sender selects an input column `C_s`.
/// 2. Sender and Receiver generate shared randomness `B_r[i]` for i in `[1,num_entries]`, `W_0` and `W_1` of size of a column without one entry.
/// 2. Sender selects the first entry and masks it with a random value `B0_p` also known to Programmer.
/// This value is assigned to `B_r[0]`.
/// 3. Sender and programmer generate a random mask phi of the duplication bits.
/// 4. Sender computes two columns `M0` and `M1` such that
///    
///    `M0[i] = C_s[i] - B_r[i] - W_(duplication_bits[i])[i],`
///    `M1[i] = B_r[i-1] - B_r[i] - W_(1-duplication_bits[i])[i].`
///    
///    for i in `[1, num_entries]`.
/// 5. Sender sends `M0` and `M1` to Programmer.
/// 6. Programmer and Receiver generate a random value `R` of size of an input share.
/// 7. Programmer masks the duplication map by computing `rho = phi XOR duplication_bits` except for the first bit.
/// 8. Programmer sends rho to Receiver.
/// 9. Receiver selects `W_(rho[i])[i]` for i in `[1, num_entries]` and sends them to Programmer.
/// 10. Programmer computes
///
///     `B_p[i] = M_(duplication_bits[i])[i] + W_(rho[i])[i] + dup_bits[i] * B_p[i-1]`
///
///     for i in `[1,num_entries]`.
/// 11. Compute the share of Programmer equal to `B_p - R + duplication_map(programmer column share)`
/// 12. Compute the share of Receiver `B_r + R`
///
/// **WARNING**: this function should not be used before MPC compilation.
///
/// # Custom operation arguments
///
/// - tuple of 2-out-of-2 shares owned by Sender and Programmer
/// - a tuple of a duplication map array and the corresponding repetition bits known to Programmer
/// - tuple of 3 PRF keys used for multiplication
///
/// # Custom operation returns
///
/// Tuple of duplicated 2-out-of-2 shares known to Receiver and Programmer
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
struct DuplicationMPC {
    pub sender_id: u64,
    pub programmer_id: u64, // The receiver ID is defined automatically
}

#[typetag::serde]
impl CustomOperationBody for DuplicationMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        // Check input types and extract their parameters
        let (num_entries, column_header_types) = check_and_extract_map_input_parameters(
            &argument_types,
            self.sender_id,
            self.programmer_id,
        )?;
        // An additional check that the duplication map is of the correct form
        let dup_map_t = argument_types[1].clone();
        if let Type::Tuple(dup_map_types) = dup_map_t.clone() {
            let dup_indices_t = dup_map_types[0].clone();
            let dup_bits_t = dup_map_types[1].clone();
            if !dup_indices_t.is_array() || !dup_bits_t.is_array() {
                return Err(runtime_error!("Duplication map should contain two arrays"));
            }
            if dup_indices_t.get_scalar_type() != UINT64 {
                return Err(runtime_error!(
                    "Duplication map indices should be of the UINT64 type"
                ));
            }
            if dup_bits_t.get_scalar_type() != BIT {
                return Err(runtime_error!(
                    "Duplication map bits should be of the BIT type"
                ));
            }
            let num_dup_indices = dup_indices_t.get_shape()[0];
            let num_dup_bits = dup_bits_t.get_shape()[0];
            if num_dup_indices != num_entries {
                return Err(runtime_error!(
                    "Duplication map indices should be of length equal to the number of entries"
                ));
            }
            if num_dup_bits != num_entries {
                return Err(runtime_error!(
                    "Duplication map bits should be of length equal to the number of entries"
                ));
            }
        } else {
            return Err(runtime_error!("Duplication map should be a tuple"));
        }

        let sender_id = self.sender_id;
        let programmer_id = self.programmer_id;
        let receiver_id = get_receiver_id(sender_id, programmer_id);

        let shares_t = argument_types[0].clone();
        let prf_t = argument_types[2].clone();

        let g = context.create_graph()?;

        let shares = g.input(shares_t)?;
        let duplication_map = g.input(dup_map_t)?;

        let duplication_indices = duplication_map.tuple_get(0)?;
        let duplication_bits = duplication_map.tuple_get(1)?;

        // Generate randomness between all possible pairs of parties.
        let prf_keys = g.input(prf_t)?;

        // If key is known to parties A and B, then it must be unknown to party C.
        let prf_key_s_p = get_hidden_prf_key(prf_keys.clone(), receiver_id)?;
        let prf_key_p_r = get_hidden_prf_key(prf_keys.clone(), sender_id)?;
        let prf_key_s_r = get_hidden_prf_key(prf_keys, programmer_id)?;

        let programmer_share = shares.tuple_get(0)?;
        let sender_share = shares.tuple_get(1)?;

        let mut receiver_columns = vec![];
        let mut programmer_columns = vec![];
        for column_header_type in column_header_types {
            let column_header = column_header_type.0;
            // Sender selects an input column
            let sender_column = sender_share.named_tuple_get(column_header.clone())?;
            let column_t = sender_column.get_type()?;
            let column_shape = column_t.get_shape();
            // If the number of entries is 1 the protocol can be simplified.
            // Namely, Programmer and Sender simply reshare one row to Programmer and Receiver
            if num_entries == 1 {
                // Sender selects the first entry share and masks it with a random mask B_p known also to Programmer.
                // The result is assigned to B_r and sent to Receiver.
                let b_p = prf_key_s_p.prf(0, sender_column.get_type()?)?;
                let b_r = sender_column
                    .subtract(b_p.clone())?
                    .nop()?
                    .add_annotation(NodeAnnotation::Send(sender_id, receiver_id))?;
                // Programmer and Receiver generate a random value R of size of an input share
                let r = prf_key_p_r.prf(0, b_p.get_type()?)?;
                // Compute the share of Programmer which is equal to
                // B_p - R + programmer column share
                let programmer_result_column = b_p
                    .subtract(r.clone())?
                    .add(programmer_share.named_tuple_get(column_header.clone())?)?;

                // Receiver's share B_r + R
                let receiver_result_column = b_r.add(r)?;

                receiver_columns.push((column_header.clone(), receiver_result_column));
                programmer_columns.push((column_header, programmer_result_column));
                continue;
            }
            // Sender and Receiver generate random B_r[i] for i in {1,...,num_entries-1}, W_0 and W_1 of size of an input share.
            let mut column_wout_entry_shape = column_shape.clone();
            column_wout_entry_shape[0] = num_entries - 1;
            let column_wout_entry_t =
                array_type(column_wout_entry_shape, column_t.get_scalar_type());
            let bi_r = prf_key_s_r.prf(0, column_wout_entry_t.clone())?;
            let w0 = prf_key_s_r.prf(0, column_wout_entry_t.clone())?;
            let w1 = prf_key_s_r.prf(0, column_wout_entry_t.clone())?;

            // Sender selects the first entry share and masks it with a random mask B_p[0] known also to Programmer.
            // The result is assigned to B_r[0] and sent to Receiver.
            let entry0 = sender_column.get(vec![0])?;
            let b0_p = prf_key_s_p.prf(0, entry0.get_type()?)?;
            let mut first_entry_shape = column_shape.clone();
            first_entry_shape[0] = 1;
            let b0_r = entry0
                .subtract(b0_p.clone())?
                .reshape(array_type(first_entry_shape, column_t.get_scalar_type()))?
                .nop()?
                .add_annotation(NodeAnnotation::Send(sender_id, receiver_id))?;

            // Merge B_r[0] and B_r[i] for i in [1,num_entries]
            let b_r = g.concatenate(vec![b0_r.clone(), bi_r], 0)?;

            // Sender and programmer generate a random mask phi of the duplication map
            let mut phi = prf_key_s_p.prf(0, array_type(vec![num_entries - 1], BIT))?;

            // Sender computes two columns M0 and M1 such that
            //
            //    M0[i] = sender_column[i] - B_r[i] - W_(duplication_bits[i])[i],
            //    M1[i] = B_r[i-1] - B_r[i] - W_(1-duplication_bits[i])[i]
            //
            // for i in [1, num_entries]
            let b_r_without_first_entry =
                b_r.get_slice(vec![SliceElement::SubArray(Some(1), None, None)])?;
            let b_r_without_last_entry = b_r.get_slice(vec![SliceElement::SubArray(
                None,
                Some(num_entries as i64 - 1),
                None,
            )])?;

            // Reshape duplication bits and phi to enable broadcasting
            let mut duplication_bits_wout_first_entry =
                duplication_bits.get_slice(vec![SliceElement::SubArray(Some(1), None, None)])?;
            if column_shape.len() > 1 {
                let mut new_shape = vec![1; column_shape.len()];
                new_shape[0] = num_entries - 1;
                duplication_bits_wout_first_entry = duplication_bits_wout_first_entry
                    .reshape(array_type(new_shape.clone(), BIT))?;
                phi = phi.reshape(array_type(new_shape, BIT))?;
            }

            let selected_w_for_m0 = select_node(phi.clone(), w1.clone(), w0.clone())?;
            let selected_w_for_m1 = select_node(phi.clone(), w0.clone(), w1.clone())?;
            let mut m0 = sender_column
                .get_slice(vec![SliceElement::SubArray(Some(1), None, None)])?
                .subtract(b_r_without_first_entry.clone())?
                .subtract(selected_w_for_m0)?;
            let mut m1 = b_r_without_last_entry
                .subtract(b_r_without_first_entry)?
                .subtract(selected_w_for_m1)?;

            // Sender sends M_0 and M_1 to Programmer
            m0 = m0
                .nop()?
                .add_annotation(NodeAnnotation::Send(sender_id, programmer_id))?;
            m1 = m1
                .nop()?
                .add_annotation(NodeAnnotation::Send(sender_id, programmer_id))?;

            // Programmer and Receiver generate a random value R of size of an input share
            let r = prf_key_p_r.prf(0, column_t.clone())?;

            // Programmer masks the duplication map by computing rho = phi XOR dup_map except for the first bit.
            let mut rho = duplication_bits_wout_first_entry.add(phi)?;
            rho = rho
                .nop()?
                .add_annotation(NodeAnnotation::Send(programmer_id, receiver_id))?;

            // Receiver selects W_(rho[i])[i] for i in [1, num_entries] and sends them to Programmer
            let selected_w_for_programmer = select_node(rho, w1, w0)?
                .nop()?
                .add_annotation(NodeAnnotation::Send(receiver_id, programmer_id))?;

            // Programmer computes
            //
            // B_p[i] = M_(duplication_bits[i])[i] + W_(rho[i])[i] + duplication_bits[i] * B_p[i-1]
            //
            // for i in {1,..., num_entries-1}.
            // B_p[0] is computed earlier.
            let m_plus_w = select_node(duplication_bits_wout_first_entry.clone(), m1, m0)?
                .add(selected_w_for_programmer)?;

            let b_p = g.segment_cumsum(
                m_plus_w,
                duplication_bits_wout_first_entry
                    .reshape(array_type(vec![num_entries - 1], BIT))?,
                b0_p.clone(),
            )?;

            // Compute the share of Programmer which is equal to
            // B_p - R + duplication_map(programmer column share)
            let programmer_result_column = b_p.subtract(r.clone())?.add(
                programmer_share
                    .named_tuple_get(column_header.clone())?
                    .gather(duplication_indices.clone(), 0)?,
            )?;

            // Receiver's share B_r + R
            let receiver_result_column = b_r.add(r)?;

            receiver_columns.push((column_header.clone(), receiver_result_column));
            programmer_columns.push((column_header, programmer_result_column));
        }
        let receiver_result_share = g.create_named_tuple(receiver_columns)?;
        let programmer_result_share = g.create_named_tuple(programmer_columns)?;

        g.create_tuple(vec![programmer_result_share, receiver_result_share])?
            .set_as_output()?;

        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!(
            "Duplication(sender:{},programming:{})",
            self.sender_id, self.programmer_id
        )
    }
}

/// Adds a node that computes a switching network on data shared by Sender and Programmer using a switching map known to Programmer.
///
/// The output shares are returned only to Receiver and Programmer.
///
/// A switching network map is a one-dimensional array of length `m` that contains non-unique indices of an array of length `n`, which is not smaller than `m`.
///
/// Input shares are assumed to be a tuple of 2-out-of-2 shares.
/// Each share must be a named tuple containing integer or binary arrays.
/// So databases converted to such named tuples are handled column-wise.
///
/// The protocol follows the Switch protocol from <https://eprint.iacr.org/2019/518.pdf>.
/// For each column header, the following steps are performed.
/// 1. Programmer decomposes a given switching map into a permutation-with-deletion, duplication and permutation maps using the `DecomposeSwitchingMap` operation.
/// 2. Sender and Programmer engage in the Permutation protocol using the permutation-with-deletion map.
/// 3. Receiver and Programmer engage in the Duplication protocol using the duplication map.
///
/// **WARNING**: this function should not be used before MPC compilation.
///
/// # Custom operation arguments
///
/// - tuple of 2-out-of-2 shares owned by Sender and Programmer
/// - an UINT64 array containing a switching map
/// - tuple of 3 PRF keys used for multiplication
///
/// # Custom operation returns
///
/// Tuple of permuted 2-out-of-2 shares known to Receiver and Programmer
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
struct SwitchingMPC {
    pub sender_id: u64,
    pub programmer_id: u64, // The receiver ID is defined automatically
}

#[typetag::serde]
impl CustomOperationBody for SwitchingMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        // Check permutation and input types
        let (num_entries, _) = check_and_extract_map_input_parameters(
            &argument_types,
            self.sender_id,
            self.programmer_id,
        )?;
        // An additional check that the switching map is of the correct form
        let switch_map_t = argument_types[1].clone();
        if !switch_map_t.is_array() {
            return Err(runtime_error!("Switching map should be an array"));
        }
        if switch_map_t.get_scalar_type() != UINT64 {
            return Err(runtime_error!(
                "Switching map indices should be of the UINT64 type"
            ));
        }
        let num_switch_indices = switch_map_t.get_shape()[0];
        if num_switch_indices > num_entries {
            return Err(runtime_error!(
                "Switching map cannot have more than {} indices",
                num_entries
            ));
        }

        let receiver_id = get_receiver_id(self.sender_id, self.programmer_id);

        let shares_t = argument_types[0].clone();
        let prf_t = argument_types[2].clone();

        let g = context.create_graph()?;

        let shares = g.input(shares_t)?;
        let switch_map = g.input(switch_map_t)?;

        // Generate randomness between all possible pairs of parties.
        let prf_keys = g.input(prf_t)?;

        // Programmer decomposes a given switching map into a permutation with deletion, duplication and permutation maps using the `DecomposeSwitchingMap` operation
        let switch_decomposition = switch_map.decompose_switching_map(num_entries)?;

        let permutation_with_deletion = switch_decomposition.tuple_get(0)?;
        let duplication = switch_decomposition.tuple_get(1)?;
        let permutation = switch_decomposition.tuple_get(2)?;

        // Sender and Programmer engage in the Permutation protocol using the permutation-with-deletion map
        let permuted_and_reduced_shares = g.custom_op(
            CustomOperation::new(PermutationMPC {
                sender_id: self.sender_id,
                programmer_id: self.programmer_id,
            }),
            vec![shares, permutation_with_deletion, prf_keys.clone()],
        )?;

        // Receiver and Programmer engage in the Duplication protocol using the duplication map
        let duplicated_shares = g.custom_op(
            CustomOperation::new(DuplicationMPC {
                sender_id: receiver_id,
                programmer_id: self.programmer_id,
            }),
            vec![permuted_and_reduced_shares, duplication, prf_keys.clone()],
        )?;

        // Sender and Programmer engage in the Permutation protocol using the last permutation to produce the output of the switching map
        let switched_shares = g.custom_op(
            CustomOperation::new(PermutationMPC {
                sender_id: self.sender_id,
                programmer_id: self.programmer_id,
            }),
            vec![duplicated_shares, permutation, prf_keys],
        )?;

        switched_shares.set_as_output()?;

        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!(
            "Switching(sender:{},programming:{})",
            self.sender_id, self.programmer_id
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ndarray::array;

    use super::*;

    use crate::custom_ops::{run_instantiation_pass, CustomOperation};
    use crate::data_types::{scalar_type, ArrayShape, INT16, INT32, INT8, UINT16, UINT8};
    use crate::data_values::Value;
    use crate::evaluators::join_test_utils::{
        column_info, column_info_with_mask, expected_set_info, expected_set_info_with_mask,
        join_info, JoinTestInfo, SetHelpers,
    };
    use crate::evaluators::{evaluate_simple_evaluator, random_evaluate};

    use crate::graphs::util::simple_context;
    use crate::inline::inline_common::DepthOptimizationLevel;
    use crate::inline::inline_ops::{inline_operations, InlineConfig, InlineMode};
    use crate::mpc::mpc_compiler::{generate_prf_key_triple, prepare_for_mpc_evaluation, IOStatus};
    use crate::mpc::mpc_equivalence_class::{
        generate_equivalence_class, private_class, share0_class, share1_class, share2_class,
        vector_class, EquivalenceClasses,
    };
    use crate::random::SEED_SIZE;

    fn simple_hash_helper(
        input_shape: ArrayShape,
        hash_shape: ArrayShape,
        inputs: Vec<Value>,
    ) -> Result<Vec<u64>> {
        let c = simple_context(|g| {
            let i = g.input(array_type(input_shape.clone(), BIT))?;
            let hash_matrix = g.input(array_type(hash_shape.clone(), BIT))?;
            g.custom_op(CustomOperation::new(SimpleHash), vec![i, hash_matrix])
        })?;

        let mapped_c = run_instantiation_pass(c)?.context;
        let result_value = random_evaluate(mapped_c.get_main_graph()?, inputs)?;
        let mut result_shape = input_shape[0..input_shape.len() - 1].to_vec();
        result_shape.insert(0, hash_shape[0]);
        let result_type = array_type(result_shape, UINT64);
        result_value.to_flattened_array_u64(result_type)
    }

    fn simple_hash_helper_fails(input_t: Type, hash_t: Type) -> Result<()> {
        let c = simple_context(|g| {
            let i = g.input(input_t)?;
            let hash_matrix = g.input(hash_t)?;
            g.custom_op(CustomOperation::new(SimpleHash), vec![i, hash_matrix])
        })?;
        run_instantiation_pass(c)?;
        Ok(())
    }

    #[test]
    fn test_simple_hash() -> Result<()> {
        // no collision
        {
            // [2,3]-array
            let input = Value::from_flattened_array(&[1, 0, 1, 0, 0, 1], BIT)?;
            // [3,2,3]-array
            let hash_matrix = Value::from_flattened_array(
                &[1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                BIT,
            )?;
            // output [3,2]-array
            let expected = vec![0, 1, 0, 2, 3, 2];
            assert_eq!(
                simple_hash_helper(vec![2, 3], vec![3, 2, 3], vec![input, hash_matrix])?,
                expected
            );
        }
        // collisions
        {
            // [2,3]-array
            let input = Value::from_flattened_array(&[1, 0, 1, 0, 0, 0], BIT)?;
            // [3,2,3]-array
            let hash_matrix = Value::from_flattened_array(
                &[1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                BIT,
            )?;
            // output [3,2]-array
            let expected = vec![0, 0, 0, 0, 3, 0];
            assert_eq!(
                simple_hash_helper(vec![2, 3], vec![3, 2, 3], vec![input, hash_matrix])?,
                expected
            );
        }
        {
            // [2,2,2]-array
            let input = Value::from_flattened_array(&[1, 0, 0, 0, 1, 1, 0, 1], BIT)?;
            // [2,3,2]-array
            let hash_matrix =
                Value::from_flattened_array(&[1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1], BIT)?;
            // output [2,2,2]-array
            let expected = vec![3, 0, 2, 0, 7, 4, 7, 5];
            assert_eq!(
                simple_hash_helper(vec![2, 2, 2], vec![2, 3, 2], vec![input, hash_matrix])?,
                expected
            );
        }
        // malformed input
        {
            let input_t = scalar_type(BIT);
            let hash_t = array_type(vec![2, 3, 4], BIT);
            assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
        }
        {
            let input_t = array_type(vec![5, 4], UINT64);
            let hash_t = array_type(vec![2, 3, 4], BIT);
            assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
        }
        {
            let input_t = array_type(vec![4], BIT);
            let hash_t = array_type(vec![2, 3, 4], BIT);
            assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
        }
        {
            let input_t = array_type(vec![5, 4], BIT);
            let hash_t = scalar_type(BIT);
            assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
        }
        {
            let input_t = array_type(vec![5, 4], BIT);
            let hash_t = array_type(vec![3, 4], BIT);
            assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
        }
        {
            let input_t = array_type(vec![5, 4], BIT);
            let hash_t = array_type(vec![2, 3, 4], UINT64);
            assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
        }
        {
            let input_t = array_type(vec![5, 4], BIT);
            let hash_t = array_type(vec![2, 64, 4], BIT);
            assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
        }
        {
            let input_t = array_type(vec![5, 4], BIT);
            let hash_t = array_type(vec![2, 3, 5], BIT);
            assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
        }
        Ok(())
    }

    #[test]
    fn test_permutation() -> Result<()> {
        let data_helper = |a_type: Type,
                           b_type: Type,
                           a_values: &[u64],
                           b_values: &[u64],
                           permutation_values: &[u64],
                           a_expected: &[u64],
                           b_expected: &[u64]|
         -> Result<()> {
            // test correct inputs
            let roles_helper = |sender_id: u64, programmer_id: u64| -> Result<()> {
                let c = simple_context(|g| {
                    let column_a = g.input(a_type.clone())?;
                    let column_b = g.input(b_type.clone())?;

                    // Generate PRF keys
                    let key_t = array_type(vec![KEY_LENGTH], BIT);
                    let keys_vec = generate_prf_key_triple(g.clone())?;
                    let keys = g.create_tuple(keys_vec)?;
                    // PRF key known only to Sender.
                    let key_s = g.random(key_t)?;
                    // Split input into two shares between Sender and Programmer
                    // Sender generates Programmer's shares
                    let column_a_programmer_share = g.prf(key_s.clone(), 0, a_type.clone())?;
                    let column_b_programmer_share = g.prf(key_s, 0, b_type.clone())?;
                    // Sender computes its shares
                    let column_a_sender_share =
                        column_a.subtract(column_a_programmer_share.clone())?;
                    let column_b_sender_share =
                        column_b.subtract(column_b_programmer_share.clone())?;

                    // Sender packs shares in named tuples and send one of them to Programmer
                    let programmer_share = g
                        .create_named_tuple(vec![
                            ("a".to_owned(), column_a_programmer_share),
                            ("b".to_owned(), column_b_programmer_share),
                        ])?
                        .nop()?
                        .add_annotation(NodeAnnotation::Send(sender_id, programmer_id))?;
                    let sender_share = g.create_named_tuple(vec![
                        ("a".to_owned(), column_a_sender_share),
                        ("b".to_owned(), column_b_sender_share),
                    ])?;

                    // Pack shares together
                    let shares = g.create_tuple(vec![programmer_share, sender_share])?;

                    // Permutation input
                    let permutation =
                        g.input(array_type(vec![permutation_values.len() as u64], UINT64))?;

                    // Permuted shares
                    let permuted_shares = g.custom_op(
                        CustomOperation::new(PermutationMPC {
                            sender_id,
                            programmer_id,
                        }),
                        vec![shares, permutation, keys],
                    )?;

                    // Sum permuted shares
                    let receiver_permuted_share = permuted_shares.tuple_get(1)?;
                    let programmer_permuted_share = permuted_shares.tuple_get(0)?;

                    let permuted_column_a = receiver_permuted_share
                        .named_tuple_get("a".to_owned())?
                        .add(programmer_permuted_share.named_tuple_get("a".to_owned())?)?;
                    let permuted_column_b = receiver_permuted_share
                        .named_tuple_get("b".to_owned())?
                        .add(programmer_permuted_share.named_tuple_get("b".to_owned())?)?;

                    // Combine permuted columns
                    g.create_tuple(vec![permuted_column_a, permuted_column_b])
                })?;

                let instantiated_c = run_instantiation_pass(c)?.context;
                let inlined_c = inline_operations(
                    &instantiated_c,
                    InlineConfig {
                        default_mode: InlineMode::Simple,
                        ..Default::default()
                    },
                )?
                .get_context();

                let result_hashmap = generate_equivalence_class(
                    &inlined_c,
                    vec![vec![
                        IOStatus::Party(sender_id),
                        IOStatus::Party(sender_id),
                        IOStatus::Party(programmer_id),
                    ]],
                )?;

                let receiver_id = PARTIES as u64 - sender_id - programmer_id;
                let private_class = EquivalenceClasses::Atomic(vec![vec![0], vec![1], vec![2]]);
                // data shared by Sender and Programmer
                let share_r_sp = EquivalenceClasses::Atomic(vec![
                    vec![receiver_id],
                    vec![sender_id, programmer_id],
                ]);
                // data shared by the Receiver and Programmer
                let share_s_rp = EquivalenceClasses::Atomic(vec![
                    vec![sender_id],
                    vec![receiver_id, programmer_id],
                ]);
                // data shared by Receiver and Sender
                let share_p_rs = EquivalenceClasses::Atomic(vec![
                    vec![programmer_id],
                    vec![receiver_id, sender_id],
                ]);
                // data shared by parties 0 and 1
                let share_2_01 = EquivalenceClasses::Atomic(vec![vec![2], vec![0, 1]]);
                // data shared by parties 1 and 2
                let share_0_12 = EquivalenceClasses::Atomic(vec![vec![0], vec![1, 2]]);
                // data shared by parties 2 and 0
                let share_1_20 = EquivalenceClasses::Atomic(vec![vec![1], vec![2, 0]]);

                let private_pair = vector_class(vec![private_class.clone(); 2]);
                let programmers_share_class = vector_class(vec![share_r_sp.clone(); 2]);

                let expected_classes = vec![
                    // both inputs should be known only to Sender
                    private_class.clone(),
                    private_class.clone(),
                    // First PRF key
                    private_class.clone(),
                    share_1_20.clone(),
                    // Second PRF key
                    private_class.clone(),
                    share_2_01.clone(),
                    // Third PRF key
                    private_class.clone(),
                    share_0_12.clone(),
                    // All PRF keys
                    vector_class(vec![share_1_20, share_2_01, share_0_12]),
                    // PRF key known to Sender
                    private_class.clone(),
                    // Programmer's input shares
                    private_class.clone(),
                    private_class.clone(),
                    // Sender's input shares
                    private_class.clone(),
                    private_class.clone(),
                    // Programmer's share
                    private_pair.clone(),
                    programmers_share_class.clone(),
                    // Sender's share
                    private_pair.clone(),
                    // Tuple of both shares
                    vector_class(vec![programmers_share_class.clone(), private_pair.clone()]),
                    // Permutation input
                    private_class.clone(),
                    // Sender's permutation generated by Programmer
                    private_class.clone(),
                    // Inverse of Sender's permutation
                    private_class.clone(),
                    // Receiver's permutation generated by Programmer
                    private_class.clone(),
                    // Sender's permutation after sending to Sender
                    share_r_sp.clone(),
                    // Receiver's permutation after sending to Receiver
                    share_s_rp.clone(),
                    // PRF key known to Sender and Programmer
                    share_r_sp.clone(),
                    // PRF key known to Receiver and Programmer
                    share_s_rp.clone(),
                    // Sender's share
                    private_pair.clone(),
                    // Programmer's share
                    programmers_share_class,
                    // Sender's share of the first column
                    private_class.clone(),
                    // Permuted Sender's share of the first column
                    private_class.clone(),
                    // Random mask known to Sender and Programmer
                    share_r_sp.clone(),
                    // Masked permuted Sender's share of the first column
                    private_class.clone(),
                    // Masked permuted Sender's share of the first column sent to Receiver
                    share_p_rs.clone(),
                    // Receiver permutes the above share
                    private_class.clone(),
                    // Random mask known to Receiver and Programmer
                    share_s_rp.clone(),
                    // Receiver's resulting share of the permuted first column
                    private_class.clone(),
                    // Programmer's share of the first column (since Sender shared data)
                    share_r_sp.clone(),
                    // Permutation of Sender's mask
                    private_class.clone(),
                    // Sum of the permuted Sender's mask and Receiver's mask
                    private_class.clone(),
                    // Permutation of Programmer's share of the first column
                    private_class.clone(),
                    // Programmer's resulting share of the permuted first column
                    private_class.clone(),
                    // Sender's share of the second column
                    private_class.clone(),
                    // Permuted Sender's share of the second column
                    private_class.clone(),
                    // Random mask known to Sender and Programmer
                    share_r_sp.clone(),
                    // Masked permuted Sender's share of the second column
                    private_class.clone(),
                    // Masked permuted Sender's share of the second column sent to Receiver
                    share_p_rs,
                    // Receiver permutes the above share
                    private_class.clone(),
                    // Random mask known to Receiver and Programmer
                    share_s_rp,
                    // Receiver's resulting share of the permuted second column
                    private_class.clone(),
                    // Programmer's share of the second column (since Sender shared data)
                    share_r_sp,
                    // Permutation of Sender's mask
                    private_class.clone(),
                    // Sum of the permuted Sender's mask and Receiver's mask
                    private_class.clone(),
                    // Permutation of Programmer's share of the second column
                    private_class.clone(),
                    // Programmer's resulting share of the permuted second column
                    private_class,
                    // Receiver's result share of the named tuple
                    private_pair.clone(),
                    // Programmer's result share of the named tuple
                    private_pair.clone(),
                    // Both shares combined (the output of the protocol)
                    vector_class(vec![private_pair; 2]),
                ];
                let mut result_classes = vec![];
                for i in 0..expected_classes.len() as u64 {
                    result_classes.push((*result_hashmap.get(&(0, i)).unwrap()).clone());
                }
                assert_eq!(result_classes, expected_classes);

                // Check evaluation
                let result = random_evaluate(
                    inlined_c.get_main_graph()?,
                    vec![
                        Value::from_flattened_array(a_values, a_type.get_scalar_type())?,
                        Value::from_flattened_array(b_values, b_type.get_scalar_type())?,
                        Value::from_flattened_array(permutation_values, UINT64)?,
                    ],
                )?;
                let mut result_a_shape = a_type.get_shape();
                result_a_shape[0] = permutation_values.len() as u64;
                let result_a_type = array_type(result_a_shape, a_type.get_scalar_type());

                let mut result_b_shape = b_type.get_shape();
                result_b_shape[0] = permutation_values.len() as u64;
                let result_b_type = array_type(result_b_shape, b_type.get_scalar_type());

                let result_a = result.to_vector()?[0].to_flattened_array_u64(result_a_type)?;
                let result_b = result.to_vector()?[1].to_flattened_array_u64(result_b_type)?;
                assert_eq!(&result_a, a_expected);
                assert_eq!(&result_b, b_expected);
                Ok(())
            };
            roles_helper(1, 0)?;
            roles_helper(0, 1)?;
            roles_helper(1, 2)?;
            roles_helper(2, 1)?;
            roles_helper(0, 2)?;
            roles_helper(2, 0)?;
            Ok(())
        };

        data_helper(
            array_type(vec![5], INT32),
            array_type(vec![5], INT16),
            &[1, 2, 3, 4, 5],
            &[10, 20, 30, 40, 50],
            &[1, 0, 3, 4, 2],
            &[2, 1, 4, 5, 3],
            &[20, 10, 40, 50, 30],
        )?;

        data_helper(
            array_type(vec![5], INT32),
            array_type(vec![5], UINT64),
            &[1, 2, 3, 4, 5],
            &[10, 20, 30, 40, 50],
            &[0, 1, 2],
            &[1, 2, 3],
            &[10, 20, 30],
        )?;

        data_helper(
            array_type(vec![5, 2], BIT),
            array_type(vec![5], UINT64),
            &[0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
            &[10, 20, 30, 40, 50],
            &[0, 2, 4, 1],
            &[0, 0, 1, 0, 0, 1, 0, 1],
            &[10, 30, 50, 20],
        )?;

        Ok(())
    }

    #[test]
    fn test_duplication() -> Result<()> {
        let data_helper = |a_type: Type,
                           b_type: Type,
                           a_values: &[u64],
                           b_values: &[u64],
                           duplication_indices: &[u64],
                           a_expected: &[u64],
                           b_expected: &[u64]|
         -> Result<()> {
            // test correct inputs
            let roles_helper = |sender_id: u64, programmer_id: u64| -> Result<()> {
                let num_entries = duplication_indices.len();
                let c = simple_context(|g| {
                    let column_a = g.input(a_type.clone())?;
                    let column_b = g.input(b_type.clone())?;

                    // Generate PRF keys
                    let key_t = array_type(vec![KEY_LENGTH], BIT);
                    let keys_vec = generate_prf_key_triple(g.clone())?;
                    let keys = g.create_tuple(keys_vec)?;
                    // PRF key known only to Sender.
                    let key_s = g.random(key_t)?;
                    // Split input into two shares between Sender and Programmer
                    // Sender generates Programmer's shares
                    let column_a_programmer_share = g.prf(key_s.clone(), 0, a_type.clone())?;
                    let column_b_programmer_share = g.prf(key_s, 0, b_type.clone())?;
                    // Sender computes its shares
                    let column_a_sender_share =
                        column_a.subtract(column_a_programmer_share.clone())?;
                    let column_b_sender_share =
                        column_b.subtract(column_b_programmer_share.clone())?;

                    // Sender packs shares in named tuples and send one of them to Programmer
                    let programmer_share = g
                        .create_named_tuple(vec![
                            ("a".to_owned(), column_a_programmer_share),
                            ("b".to_owned(), column_b_programmer_share),
                        ])?
                        .nop()?
                        .add_annotation(NodeAnnotation::Send(sender_id, programmer_id))?;
                    let sender_share = g.create_named_tuple(vec![
                        ("a".to_owned(), column_a_sender_share),
                        ("b".to_owned(), column_b_sender_share),
                    ])?;

                    // Pack shares together
                    let shares = g.create_tuple(vec![programmer_share, sender_share])?;

                    // Duplication map input
                    let duplication_map = g.input(tuple_type(vec![
                        array_type(vec![num_entries as u64], UINT64),
                        array_type(vec![num_entries as u64], BIT),
                    ]))?;

                    // Duplicated shares
                    let duplicated_shares = g
                        .custom_op(
                            CustomOperation::new(DuplicationMPC {
                                sender_id,
                                programmer_id,
                            }),
                            vec![shares, duplication_map, keys],
                        )?
                        .set_name("Duplication output")?;

                    // Sum duplicated shares
                    let receiver_duplicated_share = duplicated_shares.tuple_get(1)?;
                    let programmer_duplicated_share = duplicated_shares.tuple_get(0)?;

                    let duplicated_column_a = receiver_duplicated_share
                        .named_tuple_get("a".to_owned())?
                        .add(programmer_duplicated_share.named_tuple_get("a".to_owned())?)?;
                    let duplicated_column_b = receiver_duplicated_share
                        .named_tuple_get("b".to_owned())?
                        .add(programmer_duplicated_share.named_tuple_get("b".to_owned())?)?;

                    // Combine duplicated columns
                    g.create_tuple(vec![duplicated_column_a, duplicated_column_b])
                })?;

                let instantiated_c = run_instantiation_pass(c)?.context;
                let inlined_c = inline_operations(
                    &instantiated_c,
                    InlineConfig {
                        default_mode: InlineMode::Simple,
                        ..Default::default()
                    },
                )?
                .get_context();

                let result_hashmap = generate_equivalence_class(
                    &inlined_c,
                    vec![vec![
                        IOStatus::Party(sender_id),
                        IOStatus::Party(sender_id),
                        IOStatus::Party(programmer_id),
                    ]],
                )?;

                let receiver_id = PARTIES as u64 - sender_id - programmer_id;
                // data shared by Sender and Programmer
                let share_r_sp = EquivalenceClasses::Atomic(vec![
                    vec![receiver_id],
                    vec![sender_id, programmer_id],
                ]);
                // data shared by the Receiver and Programmer
                let share_s_rp = EquivalenceClasses::Atomic(vec![
                    vec![sender_id],
                    vec![receiver_id, programmer_id],
                ]);
                // data shared by the Receiver and Sender
                let share_p_rs = EquivalenceClasses::Atomic(vec![
                    vec![programmer_id],
                    vec![receiver_id, sender_id],
                ]);

                let private_pair = vector_class(vec![private_class(); 2]);

                let mut expected_classes = vec![
                    // Prepare protocol inputs
                    // Column A input
                    private_class(),
                    // Column B input
                    private_class(),
                    // First PRF key
                    private_class(),
                    share0_class(),
                    // Second PRF key
                    private_class(),
                    share1_class(),
                    // Third PRF key
                    private_class(),
                    share2_class(),
                    // PRF key triple
                    vector_class(vec![share0_class(), share1_class(), share2_class()]),
                    // PRF key known only to Sender
                    private_class(),
                    // Sender generates Programmer's shares
                    // Column A
                    private_class(),
                    // Column B
                    private_class(),
                    // Sender computes its shares
                    // Column A
                    private_class(),
                    // Column B
                    private_class(),
                    // Sender packs shares in named tuples and send one of them to Programmer
                    // Programmer's share
                    vector_class(vec![private_class(), private_class()]),
                    vector_class(vec![share_r_sp.clone(), share_r_sp.clone()]),
                    // Sender's share
                    vector_class(vec![private_class(), private_class()]),
                    // Pack shares together
                    vector_class(vec![
                        vector_class(vec![share_r_sp.clone(), share_r_sp.clone()]),
                        vector_class(vec![private_class(), private_class()]),
                    ]),
                    // Duplication map input
                    vector_class(vec![private_class(), private_class()]),
                ];

                // Start of Duplication protocol
                // Extraction of inputs
                expected_classes.extend(vec![
                    // Extract duplication indices
                    private_class(),
                    // Extract duplication bits
                    private_class(),
                    // Extract PRF key known to Sender and Programmer
                    share_r_sp.clone(),
                    // Extract PRF key known to Programmer and Receiver
                    share_s_rp.clone(),
                    // Extract PRF key known to Sender and Receiver
                    share_p_rs.clone(),
                    // Extract Programmer's share
                    vector_class(vec![share_r_sp.clone(), share_r_sp.clone()]),
                    // Extract Sender's share
                    vector_class(vec![private_class(), private_class()]),
                ]);

                // Execute Duplication protocol for every column
                let mut add_column_class = |t: Type| -> Result<()> {
                    expected_classes.extend(vec![
                        // Sender extracts сolumn
                        private_class(),
                        // Sender and Receiver generate random B_r[i] for i in [1,num_entries], W_0 and W_1 of size of an input share.
                        // B_r[i]
                        share_p_rs.clone(),
                        // W0
                        share_p_rs.clone(),
                        // W1
                        share_p_rs.clone(),
                        // Sender selects the first entry share
                        private_class(),
                        // and masks it with a random mask B_p[0] known also to Programmer.
                        share_r_sp.clone(),
                        // The result is assigned to B_r[0].
                        private_class(),
                        private_class(),
                        // B_r is sent to Receiver
                        share_p_rs.clone(),
                        // Merge B_r[0] and B_r[i] for i in [1,num_entries]
                        share_p_rs.clone(),
                        // Sender and Programmer generate a random mask phi of the duplication map
                        share_r_sp.clone(),
                        // Sender computes two columns M0 and M1 such that
                        //
                        //    M0[i] = sender_column[i] - B_r[i] - W_(duplication_bits[i])[i],
                        //    M1[i] = B_r[i-1] - B_r[i] - W_(1-duplication_bits[i])[i]
                        //
                        // for i in [1, num_entries]
                        // B_r without first entry
                        share_p_rs.clone(),
                        // B_r without last entry
                        share_p_rs.clone(),
                        // Reshape duplication bits and phi to enable broadcasting
                        // Duplication bits without first entry
                        private_class(),
                    ]);

                    if t.get_shape().len() > 1 {
                        // Reshaped duplication bits without first entry
                        expected_classes.push(private_class());
                        // Reshaped phi
                        expected_classes.push(share_r_sp.clone());
                    }

                    expected_classes.extend(vec![
                        // Select W_(phi[i])[i]
                        // Difference of W1 and W0
                        share_p_rs.clone(),
                        // Multiply by bits of phi
                        private_class(),
                        // Add W0
                        private_class(),
                        // Selected W using bits of NOT phi
                        // Difference of W0 and W1
                        share_p_rs.clone(),
                        // Multiply by bits of phi
                        private_class(),
                        // Add W1
                        private_class(),
                        // Sender computes M0
                        private_class(),
                        private_class(),
                        private_class(),
                        // Sender computes M1
                        // B_r[i-1] - B_r[i]
                        share_p_rs.clone(),
                        private_class(),
                        // Sender sends M0 to Programmer
                        share_r_sp.clone(),
                        // Sender sends M1 to Programmer
                        share_r_sp.clone(),
                        // Programmer and Receiver generate a random value R of size of an input share
                        share_s_rp.clone(),
                        // Programmer masks the duplication map by computing rho = phi XOR dup_map except for the first bit.
                        private_class(),
                        // Programmer sends rho to Receiver
                        share_s_rp.clone(),
                        // Receiver selects W_(rho[i])[i] for i in [1, num_entries] and sends them to Programmer
                        // Difference W1 and W0
                        share_p_rs.clone(),
                        // Multiply by bits of rho
                        private_class(),
                        // Add W0
                        private_class(),
                        // Receiver sends to Programmer
                        share_s_rp.clone(),
                        // Programmer computes
                        //
                        // B_p[i] = M_(duplication_bits[i])[i] + W_(rho[i])[i] + duplication_bits[i] * B_p[i-1]
                        //
                        // for i in {1,..., num_entries-1}.
                        // Compute M_(duplication_bits[i])[i]
                        // Difference M1 and M0
                        share_r_sp.clone(),
                        // Multiply by duplication bits
                        private_class(),
                        // Add M0
                        private_class(),
                        // Add W_(rho[i])[i]
                        private_class(),
                        // Reshape duplication bits
                        private_class(),
                        // Compute iteration to get B_p[i] for i in {1,..., num_entries-1}
                        private_class(),
                        // Compute the share of Programmer which is equal to
                        // B_p - R + duplication_map(programmer column share)
                        // B_p - R
                        private_class(),
                        // Extract Programmer's column share
                        share_r_sp.clone(),
                        // duplication_map(programmer column share)
                        private_class(),
                        // B_p - R + duplication_map(programmer column share)
                        private_class(),
                        // Receiver resulting column share B_r + R
                        private_class(),
                    ]);
                    Ok(())
                };
                add_column_class(a_type.clone())?;
                add_column_class(b_type.clone())?;

                // Create result Receiver's share
                expected_classes.push(vector_class(vec![private_class(), private_class()]));
                // Create result Programmer's share
                expected_classes.push(vector_class(vec![private_class(), private_class()]));
                // Final result
                expected_classes.push(vector_class(vec![
                    vector_class(vec![private_class(), private_class()]),
                    vector_class(vec![private_class(), private_class()]),
                ]));

                let mut nodes_classes = vec![];
                for i in 0..expected_classes.len() as u64 {
                    nodes_classes.push((*result_hashmap.get(&(0, i)).unwrap()).clone());
                }

                assert_eq!(nodes_classes, expected_classes);

                // Check the ownership of the protocol output
                let output_node_id = inlined_c
                    .get_main_graph()?
                    .retrieve_node("Duplication output")?
                    .get_global_id();
                assert_eq!(
                    result_hashmap.get(&output_node_id).unwrap(),
                    &vector_class(vec![private_pair; 2])
                );

                // Check evaluation
                let mut duplication_bits = vec![0u64; num_entries];
                for i in 1..num_entries {
                    if duplication_indices[i] == duplication_indices[i - 1] {
                        duplication_bits[i] = 1;
                    }
                }
                let result = random_evaluate(
                    inlined_c.get_main_graph()?,
                    vec![
                        Value::from_flattened_array(a_values, a_type.get_scalar_type())?,
                        Value::from_flattened_array(b_values, b_type.get_scalar_type())?,
                        Value::from_vector(vec![
                            Value::from_flattened_array(duplication_indices, UINT64)?,
                            Value::from_flattened_array(&duplication_bits, BIT)?,
                        ]),
                    ],
                )?;
                let mut result_a_shape = a_type.get_shape();
                result_a_shape[0] = num_entries as u64;
                let result_a_type = array_type(result_a_shape, a_type.get_scalar_type());

                let mut result_b_shape = b_type.get_shape();
                result_b_shape[0] = num_entries as u64;
                let result_b_type = array_type(result_b_shape, b_type.get_scalar_type());

                let result_a = result.to_vector()?[0].to_flattened_array_u64(result_a_type)?;
                let result_b = result.to_vector()?[1].to_flattened_array_u64(result_b_type)?;
                assert_eq!(&result_a, a_expected);
                assert_eq!(&result_b, b_expected);
                Ok(())
            };
            roles_helper(1, 0)?;
            roles_helper(0, 1)?;
            roles_helper(1, 2)?;
            roles_helper(2, 1)?;
            roles_helper(0, 2)?;
            roles_helper(2, 0)?;
            Ok(())
        };

        data_helper(
            array_type(vec![5], INT32),
            array_type(vec![5], INT16),
            &[1, 2, 3, 4, 5],
            &[10, 20, 30, 40, 50],
            &[0, 1, 2, 3, 4],
            &[1, 2, 3, 4, 5],
            &[10, 20, 30, 40, 50],
        )?;

        data_helper(
            array_type(vec![5], INT32),
            array_type(vec![5], INT16),
            &[1, 2, 3, 4, 5],
            &[10, 20, 30, 40, 50],
            &[0, 1, 1, 3, 4],
            &[1, 2, 2, 4, 5],
            &[10, 20, 20, 40, 50],
        )?;

        data_helper(
            array_type(vec![5], INT32),
            array_type(vec![5], UINT64),
            &[1, 2, 3, 4, 5],
            &[10, 20, 30, 40, 50],
            &[0, 0, 0, 0, 0],
            &[1, 1, 1, 1, 1],
            &[10, 10, 10, 10, 10],
        )?;

        data_helper(
            array_type(vec![5, 2], INT32),
            array_type(vec![5], UINT64),
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            &[10, 20, 30, 40, 50],
            &[0, 1, 1, 3, 4],
            &[1, 2, 3, 4, 3, 4, 7, 8, 9, 10],
            &[10, 20, 20, 40, 50],
        )?;

        Ok(())
    }

    fn evaluate_join_helper(
        join_t: JoinType,
        test_info: JoinTestInfo,
        c: Context,
        prng_seed: Option<[u8; SEED_SIZE]>,
        num_trials: u64,
    ) -> Result<()> {
        let g = c.get_main_graph()?;
        let o = g.get_output_node()?;
        let result_t = o.get_type()?;
        for _ in 0..num_trials {
            let result = evaluate_simple_evaluator(
                g.clone(),
                vec![test_info.set0.get_value()?, test_info.set1.get_value()?],
                prng_seed,
            )?
            .to_vector()?;

            let headers_types = result_t.get_named_types()?;

            assert_eq!(test_info.expected[&join_t].len(), headers_types.len());
            for (i, (expected_header, expected_mask, expected_column)) in
                test_info.expected[&join_t].iter().enumerate()
            {
                assert_eq!(*expected_header, headers_types[i].0);
                if expected_header == NULL_HEADER || !test_info.has_column_masks {
                    assert_eq!(
                        result[i].to_flattened_array_u64((headers_types[i].1).clone())?,
                        *expected_column
                    );
                    continue;
                }
                if let Some(mask) = expected_mask {
                    let column_t =
                        ColumnType::new((headers_types[i].1).clone(), true, expected_header)?;
                    let mask_data = result[i].to_vector()?;
                    assert_eq!(
                        mask_data[0].to_flattened_array_u64(column_t.get_mask_type()?)?,
                        *mask
                    );
                    assert_eq!(
                        mask_data[1].to_flattened_array_u64(column_t.get_data_type())?,
                        *expected_column
                    )
                }
            }
        }

        Ok(())
    }

    fn join_context(
        join_t: JoinType,
        test_info: &JoinTestInfo,
        is_x_private: bool,
        is_y_private: bool,
    ) -> Result<Context> {
        // test correct inputs
        let c = simple_context(|g| {
            let i0 = g.input(test_info.set0.get_type())?;
            let i1 = g.input(test_info.set1.get_type())?;
            if test_info.has_column_masks {
                i0.join_with_column_masks(i1, join_t, test_info.headers.clone())
            } else {
                i0.join(i1, join_t, test_info.headers.clone())
            }
        })?;

        let mut input_parties = vec![];
        if is_x_private {
            input_parties.push(IOStatus::Party(0));
        } else {
            input_parties.push(IOStatus::Public);
        }
        if is_y_private {
            input_parties.push(IOStatus::Party(0));
        } else {
            input_parties.push(IOStatus::Public);
        }

        Ok(prepare_for_mpc_evaluation(
            &c,
            vec![input_parties],
            vec![vec![IOStatus::Party(0)]],
            InlineConfig {
                default_mode: InlineMode::DepthOptimized(DepthOptimizationLevel::Default),
                ..Default::default()
            },
        )?
        .get_context())
    }

    fn deterministic_join_helper(
        join_t: JoinType,
        test_info: JoinTestInfo,
        is_x_private: bool,
        is_y_private: bool,
    ) -> Result<()> {
        let c = join_context(join_t, &test_info, is_x_private, is_y_private)?;

        let prng_seed: [u8; SEED_SIZE] = core::array::from_fn(|i| i as u8);
        evaluate_join_helper(join_t, test_info, c, Some(prng_seed), 1)
    }

    fn private_join_data_helper(join_t: JoinType, join_test_info: JoinTestInfo) -> Result<()> {
        deterministic_join_helper(join_t, join_test_info, true, true)
    }

    fn get_join_test_setup(with_masks: bool, index: usize) -> Result<JoinTestInfo> {
        let tests_without_masks = vec![
            // Setup 0
            join_info(
                vec![
                    column_info(NULL_HEADER, &[5], BIT, &[1, 1, 1, 1, 1])?,
                    column_info("a", &[5], INT8, &[1, 2, 3, 4, 5])?,
                    column_info("b", &[5], INT8, &[10, 20, 30, 40, 50])?,
                ],
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 1, 1, 1, 1, 1])?,
                    column_info("c", &[6], INT8, &[30, 21, 40, 41, 51, 61])?,
                    column_info("d", &[6], INT16, &[300, 210, 400, 410, 510, 610])?,
                ],
                vec![("b", "c")],
                {
                    let mut expected = HashMap::new();
                    expected.insert(
                        JoinType::Inner,
                        expected_set_info(vec![
                            (NULL_HEADER, &[0, 0, 1, 1, 0]),
                            ("a", &[0, 0, 3, 4, 0]),
                            ("b", &[0, 0, 30, 40, 0]),
                            ("d", &[0, 0, 300, 400, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Left,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 1, 1, 1, 1]),
                            ("a", &[1, 2, 3, 4, 5]),
                            ("b", &[10, 20, 30, 40, 50]),
                            ("d", &[0, 0, 300, 400, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Union,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
                            ("a", &[1, 2, 0, 0, 5, 0, 0, 0, 0, 0, 0]),
                            ("b", &[10, 20, 0, 0, 50, 30, 21, 40, 41, 51, 61]),
                            ("d", &[0, 0, 0, 0, 0, 300, 210, 400, 410, 510, 610]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Full,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
                            ("a", &[1, 2, 0, 0, 5, 3, 0, 4, 0, 0, 0]),
                            ("b", &[10, 20, 0, 0, 50, 30, 21, 40, 41, 51, 61]),
                            ("d", &[0, 0, 0, 0, 0, 300, 210, 400, 410, 510, 610]),
                        ]),
                    );
                    expected
                },
            ),
            // Setup 1
            join_info(
                vec![
                    column_info(NULL_HEADER, &[5], BIT, &[1, 1, 1, 0, 1])?,
                    column_info("a", &[5], INT8, &[1, 2, 3, 4, 5])?,
                    column_info(
                        "b",
                        &[5, 4],
                        BIT,
                        &array!([
                            [0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 1, 1],
                            [0, 1, 0, 0],
                            [0, 1, 0, 1]
                        ])
                        .into_raw_vec(),
                    )?,
                ],
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 0, 1, 1, 1, 1])?,
                    column_info(
                        "b",
                        &[6, 4],
                        BIT,
                        &array!([
                            [0, 0, 1, 1],
                            [0, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 1, 1, 0],
                            [0, 1, 1, 1],
                            [1, 0, 0, 0]
                        ])
                        .into_raw_vec(),
                    )?,
                    column_info("c", &[6], INT16, &[300, 210, 400, 410, 510, 610])?,
                ],
                vec![("b", "b")],
                {
                    let mut expected = HashMap::new();
                    expected.insert(
                        JoinType::Inner,
                        expected_set_info(vec![
                            (NULL_HEADER, &[0, 0, 1, 0, 0]),
                            ("a", &[0, 0, 3, 0, 0]),
                            (
                                "b",
                                &array!([
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 1, 1],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0]
                                ])
                                .into_raw_vec(),
                            ),
                            ("c", &[0, 0, 300, 0, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Left,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 1, 1, 0, 1]),
                            ("a", &[1, 2, 3, 0, 5]),
                            (
                                "b",
                                &array!([
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 1, 1],
                                    [0, 0, 0, 0],
                                    [0, 1, 0, 1]
                                ])
                                .into_raw_vec(),
                            ),
                            ("c", &[0, 0, 300, 0, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Union,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1]),
                            ("a", &[1, 2, 0, 0, 5, 0, 0, 0, 0, 0, 0]),
                            (
                                "b",
                                &array!([
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 1],
                                    [0, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 1, 1, 0],
                                    [0, 1, 1, 1],
                                    [1, 0, 0, 0]
                                ])
                                .into_raw_vec(),
                            ),
                            ("c", &[0, 0, 0, 0, 0, 300, 0, 400, 410, 510, 610]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Full,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1]),
                            ("a", &[1, 2, 0, 0, 5, 3, 0, 0, 0, 0, 0]),
                            (
                                "b",
                                &array!([
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 1],
                                    [0, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 1, 1, 0],
                                    [0, 1, 1, 1],
                                    [1, 0, 0, 0]
                                ])
                                .into_raw_vec(),
                            ),
                            ("c", &[0, 0, 0, 0, 0, 300, 0, 400, 410, 510, 610]),
                        ]),
                    );
                    expected
                },
            ),
            // Setup 2
            join_info(
                vec![
                    column_info(NULL_HEADER, &[5], BIT, &[1, 1, 1, 1, 1])?,
                    column_info("a", &[5], INT8, &[1, 2, 3, 4, 5])?,
                    column_info(
                        "b",
                        &[5, 4],
                        BIT,
                        &array!([
                            [0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 1, 1],
                            [0, 1, 0, 0],
                            [0, 1, 0, 1]
                        ])
                        .into_raw_vec(),
                    )?,
                    column_info("c", &[5], INT16, &[100, 200, 300, 400, 500])?,
                ],
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 1, 1, 1, 1, 1])?,
                    column_info(
                        "d",
                        &[6, 4],
                        BIT,
                        &array!([
                            [0, 0, 1, 1],
                            [0, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 1, 1, 0],
                            [0, 1, 1, 1],
                            [1, 0, 0, 0]
                        ])
                        .into_raw_vec(),
                    )?,
                    column_info("e", &[6], INT16, &[300, 210, 400, 410, 510, 610])?,
                    column_info("f", &[6, 2], BIT, &[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0])?,
                ],
                vec![("b", "d"), ("c", "e")],
                {
                    let mut expected = HashMap::new();
                    expected.insert(
                        JoinType::Inner,
                        expected_set_info(vec![
                            (NULL_HEADER, &[0, 0, 1, 1, 0]),
                            ("a", &[0, 0, 3, 4, 0]),
                            (
                                "b",
                                &array!([
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 1, 1],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 0]
                                ])
                                .into_raw_vec(),
                            ),
                            ("c", &[0, 0, 300, 400, 0]),
                            ("f", &[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Left,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 1, 1, 1, 1]),
                            ("a", &[1, 2, 3, 4, 5]),
                            (
                                "b",
                                &array!([
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 1, 1],
                                    [0, 1, 0, 0],
                                    [0, 1, 0, 1]
                                ])
                                .into_raw_vec(),
                            ),
                            ("c", &[100, 200, 300, 400, 500]),
                            ("f", &[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Union,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
                            ("a", &[1, 2, 0, 0, 5, 0, 0, 0, 0, 0, 0]),
                            (
                                "b",
                                &array!([
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 1],
                                    [0, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 1, 1, 0],
                                    [0, 1, 1, 1],
                                    [1, 0, 0, 0]
                                ])
                                .into_raw_vec(),
                            ),
                            ("c", &[100, 200, 0, 0, 500, 300, 210, 400, 410, 510, 610]),
                            (
                                "f",
                                &[
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0,
                                    0,
                                ],
                            ),
                        ]),
                    );
                    expected.insert(
                        JoinType::Full,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
                            ("a", &[1, 2, 0, 0, 5, 3, 0, 4, 0, 0, 0]),
                            (
                                "b",
                                &array!([
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 1],
                                    [0, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 1, 1, 0],
                                    [0, 1, 1, 1],
                                    [1, 0, 0, 0]
                                ])
                                .into_raw_vec(),
                            ),
                            ("c", &[100, 200, 0, 0, 500, 300, 210, 400, 410, 510, 610]),
                            (
                                "f",
                                &[
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0,
                                    0,
                                ],
                            ),
                        ]),
                    );
                    expected
                },
            ),
            // Setup 3
            join_info(
                vec![
                    column_info(NULL_HEADER, &[5], BIT, &[1, 1, 0, 1, 1])?,
                    column_info("a", &[5], INT8, &[1, 2, 3, 4, 5])?,
                    column_info(
                        "b",
                        &[5, 2],
                        INT8,
                        &array!([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]).into_raw_vec(),
                    )?,
                    column_info("c", &[5], INT16, &[100, 200, 300, 400, 500])?,
                ],
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 0, 1, 1, 1, 0])?,
                    column_info(
                        "b",
                        &[6, 2],
                        INT8,
                        &array!([[30, 30], [21, 21], [40, 40], [41, 41], [51, 51], [61, 61]])
                            .into_raw_vec(),
                    )?,
                    column_info("c", &[6], INT16, &[300, 210, 400, 410, 510, 610])?,
                    column_info("d", &[6], BIT, &[0, 1, 1, 0, 1, 0])?,
                ],
                vec![("b", "b"), ("c", "c")],
                {
                    let mut expected = HashMap::new();
                    expected.insert(
                        JoinType::Inner,
                        expected_set_info(vec![
                            (NULL_HEADER, &[0, 0, 0, 1, 0]),
                            ("a", &[0, 0, 0, 4, 0]),
                            (
                                "b",
                                &array!([[0, 0], [0, 0], [0, 0], [40, 40], [0, 0]]).into_raw_vec(),
                            ),
                            ("c", &[0, 0, 0, 400, 0]),
                            ("d", &[0, 0, 0, 1, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Left,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 1, 0, 1, 1]),
                            ("a", &[1, 2, 0, 4, 5]),
                            (
                                "b",
                                &array!([[10, 10], [20, 20], [0, 0], [40, 40], [50, 50]])
                                    .into_raw_vec(),
                            ),
                            ("c", &[100, 200, 0, 400, 500]),
                            ("d", &[0, 0, 0, 1, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Union,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0]),
                            ("a", &[1, 2, 0, 0, 5, 0, 0, 0, 0, 0, 0]),
                            (
                                "b",
                                &array!([
                                    [10, 10],
                                    [20, 20],
                                    [0, 0],
                                    [0, 0],
                                    [50, 50],
                                    [30, 30],
                                    [0, 0],
                                    [40, 40],
                                    [41, 41],
                                    [51, 51],
                                    [0, 0]
                                ])
                                .into_raw_vec(),
                            ),
                            ("c", &[100, 200, 0, 0, 500, 300, 0, 400, 410, 510, 0]),
                            ("d", &[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Full,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0]),
                            ("a", &[1, 2, 0, 0, 5, 0, 0, 4, 0, 0, 0]),
                            (
                                "b",
                                &array!([
                                    [10, 10],
                                    [20, 20],
                                    [0, 0],
                                    [0, 0],
                                    [50, 50],
                                    [30, 30],
                                    [0, 0],
                                    [40, 40],
                                    [41, 41],
                                    [51, 51],
                                    [0, 0]
                                ])
                                .into_raw_vec(),
                            ),
                            ("c", &[100, 200, 0, 0, 500, 300, 0, 400, 410, 510, 0]),
                            ("d", &[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]),
                        ]),
                    );
                    expected
                },
            ),
            // Setup 4
            join_info(
                vec![
                    column_info(NULL_HEADER, &[5], BIT, &[1, 1, 1, 1, 1])?,
                    column_info("a", &[5], INT8, &[1, 2, 3, 4, 5])?,
                    column_info("b", &[5], INT8, &[10, 20, 30, 40, 50])?,
                    column_info("c", &[5], INT16, &[100, 200, 300, 400, 500])?,
                ],
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 1, 1, 1, 1, 1])?,
                    column_info("b", &[6], INT8, &[60, 70, 80, 90, 100, 110])?,
                    column_info("c", &[6], INT16, &[600, 700, 800, 900, 1000, 1100])?,
                    column_info("d", &[6], BIT, &[0, 1, 1, 0, 1, 0])?,
                ],
                vec![("b", "b"), ("c", "c")],
                {
                    let mut expected = HashMap::new();
                    expected.insert(
                        JoinType::Inner,
                        expected_set_info(vec![
                            (NULL_HEADER, &[0, 0, 0, 0, 0]),
                            ("a", &[0, 0, 0, 0, 0]),
                            ("b", &[0, 0, 0, 0, 0]),
                            ("c", &[0, 0, 0, 0, 0]),
                            ("d", &[0, 0, 0, 0, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Left,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 1, 1, 1, 1]),
                            ("a", &[1, 2, 3, 4, 5]),
                            ("b", &[10, 20, 30, 40, 50]),
                            ("c", &[100, 200, 300, 400, 500]),
                            ("d", &[0, 0, 0, 0, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Union,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                            ("a", &[1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0]),
                            ("b", &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]),
                            (
                                "c",
                                &[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100],
                            ),
                            ("d", &[0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Full,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                            ("a", &[1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0]),
                            ("b", &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]),
                            (
                                "c",
                                &[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100],
                            ),
                            ("d", &[0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]),
                        ]),
                    );
                    expected
                },
            ),
            // Setup 5
            join_info(
                vec![
                    column_info(NULL_HEADER, &[5], BIT, &[0, 0, 0, 1, 1])?,
                    column_info("a", &[5], INT8, &[1, 2, 3, 4, 5])?,
                    column_info("b", &[5], INT8, &[10, 20, 30, 40, 50])?,
                    column_info("c", &[5], INT16, &[100, 200, 300, 400, 500])?,
                ],
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 1, 1, 0, 0, 0])?,
                    column_info("b", &[6], INT8, &[60, 70, 80, 90, 100, 110])?,
                    column_info("c", &[6], INT16, &[600, 700, 800, 900, 1000, 1100])?,
                    column_info("d", &[6], BIT, &[0, 1, 1, 0, 1, 0])?,
                ],
                vec![("b", "b"), ("c", "c")],
                {
                    let mut expected = HashMap::new();
                    expected.insert(
                        JoinType::Inner,
                        expected_set_info(vec![
                            (NULL_HEADER, &[0, 0, 0, 0, 0]),
                            ("a", &[0, 0, 0, 0, 0]),
                            ("b", &[0, 0, 0, 0, 0]),
                            ("c", &[0, 0, 0, 0, 0]),
                            ("d", &[0, 0, 0, 0, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Left,
                        expected_set_info(vec![
                            (NULL_HEADER, &[0, 0, 0, 1, 1]),
                            ("a", &[0, 0, 0, 4, 5]),
                            ("b", &[0, 0, 0, 40, 50]),
                            ("c", &[0, 0, 0, 400, 500]),
                            ("d", &[0, 0, 0, 0, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Union,
                        expected_set_info(vec![
                            (NULL_HEADER, &[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]),
                            ("a", &[0, 0, 0, 4, 5, 0, 0, 0, 0, 0, 0]),
                            ("b", &[0, 0, 0, 40, 50, 60, 70, 80, 0, 0, 0]),
                            ("c", &[0, 0, 0, 400, 500, 600, 700, 800, 0, 0, 0]),
                            ("d", &[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Full,
                        expected_set_info(vec![
                            (NULL_HEADER, &[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]),
                            ("a", &[0, 0, 0, 4, 5, 0, 0, 0, 0, 0, 0]),
                            ("b", &[0, 0, 0, 40, 50, 60, 70, 80, 0, 0, 0]),
                            ("c", &[0, 0, 0, 400, 500, 600, 700, 800, 0, 0, 0]),
                            ("d", &[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]),
                        ]),
                    );
                    expected
                },
            ),
            // Setup 6
            join_info(
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1])?,
                    column_info("a", &[1], INT8, &[10])?,
                ],
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1])?,
                    column_info("b", &[1], INT8, &[10])?,
                ],
                vec![("a", "b")],
                {
                    let mut expected = HashMap::new();
                    expected.insert(
                        JoinType::Inner,
                        expected_set_info(vec![(NULL_HEADER, &[1]), ("a", &[10])]),
                    );
                    expected.insert(
                        JoinType::Left,
                        expected_set_info(vec![(NULL_HEADER, &[1]), ("a", &[10])]),
                    );
                    expected.insert(
                        JoinType::Union,
                        expected_set_info(vec![(NULL_HEADER, &[0, 1]), ("a", &[0, 10])]),
                    );
                    expected.insert(
                        JoinType::Full,
                        expected_set_info(vec![(NULL_HEADER, &[0, 1]), ("a", &[0, 10])]),
                    );
                    expected
                },
            ),
            // Setup 7
            join_info(
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1])?,
                    column_info("a", &[1], INT8, &[2])?,
                    column_info("b", &[1], INT8, &[3])?,
                    column_info("c", &[1], INT8, &[4])?,
                ],
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1])?,
                    column_info("b", &[1], INT8, &[3])?,
                    column_info("a", &[1], INT8, &[2])?,
                ],
                vec![("a", "a"), ("b", "b")],
                {
                    let mut expected = HashMap::new();
                    expected.insert(
                        JoinType::Inner,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1]),
                            ("a", &[2]),
                            ("b", &[3]),
                            ("c", &[4]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Left,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1]),
                            ("a", &[2]),
                            ("b", &[3]),
                            ("c", &[4]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Union,
                        expected_set_info(vec![
                            (NULL_HEADER, &[0, 1]),
                            ("a", &[0, 2]),
                            ("b", &[0, 3]),
                            ("c", &[0, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Full,
                        expected_set_info(vec![
                            (NULL_HEADER, &[0, 1]),
                            ("a", &[0, 2]),
                            ("b", &[0, 3]),
                            ("c", &[0, 4]),
                        ]),
                    );
                    expected
                },
            ),
            // Setup 8
            join_info(
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1])?,
                    column_info("a", &[1], INT8, &[2])?,
                    column_info("b", &[1], INT8, &[3])?,
                    column_info("c", &[1], INT8, &[4])?,
                ],
                vec![
                    column_info(NULL_HEADER, &[5], BIT, &[1, 1, 1, 1, 1])?,
                    column_info("b", &[5], INT8, &[3, 4, 5, 6, 7])?,
                    column_info("a", &[5], INT8, &[2, 3, 4, 5, 6])?,
                ],
                vec![("a", "a"), ("b", "b")],
                {
                    let mut expected = HashMap::new();
                    expected.insert(
                        JoinType::Inner,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1]),
                            ("a", &[2]),
                            ("b", &[3]),
                            ("c", &[4]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Left,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1]),
                            ("a", &[2]),
                            ("b", &[3]),
                            ("c", &[4]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Union,
                        expected_set_info(vec![
                            (NULL_HEADER, &[0, 1, 1, 1, 1, 1]),
                            ("a", &[0, 2, 3, 4, 5, 6]),
                            ("b", &[0, 3, 4, 5, 6, 7]),
                            ("c", &[0, 0, 0, 0, 0, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Full,
                        expected_set_info(vec![
                            (NULL_HEADER, &[0, 1, 1, 1, 1, 1]),
                            ("a", &[0, 2, 3, 4, 5, 6]),
                            ("b", &[0, 3, 4, 5, 6, 7]),
                            ("c", &[0, 4, 0, 0, 0, 0]),
                        ]),
                    );
                    expected
                },
            ),
            // Setup 9
            join_info(
                vec![
                    column_info(NULL_HEADER, &[5], BIT, &[1, 1, 1, 1, 1])?,
                    column_info("b", &[5], INT8, &[3, 4, 5, 6, 7])?,
                    column_info("a", &[5], INT8, &[2, 3, 4, 5, 6])?,
                ],
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1])?,
                    column_info("a", &[1], INT8, &[2])?,
                    column_info("b", &[1], INT8, &[3])?,
                    column_info("c", &[1], INT8, &[4])?,
                ],
                vec![("a", "a"), ("b", "b")],
                {
                    let mut expected = HashMap::new();
                    expected.insert(
                        JoinType::Inner,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 0, 0, 0, 0]),
                            ("b", &[3, 0, 0, 0, 0]),
                            ("a", &[2, 0, 0, 0, 0]),
                            ("c", &[4, 0, 0, 0, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Left,
                        expected_set_info(vec![
                            (NULL_HEADER, &[1, 1, 1, 1, 1]),
                            ("b", &[3, 4, 5, 6, 7]),
                            ("a", &[2, 3, 4, 5, 6]),
                            ("c", &[4, 0, 0, 0, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Union,
                        expected_set_info(vec![
                            (NULL_HEADER, &[0, 1, 1, 1, 1, 1]),
                            ("b", &[0, 4, 5, 6, 7, 3]),
                            ("a", &[0, 3, 4, 5, 6, 2]),
                            ("c", &[0, 0, 0, 0, 0, 4]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Full,
                        expected_set_info(vec![
                            (NULL_HEADER, &[0, 1, 1, 1, 1, 1]),
                            ("b", &[0, 4, 5, 6, 7, 3]),
                            ("a", &[0, 3, 4, 5, 6, 2]),
                            ("c", &[0, 0, 0, 0, 0, 4]),
                        ]),
                    );
                    expected
                },
            ),
        ];

        let tests_with_masks: Vec<JoinTestInfo> = vec![
            // Setup 0
            join_info(
                vec![
                    column_info_with_mask(NULL_HEADER, &[2], BIT, None, &[1, 1])?,
                    column_info_with_mask("Income1", &[2], UINT8, Some(&[1, 1]), &[40, 50])?,
                    column_info_with_mask(
                        "Outcome1",
                        &[2, 2],
                        UINT16,
                        Some(&[1, 1]),
                        &[400, 401, 500, 501],
                    )?,
                    column_info_with_mask("ID", &[2], UINT8, Some(&[1, 0]), &[4, 5])?,
                ],
                vec![
                    column_info_with_mask(NULL_HEADER, &[2], BIT, None, &[1, 1])?,
                    column_info_with_mask("ID", &[2], UINT8, Some(&[1, 1]), &[5, 3])?,
                    column_info_with_mask("Income2", &[2], UINT8, Some(&[1, 1]), &[50, 30])?,
                    column_info_with_mask(
                        "Outcome2",
                        &[2, 2],
                        UINT16,
                        Some(&[1, 1]),
                        &[500, 501, 300, 301],
                    )?,
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                {
                    let mut expected = HashMap::new();
                    expected.insert(
                        JoinType::Inner,
                        expected_set_info_with_mask(vec![
                            (NULL_HEADER, None, &[0, 0]),
                            ("Income1", Some(&[0, 0]), &[0, 0]),
                            ("Outcome1", Some(&[0, 0]), &[0, 0, 0, 0]),
                            ("ID", Some(&[0, 0]), &[0, 0]),
                            ("Outcome2", Some(&[0, 0]), &[0, 0, 0, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Left,
                        expected_set_info_with_mask(vec![
                            (NULL_HEADER, None, &[1, 1]),
                            ("Income1", Some(&[1, 1]), &[40, 50]),
                            ("Outcome1", Some(&[1, 1]), &[400, 401, 500, 501]),
                            ("ID", Some(&[1, 0]), &[4, 0]),
                            ("Outcome2", Some(&[0, 0]), &[0, 0, 0, 0]),
                        ]),
                    );
                    expected.insert(
                        JoinType::Union,
                        expected_set_info_with_mask(vec![
                            (NULL_HEADER, None, &[1, 1, 1, 1]),
                            ("Income1", Some(&[1, 1, 1, 1]), &[40, 50, 50, 30]),
                            (
                                "Outcome1",
                                Some(&[1, 1, 0, 0]),
                                &[400, 401, 500, 501, 0, 0, 0, 0],
                            ),
                            ("ID", Some(&[1, 0, 1, 1]), &[4, 0, 5, 3]),
                            (
                                "Outcome2",
                                Some(&[0, 0, 1, 1]),
                                &[0, 0, 0, 0, 500, 501, 300, 301],
                            ),
                        ]),
                    );
                    expected.insert(
                        JoinType::Full,
                        expected_set_info_with_mask(vec![
                            (NULL_HEADER, None, &[1, 1, 1, 1]),
                            ("Income1", Some(&[1, 1, 1, 1]), &[40, 50, 50, 30]),
                            (
                                "Outcome1",
                                Some(&[1, 1, 0, 0]),
                                &[400, 401, 500, 501, 0, 0, 0, 0],
                            ),
                            ("ID", Some(&[1, 0, 1, 1]), &[4, 0, 5, 3]),
                            (
                                "Outcome2",
                                Some(&[0, 0, 1, 1]),
                                &[0, 0, 0, 0, 500, 501, 300, 301],
                            ),
                        ]),
                    );
                    expected
                },
            ),
            // Setup 1
            join_info(
                vec![
                    column_info_with_mask(NULL_HEADER, &[4], BIT, None, &[1, 1, 1, 1])?,
                    column_info_with_mask(
                        "Income1",
                        &[4],
                        UINT8,
                        Some(&[1, 1, 1, 1]),
                        &[40, 50, 20, 30],
                    )?,
                    column_info_with_mask(
                        "Outcome1",
                        &[4, 2],
                        UINT16,
                        Some(&[1, 1, 1, 0]),
                        &[400, 401, 500, 501, 200, 201, 300, 301],
                    )?,
                    column_info_with_mask("ID", &[4], UINT8, Some(&[1, 0, 1, 1]), &[4, 5, 2, 3])?,
                ],
                vec![
                    column_info_with_mask(NULL_HEADER, &[4], BIT, None, &[1, 1, 1, 1])?,
                    column_info_with_mask("ID", &[4], UINT8, Some(&[1, 1, 1, 1]), &[5, 3, 6, 7])?,
                    column_info_with_mask(
                        "Income2",
                        &[4],
                        UINT8,
                        Some(&[1, 1, 1, 1]),
                        &[50, 30, 60, 70],
                    )?,
                    column_info_with_mask(
                        "Outcome2",
                        &[4, 2],
                        UINT16,
                        Some(&[1, 1, 1, 1]),
                        &[500, 501, 300, 301, 600, 601, 700, 701],
                    )?,
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                {
                    let mut expected = HashMap::new();
                    expected.insert(
                        JoinType::Inner,
                        expected_set_info_with_mask(vec![
                            (NULL_HEADER, None, &[0, 0, 0, 1]),
                            ("Income1", Some(&[0, 0, 0, 1]), &[0, 0, 0, 30]),
                            ("Outcome1", Some(&[0, 0, 0, 0]), &[0, 0, 0, 0, 0, 0, 0, 0]),
                            ("ID", Some(&[0, 0, 0, 1]), &[0, 0, 0, 3]),
                            (
                                "Outcome2",
                                Some(&[0, 0, 0, 1]),
                                &[0, 0, 0, 0, 0, 0, 300, 301],
                            ),
                        ]),
                    );
                    expected.insert(
                        JoinType::Left,
                        expected_set_info_with_mask(vec![
                            (NULL_HEADER, None, &[1, 1, 1, 1]),
                            ("Income1", Some(&[1, 1, 1, 1]), &[40, 50, 20, 30]),
                            (
                                "Outcome1",
                                Some(&[1, 1, 1, 0]),
                                &[400, 401, 500, 501, 200, 201, 0, 0],
                            ),
                            ("ID", Some(&[1, 0, 1, 1]), &[4, 0, 2, 3]),
                            (
                                "Outcome2",
                                Some(&[0, 0, 0, 1]),
                                &[0, 0, 0, 0, 0, 0, 300, 301],
                            ),
                        ]),
                    );
                    expected.insert(
                        JoinType::Union,
                        expected_set_info_with_mask(vec![
                            (NULL_HEADER, None, &[1, 1, 1, 0, 1, 1, 1, 1]),
                            (
                                "Income1",
                                Some(&[1, 1, 1, 0, 1, 1, 1, 1]),
                                &[40, 50, 20, 0, 50, 30, 60, 70],
                            ),
                            (
                                "Outcome1",
                                Some(&[1, 1, 1, 0, 0, 0, 0, 0]),
                                &[400, 401, 500, 501, 200, 201, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            ),
                            (
                                "ID",
                                Some(&[1, 0, 1, 0, 1, 1, 1, 1]),
                                &[4, 0, 2, 0, 5, 3, 6, 7],
                            ),
                            (
                                "Outcome2",
                                Some(&[0, 0, 0, 0, 1, 1, 1, 1]),
                                &[
                                    0, 0, 0, 0, 0, 0, 0, 0, 500, 501, 300, 301, 600, 601, 700, 701,
                                ],
                            ),
                        ]),
                    );
                    expected.insert(
                        JoinType::Full,
                        expected_set_info_with_mask(vec![
                            (NULL_HEADER, None, &[1, 1, 1, 0, 1, 1, 1, 1]),
                            (
                                "Income1",
                                Some(&[1, 1, 1, 0, 1, 1, 1, 1]),
                                &[40, 50, 20, 0, 50, 30, 60, 70],
                            ),
                            (
                                "Outcome1",
                                Some(&[1, 1, 1, 0, 0, 0, 0, 0]),
                                &[400, 401, 500, 501, 200, 201, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            ),
                            (
                                "ID",
                                Some(&[1, 0, 1, 0, 1, 1, 1, 1]),
                                &[4, 0, 2, 0, 5, 3, 6, 7],
                            ),
                            (
                                "Outcome2",
                                Some(&[0, 0, 0, 0, 1, 1, 1, 1]),
                                &[
                                    0, 0, 0, 0, 0, 0, 0, 0, 500, 501, 300, 301, 600, 601, 700, 701,
                                ],
                            ),
                        ]),
                    );
                    expected
                },
            ),
            // Setup 2
            join_info(
                vec![
                    column_info_with_mask(NULL_HEADER, &[4], BIT, None, &[1, 1, 1, 1])?,
                    column_info_with_mask(
                        "Income1",
                        &[4],
                        UINT8,
                        Some(&[1, 1, 1, 1]),
                        &[40, 50, 20, 30],
                    )?,
                    column_info_with_mask(
                        "Outcome1",
                        &[4, 2],
                        UINT16,
                        Some(&[1, 1, 1, 1]),
                        &[400, 401, 500, 501, 200, 201, 300, 301],
                    )?,
                    column_info_with_mask("ID", &[4], UINT8, Some(&[1, 1, 1, 1]), &[4, 5, 2, 3])?,
                ],
                vec![
                    column_info_with_mask(NULL_HEADER, &[4], BIT, None, &[1, 1, 1, 1])?,
                    column_info_with_mask("ID", &[4], UINT8, Some(&[1, 1, 1, 1]), &[5, 3, 6, 7])?,
                    column_info_with_mask(
                        "Income2",
                        &[4],
                        UINT8,
                        Some(&[1, 1, 1, 1]),
                        &[50, 30, 60, 70],
                    )?,
                    column_info_with_mask(
                        "Outcome2",
                        &[4, 2],
                        UINT16,
                        Some(&[1, 0, 1, 1]),
                        &[500, 501, 300, 301, 600, 601, 700, 701],
                    )?,
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                {
                    let mut expected = HashMap::new();
                    expected.insert(
                        JoinType::Inner,
                        expected_set_info_with_mask(vec![
                            (NULL_HEADER, None, &[0, 1, 0, 1]),
                            ("Income1", Some(&[0, 1, 0, 1]), &[0, 50, 0, 30]),
                            (
                                "Outcome1",
                                Some(&[0, 1, 0, 1]),
                                &[0, 0, 500, 501, 0, 0, 300, 301],
                            ),
                            ("ID", Some(&[0, 1, 0, 1]), &[0, 5, 0, 3]),
                            (
                                "Outcome2",
                                Some(&[0, 1, 0, 0]),
                                &[0, 0, 500, 501, 0, 0, 0, 0],
                            ),
                        ]),
                    );
                    expected.insert(
                        JoinType::Left,
                        expected_set_info_with_mask(vec![
                            (NULL_HEADER, None, &[1, 1, 1, 1]),
                            ("Income1", Some(&[1, 1, 1, 1]), &[40, 50, 20, 30]),
                            (
                                "Outcome1",
                                Some(&[1, 1, 1, 1]),
                                &[400, 401, 500, 501, 200, 201, 300, 301],
                            ),
                            ("ID", Some(&[1, 1, 1, 1]), &[4, 5, 2, 3]),
                            (
                                "Outcome2",
                                Some(&[0, 1, 0, 0]),
                                &[0, 0, 500, 501, 0, 0, 0, 0],
                            ),
                        ]),
                    );
                    expected.insert(
                        JoinType::Union,
                        expected_set_info_with_mask(vec![
                            (NULL_HEADER, None, &[1, 0, 1, 0, 1, 1, 1, 1]),
                            (
                                "Income1",
                                Some(&[1, 0, 1, 0, 1, 1, 1, 1]),
                                &[40, 0, 20, 0, 50, 30, 60, 70],
                            ),
                            (
                                "Outcome1",
                                Some(&[1, 0, 1, 0, 0, 0, 0, 0]),
                                &[400, 401, 0, 0, 200, 201, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            ),
                            (
                                "ID",
                                Some(&[1, 0, 1, 0, 1, 1, 1, 1]),
                                &[4, 0, 2, 0, 5, 3, 6, 7],
                            ),
                            (
                                "Outcome2",
                                Some(&[0, 0, 0, 0, 1, 0, 1, 1]),
                                &[0, 0, 0, 0, 0, 0, 0, 0, 500, 501, 0, 0, 600, 601, 700, 701],
                            ),
                        ]),
                    );
                    expected.insert(
                        JoinType::Full,
                        expected_set_info_with_mask(vec![
                            (NULL_HEADER, None, &[1, 0, 1, 0, 1, 1, 1, 1]),
                            (
                                "Income1",
                                Some(&[1, 0, 1, 0, 1, 1, 1, 1]),
                                &[40, 0, 20, 0, 50, 30, 60, 70],
                            ),
                            (
                                "Outcome1",
                                Some(&[1, 0, 1, 0, 1, 1, 0, 0]),
                                &[
                                    400, 401, 0, 0, 200, 201, 0, 0, 500, 501, 300, 301, 0, 0, 0, 0,
                                ],
                            ),
                            (
                                "ID",
                                Some(&[1, 0, 1, 0, 1, 1, 1, 1]),
                                &[4, 0, 2, 0, 5, 3, 6, 7],
                            ),
                            (
                                "Outcome2",
                                Some(&[0, 0, 0, 0, 1, 0, 1, 1]),
                                &[0, 0, 0, 0, 0, 0, 0, 0, 500, 501, 0, 0, 600, 601, 700, 701],
                            ),
                        ]),
                    );
                    expected
                },
            ),
            // Setup 3
            join_info(
                vec![
                    column_info_with_mask(NULL_HEADER, &[4], BIT, None, &[1, 1, 1, 0])?,
                    column_info_with_mask(
                        "Income1",
                        &[4],
                        UINT8,
                        Some(&[1, 1, 1, 1]),
                        &[40, 50, 20, 30],
                    )?,
                    column_info_with_mask(
                        "Outcome1",
                        &[4, 2],
                        UINT16,
                        Some(&[1, 1, 1, 0]),
                        &[400, 401, 500, 501, 200, 201, 300, 301],
                    )?,
                    column_info_with_mask("ID", &[4], UINT8, Some(&[1, 1, 1, 1]), &[4, 5, 2, 3])?,
                ],
                vec![
                    column_info_with_mask(NULL_HEADER, &[4], BIT, None, &[1, 1, 1, 1])?,
                    column_info_with_mask("ID", &[4], UINT8, Some(&[1, 1, 1, 1]), &[5, 3, 6, 7])?,
                    column_info_with_mask(
                        "Income2",
                        &[4],
                        UINT8,
                        Some(&[1, 1, 1, 1]),
                        &[50, 30, 60, 70],
                    )?,
                    column_info_with_mask(
                        "Outcome2",
                        &[4, 2],
                        UINT16,
                        Some(&[1, 1, 1, 1]),
                        &[500, 501, 300, 301, 600, 601, 700, 701],
                    )?,
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                {
                    let mut expected = HashMap::new();
                    expected.insert(
                        JoinType::Inner,
                        expected_set_info_with_mask(vec![
                            (NULL_HEADER, None, &[0, 1, 0, 0]),
                            ("Income1", Some(&[0, 1, 0, 0]), &[0, 50, 0, 0]),
                            (
                                "Outcome1",
                                Some(&[0, 1, 0, 0]),
                                &[0, 0, 500, 501, 0, 0, 0, 0],
                            ),
                            ("ID", Some(&[0, 1, 0, 0]), &[0, 5, 0, 0]),
                            (
                                "Outcome2",
                                Some(&[0, 1, 0, 0]),
                                &[0, 0, 500, 501, 0, 0, 0, 0],
                            ),
                        ]),
                    );
                    expected.insert(
                        JoinType::Left,
                        expected_set_info_with_mask(vec![
                            (NULL_HEADER, None, &[1, 1, 1, 0]),
                            ("Income1", Some(&[1, 1, 1, 0]), &[40, 50, 20, 0]),
                            (
                                "Outcome1",
                                Some(&[1, 1, 1, 0]),
                                &[400, 401, 500, 501, 200, 201, 0, 0],
                            ),
                            ("ID", Some(&[1, 1, 1, 0]), &[4, 5, 2, 0]),
                            (
                                "Outcome2",
                                Some(&[0, 1, 0, 0]),
                                &[0, 0, 500, 501, 0, 0, 0, 0],
                            ),
                        ]),
                    );
                    expected.insert(
                        JoinType::Union,
                        expected_set_info_with_mask(vec![
                            (NULL_HEADER, None, &[1, 0, 1, 0, 1, 1, 1, 1]),
                            (
                                "Income1",
                                Some(&[1, 0, 1, 0, 1, 1, 1, 1]),
                                &[40, 0, 20, 0, 50, 30, 60, 70],
                            ),
                            (
                                "Outcome1",
                                Some(&[1, 0, 1, 0, 0, 0, 0, 0]),
                                &[400, 401, 0, 0, 200, 201, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            ),
                            (
                                "ID",
                                Some(&[1, 0, 1, 0, 1, 1, 1, 1]),
                                &[4, 0, 2, 0, 5, 3, 6, 7],
                            ),
                            (
                                "Outcome2",
                                Some(&[0, 0, 0, 0, 1, 1, 1, 1]),
                                &[
                                    0, 0, 0, 0, 0, 0, 0, 0, 500, 501, 300, 301, 600, 601, 700, 701,
                                ],
                            ),
                        ]),
                    );
                    expected.insert(
                        JoinType::Full,
                        expected_set_info_with_mask(vec![
                            (NULL_HEADER, None, &[1, 0, 1, 0, 1, 1, 1, 1]),
                            (
                                "Income1",
                                Some(&[1, 0, 1, 0, 1, 1, 1, 1]),
                                &[40, 0, 20, 0, 50, 30, 60, 70],
                            ),
                            (
                                "Outcome1",
                                Some(&[1, 0, 1, 0, 1, 0, 0, 0]),
                                &[400, 401, 0, 0, 200, 201, 0, 0, 500, 501, 0, 0, 0, 0, 0, 0],
                            ),
                            (
                                "ID",
                                Some(&[1, 0, 1, 0, 1, 1, 1, 1]),
                                &[4, 0, 2, 0, 5, 3, 6, 7],
                            ),
                            (
                                "Outcome2",
                                Some(&[0, 0, 0, 0, 1, 1, 1, 1]),
                                &[
                                    0, 0, 0, 0, 0, 0, 0, 0, 500, 501, 300, 301, 600, 601, 700, 701,
                                ],
                            ),
                        ]),
                    );
                    expected
                },
            ),
        ];

        if with_masks {
            Ok(tests_with_masks[index].clone())
        } else {
            Ok(tests_without_masks[index].clone())
        }
    }

    fn semi_private_join_helper(join_t: JoinType, has_column_masks: bool) -> Result<()> {
        for is_x_private in [false, true] {
            for is_y_private in [false, true] {
                if !(is_x_private && is_y_private) {
                    deterministic_join_helper(
                        join_t,
                        get_join_test_setup(has_column_masks, 2)?,
                        is_x_private,
                        is_y_private,
                    )?;
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_private_psi_0() -> Result<()> {
        private_join_data_helper(JoinType::Inner, get_join_test_setup(false, 0)?)
    }

    #[test]
    fn test_private_psi_1() -> Result<()> {
        private_join_data_helper(JoinType::Inner, get_join_test_setup(false, 1)?)
    }

    #[test]
    fn test_private_psi_2() -> Result<()> {
        private_join_data_helper(JoinType::Inner, get_join_test_setup(false, 2)?)
    }

    #[test]
    fn test_private_psi_3() -> Result<()> {
        private_join_data_helper(JoinType::Inner, get_join_test_setup(false, 3)?)
    }

    #[test]
    fn test_private_psi_4() -> Result<()> {
        private_join_data_helper(JoinType::Inner, get_join_test_setup(false, 4)?)
    }

    #[test]
    fn test_private_psi_5() -> Result<()> {
        private_join_data_helper(JoinType::Inner, get_join_test_setup(false, 5)?)
    }

    #[test]
    fn test_private_psi_6() -> Result<()> {
        private_join_data_helper(JoinType::Inner, get_join_test_setup(false, 6)?)
    }

    #[test]
    fn test_private_psi_7() -> Result<()> {
        private_join_data_helper(JoinType::Inner, get_join_test_setup(false, 7)?)
    }

    #[test]
    fn test_private_psi_8() -> Result<()> {
        private_join_data_helper(JoinType::Inner, get_join_test_setup(false, 8)?)
    }

    #[test]
    fn test_private_psi_9() -> Result<()> {
        private_join_data_helper(JoinType::Inner, get_join_test_setup(false, 9)?)
    }

    #[test]
    fn test_private_psi_0_with_masks() -> Result<()> {
        private_join_data_helper(JoinType::Inner, get_join_test_setup(true, 0)?)
    }

    #[test]
    fn test_private_psi_1_with_masks() -> Result<()> {
        private_join_data_helper(JoinType::Inner, get_join_test_setup(true, 1)?)
    }

    #[test]
    fn test_private_psi_2_with_masks() -> Result<()> {
        private_join_data_helper(JoinType::Inner, get_join_test_setup(true, 2)?)
    }

    #[test]
    fn test_private_psi_3_with_masks() -> Result<()> {
        private_join_data_helper(JoinType::Inner, get_join_test_setup(true, 3)?)
    }

    #[test]
    fn test_semi_private_psi() -> Result<()> {
        semi_private_join_helper(JoinType::Inner, false)
    }

    #[test]
    fn test_semi_private_psi_with_masks() -> Result<()> {
        semi_private_join_helper(JoinType::Inner, true)
    }

    #[test]
    fn test_private_left_join_0() -> Result<()> {
        private_join_data_helper(JoinType::Left, get_join_test_setup(false, 0)?)
    }

    #[test]
    fn test_private_left_join_1() -> Result<()> {
        private_join_data_helper(JoinType::Left, get_join_test_setup(false, 1)?)
    }

    #[test]
    fn test_private_left_join_2() -> Result<()> {
        private_join_data_helper(JoinType::Left, get_join_test_setup(false, 2)?)
    }

    #[test]
    fn test_private_left_join_3() -> Result<()> {
        private_join_data_helper(JoinType::Left, get_join_test_setup(false, 3)?)
    }

    #[test]
    fn test_private_left_join_4() -> Result<()> {
        private_join_data_helper(JoinType::Left, get_join_test_setup(false, 4)?)
    }

    #[test]
    fn test_private_left_join_5() -> Result<()> {
        private_join_data_helper(JoinType::Left, get_join_test_setup(false, 5)?)
    }

    #[test]
    fn test_private_left_join_6() -> Result<()> {
        private_join_data_helper(JoinType::Left, get_join_test_setup(false, 6)?)
    }

    #[test]
    fn test_private_left_join_7() -> Result<()> {
        private_join_data_helper(JoinType::Left, get_join_test_setup(false, 7)?)
    }

    #[test]
    fn test_private_left_join_8() -> Result<()> {
        private_join_data_helper(JoinType::Left, get_join_test_setup(false, 8)?)
    }

    #[test]
    fn test_private_left_join_9() -> Result<()> {
        private_join_data_helper(JoinType::Left, get_join_test_setup(false, 9)?)
    }

    #[test]
    fn test_private_left_join_0_with_masks() -> Result<()> {
        private_join_data_helper(JoinType::Left, get_join_test_setup(true, 0)?)
    }

    #[test]
    fn test_private_left_join_1_with_masks() -> Result<()> {
        private_join_data_helper(JoinType::Left, get_join_test_setup(true, 1)?)
    }

    #[test]
    fn test_private_left_join_2_with_masks() -> Result<()> {
        private_join_data_helper(JoinType::Left, get_join_test_setup(true, 2)?)
    }

    #[test]
    fn test_private_left_join_3_with_masks() -> Result<()> {
        private_join_data_helper(JoinType::Left, get_join_test_setup(true, 3)?)
    }

    #[test]
    fn test_semi_private_left_join() -> Result<()> {
        semi_private_join_helper(JoinType::Left, false)
    }

    #[test]
    fn test_semi_private_left_join_with_masks() -> Result<()> {
        semi_private_join_helper(JoinType::Left, true)
    }

    #[test]
    fn test_private_union_join_0() -> Result<()> {
        private_join_data_helper(JoinType::Union, get_join_test_setup(false, 0)?)
    }

    #[test]
    fn test_private_union_join_1() -> Result<()> {
        private_join_data_helper(JoinType::Union, get_join_test_setup(false, 1)?)
    }

    #[test]
    fn test_private_union_join_2() -> Result<()> {
        private_join_data_helper(JoinType::Union, get_join_test_setup(false, 2)?)
    }

    #[test]
    fn test_private_union_join_3() -> Result<()> {
        private_join_data_helper(JoinType::Union, get_join_test_setup(false, 3)?)
    }

    #[test]
    fn test_private_union_join_4() -> Result<()> {
        private_join_data_helper(JoinType::Union, get_join_test_setup(false, 4)?)
    }

    #[test]
    fn test_private_union_join_5() -> Result<()> {
        private_join_data_helper(JoinType::Union, get_join_test_setup(false, 5)?)
    }

    #[test]
    fn test_private_union_join_6() -> Result<()> {
        private_join_data_helper(JoinType::Union, get_join_test_setup(false, 6)?)
    }

    #[test]
    fn test_private_union_join_7() -> Result<()> {
        private_join_data_helper(JoinType::Union, get_join_test_setup(false, 7)?)
    }

    #[test]
    fn test_private_union_join_8() -> Result<()> {
        private_join_data_helper(JoinType::Union, get_join_test_setup(false, 8)?)
    }

    #[test]
    fn test_private_union_join_9() -> Result<()> {
        private_join_data_helper(JoinType::Union, get_join_test_setup(false, 9)?)
    }

    #[test]
    fn test_semi_private_union_join() -> Result<()> {
        semi_private_join_helper(JoinType::Union, false)
    }

    #[test]
    fn test_semi_private_union_join_with_masks() -> Result<()> {
        semi_private_join_helper(JoinType::Union, true)
    }

    #[test]
    fn test_private_union_join_0_with_masks() -> Result<()> {
        private_join_data_helper(JoinType::Union, get_join_test_setup(true, 0)?)
    }

    #[test]
    fn test_private_union_join_1_with_masks() -> Result<()> {
        private_join_data_helper(JoinType::Union, get_join_test_setup(true, 1)?)
    }

    #[test]
    fn test_private_union_join_2_with_masks() -> Result<()> {
        private_join_data_helper(JoinType::Union, get_join_test_setup(true, 2)?)
    }

    #[test]
    fn test_private_union_join_3_with_masks() -> Result<()> {
        private_join_data_helper(JoinType::Union, get_join_test_setup(true, 3)?)
    }

    #[test]
    fn test_private_full_join_0() -> Result<()> {
        private_join_data_helper(JoinType::Full, get_join_test_setup(false, 0)?)
    }

    #[test]
    fn test_private_full_join_1() -> Result<()> {
        private_join_data_helper(JoinType::Full, get_join_test_setup(false, 1)?)
    }

    #[test]
    fn test_private_full_join_2() -> Result<()> {
        private_join_data_helper(JoinType::Full, get_join_test_setup(false, 2)?)
    }

    #[test]
    fn test_private_full_join_3() -> Result<()> {
        private_join_data_helper(JoinType::Full, get_join_test_setup(false, 3)?)
    }

    #[test]
    fn test_private_full_join_4() -> Result<()> {
        private_join_data_helper(JoinType::Full, get_join_test_setup(false, 4)?)
    }

    #[test]
    fn test_private_full_join_5() -> Result<()> {
        private_join_data_helper(JoinType::Full, get_join_test_setup(false, 5)?)
    }

    #[test]
    fn test_private_full_join_6() -> Result<()> {
        private_join_data_helper(JoinType::Full, get_join_test_setup(false, 6)?)
    }

    #[test]
    fn test_private_full_join_7() -> Result<()> {
        private_join_data_helper(JoinType::Full, get_join_test_setup(false, 7)?)
    }

    #[test]
    fn test_private_full_join_8() -> Result<()> {
        private_join_data_helper(JoinType::Full, get_join_test_setup(false, 8)?)
    }

    #[test]
    fn test_private_full_join_9() -> Result<()> {
        private_join_data_helper(JoinType::Full, get_join_test_setup(false, 9)?)
    }

    #[test]
    fn test_private_full_join_0_with_masks() -> Result<()> {
        private_join_data_helper(JoinType::Full, get_join_test_setup(true, 0)?)
    }

    #[test]
    fn test_private_full_join_1_with_masks() -> Result<()> {
        private_join_data_helper(JoinType::Full, get_join_test_setup(true, 1)?)
    }

    #[test]
    fn test_private_full_join_2_with_masks() -> Result<()> {
        private_join_data_helper(JoinType::Full, get_join_test_setup(true, 2)?)
    }

    #[test]
    fn test_private_full_join_3_with_masks() -> Result<()> {
        private_join_data_helper(JoinType::Full, get_join_test_setup(true, 3)?)
    }

    #[test]
    fn test_semi_private_full_join() -> Result<()> {
        semi_private_join_helper(JoinType::Full, false)
    }

    #[test]
    fn test_semi_private_full_join_with_masks() -> Result<()> {
        semi_private_join_helper(JoinType::Full, true)
    }
}
