use crate::broadcast::number_to_index;
use crate::custom_ops::{CustomOperation, CustomOperationBody};
use crate::data_types::{
    array_type, scalar_type, tuple_type, ArrayShape, Type, BIT, INT64, UINT64,
};
use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::Context;
use crate::graphs::{Graph, GraphAnnotation, Node};
use crate::ops::comparisons::{Equal, GreaterThan};
use crate::ops::multiplexer::Mux;
use crate::ops::newton_inversion::NewtonInversion;
use crate::ops::utils::{pull_out_bits, put_in_bits};
use crate::random::{PRNG, SEED_SIZE};

use serde::{Deserialize, Serialize};

pub struct KMeansTrainingParams {
    pub data_size: u64,
    pub dims: u64,
    pub k: u64,
    pub iterations: u64,
    pub restarts: u64,
    pub avoid_divisions: bool,
    pub seed: [u8; SEED_SIZE],
}

/// Runs (approximate) KMeans clustering on the given data (INT64 scalar type is required).
/// Parameters:
/// -- data_size: number of points in the training set;
/// -- dims: number of dimensions;
/// -- k: number of resulting clusters;
/// -- iterations: number of kmeans iterations per kmeans run;
/// -- restarts: number of kmeans runs with different initialization.
pub fn fit_graph(context: Context, params: KMeansTrainingParams) -> Result<Graph> {
    let k = params.k;
    if k <= 1 {
        return Err(runtime_error!("At least 2 centers are needed"));
    }
    let data_size = params.data_size;
    if data_size < k {
        return Err(runtime_error!(
            "Number of centers must not be bigger than data size"
        ));
    }
    let mut newton_bits = 2;
    while (1 << newton_bits) <= data_size {
        newton_bits += 1;
    }
    let graph = context.create_graph()?;
    let dims = params.dims;
    let input_arr = graph.input(array_type(vec![data_size, dims], INT64))?;
    let arr2 = input_arr.multiply(input_arr.clone())?.sum(vec![1])?;
    let arr2_expanded = arr2.reshape(array_type(vec![data_size, 1], INT64))?;
    let bit_type = array_type(vec![k, dims], BIT);
    let infinite_scores = constant_of_type(
        graph.clone(),
        array_type(vec![data_size], INT64),
        i64::MAX as u64,
    )?;
    let full_mask = constant_of_type(graph.clone(), array_type(vec![k, data_size], INT64), 1)?;

    let get_distances = |centers: Node| -> Result<Node> {
        let centers2 = centers.multiply(centers.clone())?.sum(vec![1])?;
        let cosines = input_arr.matmul(centers.permute_axes(vec![1, 0])?)?;
        let centers2_expanded = centers2.reshape(array_type(vec![1, k], INT64))?;
        centers2_expanded
            .add(arr2_expanded.clone())?
            .subtract(cosines.add(cosines.clone())?)
    };

    let mut result_centers = input_arr.clone(); // Initialize with smth, so that Rust doesn't complain.
    let mut result_score = input_arr.clone();
    let mut prng = PRNG::new(Some(params.seed))?;
    for global_iteration in 0..params.restarts {
        // Choose initial center indices.
        let mut indices: Vec<u64> = (0..data_size).collect();
        for i in 0..data_size {
            let j = prng.get_random_value(scalar_type(UINT64))?.to_u64(UINT64)? % data_size;
            if i != j {
                indices.swap(i as usize, j as usize);
            }
        }
        let input_vec = input_arr.array_to_vector()?;
        let mut initial_centers = vec![];
        for i in 0..k {
            initial_centers.push(input_vec.vector_get(graph.constant(
                scalar_type(UINT64),
                Value::from_scalar(indices[i as usize], UINT64)?,
            )?)?);
        }
        let mut centers = graph
            .create_vector(initial_centers[0].get_type()?, initial_centers)?
            .vector_to_array()?;
        for _ in 0..params.iterations {
            // First, we compute cluster assignments.
            let distances2 = get_distances(centers)?;
            let cluster_ids = graph.custom_op(CustomOperation::new(Argmin {}), vec![distances2])?;
            let cluster_ids_onehot =
                graph.custom_op(CustomOperation::new(Onehot { depth: k }), vec![cluster_ids])?;

            // Let's compute cluster sizes and sums of points in clusters.
            let cluster_sizes = cluster_ids_onehot.sum(vec![0])?; // [k]
            let cluster2point = cluster_ids_onehot.permute_axes(vec![1, 0])?;
            let cluster_sums = cluster2point.matmul(input_arr.clone())?; // [k, dims]

            if params.avoid_divisions {
                // In the regular KMeans, we would just divide cluster_sums by cluster_sizes.
                // But since we don't have division, we find the median instead: the point across all
                // points in the cluster which minimizes sum of distances to other points.
                // I.e. the point c which minimizes \sum_{p \in cluster} (c - p) ** 2.

                // We compute all 3 terms needed: cluster_size * (c ** 2), \sum p ** 2, \sum c * p.
                let cluster_sizes_propagated = cluster_ids_onehot.matmul(cluster_sizes.clone())?;
                // This one is cluster_size * (c ** 2).
                let self_contribution = arr2.multiply(cluster_sizes_propagated)?;
                // This one is \sum c * p.
                let cosines_with_sum = input_arr
                    .matmul(cluster_sums.permute_axes(vec![1, 0])?)?
                    .sum(vec![1])?;
                // This one is \sum p ** 2.
                let cluster_sums2 = cluster2point.matmul(arr2_expanded.clone())?;
                // Finally, we compute \sum_{p \in cluster} (c - p) ** 2.
                let point_scores = self_contribution
                    .add(cluster_sums2)?
                    .subtract(cosines_with_sum.add(cosines_with_sum.clone())?)?;
                // Now we want to choose point with the smallest score per cluster.
                let cluster_point_score = cluster2point.multiply(point_scores)?.add(
                    full_mask
                        .subtract(cluster2point)?
                        .multiply(infinite_scores.clone())?,
                )?;
                let new_center_inds =
                    graph.custom_op(CustomOperation::new(Argmin {}), vec![cluster_point_score])?;
                // Finally, we need to do gather(input_arr, new_center_inds), and we're doing it through onehot, which is slooow.
                let new_center_inds_onehot = graph.custom_op(
                    CustomOperation::new(Onehot { depth: data_size }),
                    vec![new_center_inds],
                )?;
                centers = new_center_inds_onehot.matmul(input_arr.clone())?;
            } else {
                let inverse_sizes = graph
                    .custom_op(
                        CustomOperation::new(NewtonInversion {
                            iterations: 5,
                            denominator_cap_2k: newton_bits,
                        }),
                        vec![cluster_sizes.a2b()?.b2a(UINT64)?],
                    )?
                    .b2a(INT64)?;
                let cluster_means_2k =
                    cluster_sums.multiply(inverse_sizes.reshape(array_type(vec![k, 1], INT64))?)?;
                // TODO: this can be changed to Truncate once MPC supports it.
                let cluster_means_2k_binary = cluster_means_2k.a2b()?;
                let cluster_means_2k_bits =
                    pull_out_bits(cluster_means_2k_binary)?.array_to_vector()?;
                let mut shifted_bits = vec![];
                // We treat negative numbers as follows: we're still performing right shift, but we're using
                // the last bit (one in case of negative numbers) for filling missing bits.
                // This might be different from some understanding of division (e.g. 57 / 4 = 14, while -57 / 4 becomes -15),
                // but is good enough for our case.
                for i in 0..64 - newton_bits {
                    let index = graph.constant(
                        scalar_type(UINT64),
                        Value::from_scalar(i + newton_bits, UINT64)?,
                    )?;
                    shifted_bits.push(cluster_means_2k_bits.vector_get(index)?);
                }
                let last_index =
                    graph.constant(scalar_type(UINT64), Value::from_scalar(63, UINT64)?)?;
                let last_bit = cluster_means_2k_bits.vector_get(last_index)?;
                for _ in 0..newton_bits {
                    shifted_bits.push(last_bit.clone());
                }
                centers = put_in_bits(
                    graph
                        .create_vector(bit_type.clone(), shifted_bits)?
                        .vector_to_array()?,
                )?
                .b2a(INT64)?;
            }
        }
        let dists = get_distances(centers.clone())?;
        let cluster_assignments =
            graph.custom_op(CustomOperation::new(Argmin {}), vec![dists.clone()])?;
        let cluster_assignments_onehot = graph.custom_op(
            CustomOperation::new(Onehot { depth: k }),
            vec![cluster_assignments],
        )?;
        // We pick the best clustering based on the following metric:
        // \sum_{cluster} \sum_{p \in cluster} L2Distance(p, center(cluster)) ** 2.
        let score = dists
            .multiply(cluster_assignments_onehot)?
            .sum(vec![0, 1])?
            .a2b()?;
        if global_iteration == 0 {
            result_centers = centers.a2b()?;
            result_score = score;
        } else {
            let comparison_result = graph.custom_op(
                CustomOperation::new(GreaterThan {
                    signed_comparison: true,
                }),
                vec![result_score.clone(), score.clone()],
            )?;
            result_centers = graph.custom_op(
                CustomOperation::new(Mux {}),
                vec![
                    comparison_result.clone(),
                    centers.a2b()?,
                    result_centers.clone(),
                ],
            )?;
            result_score = graph.custom_op(
                CustomOperation::new(Mux {}),
                vec![comparison_result.clone(), score, result_score.clone()],
            )?;
        }
    }
    result_centers.b2a(INT64)?.set_as_output()?;
    graph.finalize()?;
    Ok(graph)
}

pub fn inference_graph(context: Context, k: u64, dims: u64) -> Result<Graph> {
    // self.centers: [centers, dims]
    // point: [dims]
    let graph = context.create_graph()?;
    let centers = graph.input(array_type(vec![k, dims], INT64))?;
    let point = graph.input(array_type(vec![dims], INT64))?;
    let cosines = centers.matmul(point.clone())?;
    let centers2 = centers.multiply(centers.clone())?.sum(vec![1])?;
    let point2 = point.multiply(point.clone())?.sum(vec![0])?;
    let distances = centers2
        .add(point2)?
        .subtract(cosines.add(cosines.clone())?)?;
    let result = graph.custom_op(CustomOperation::new(Argmin {}), vec![distances])?;
    result.set_as_output()?;
    graph.finalize()?;
    Ok(graph)
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
struct Argmin {}

/// Given an array, returns indices of minimum values over the last dimension.
/// Only accepts int64's. Returns indices in the binary format.
#[typetag::serde]
impl CustomOperationBody for Argmin {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 1 {
            return Err(runtime_error!("Invalid number of arguments for Argmin"));
        }
        let input_type = arguments_types[0].clone();
        if !input_type.is_array() || input_type.get_scalar_type() != INT64 {
            return Err(runtime_error!(
                "Argmin can only be applied to arrays of int64"
            ));
        }
        let input_shape = input_type.get_shape();
        let mut element_shape = input_shape[..input_shape.len() - 1].to_vec();
        element_shape.push(64);
        let element_type = array_type(element_shape.clone(), BIT);

        let argmin_graph = context.create_graph()?;
        let value_and_index_type = tuple_type(vec![element_type.clone(), element_type]);
        let input_state = argmin_graph.input(value_and_index_type.clone())?;
        let input_element_index = argmin_graph.input(value_and_index_type)?;
        let current_min = input_state.tuple_get(0)?;
        let current_min_index = input_state.tuple_get(1)?;
        let input_element = input_element_index.tuple_get(0)?;
        let input_index = input_element_index.tuple_get(1)?;
        let comparison_result = argmin_graph.custom_op(
            CustomOperation::new(GreaterThan {
                signed_comparison: true,
            }),
            vec![current_min.clone(), input_element.clone()],
        )?;
        let mut expanded_shape = element_shape[0..element_shape.len() - 1].to_vec();
        expanded_shape.push(1);
        let comparison_result_expanded =
            comparison_result.reshape(array_type(expanded_shape, BIT))?;
        let output_index = argmin_graph.custom_op(
            CustomOperation::new(Mux {}),
            vec![
                comparison_result_expanded.clone(),
                input_index,
                current_min_index,
            ],
        )?;
        let output_element = argmin_graph.custom_op(
            CustomOperation::new(Mux {}),
            vec![comparison_result_expanded, input_element, current_min],
        )?;
        let output = argmin_graph.create_tuple(vec![output_element, output_index])?;
        argmin_graph
            .create_tuple(vec![output, argmin_graph.create_tuple(vec![])?])?
            .set_as_output()?;
        argmin_graph.add_annotation(GraphAnnotation::AssociativeOperation)?;
        argmin_graph.finalize()?;

        let graph = context.create_graph()?;
        let input_arr = graph.input(input_type)?;
        let arr = pull_out_bits(input_arr)?.a2b()?;
        let enumerated_data = enumerate(arr)?;
        let initial_value = enumerated_data
            .vector_get(graph.constant(scalar_type(UINT64), Value::from_scalar(0, UINT64)?)?)?;
        let final_state = graph.iterate(argmin_graph, initial_value, enumerated_data)?;
        let result_index = final_state.tuple_get(0)?.tuple_get(1)?;
        result_index.set_as_output()?;
        graph.finalize()?;
        Ok(graph)
    }

    fn get_name(&self) -> String {
        "Argmin".to_string()
    }
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
struct Onehot {
    depth: u64,
}

/// Given an array, one-hot encodes the last dimension (output type is int64, not bit, as one might think).
/// The indices in the array must be in the bnary format.
#[typetag::serde]
impl CustomOperationBody for Onehot {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 1 {
            return Err(runtime_error!("Invalid number of arguments for Onehot"));
        }
        let input_type = arguments_types[0].clone();
        if !input_type.is_array()
            || input_type.get_scalar_type() != BIT
            || input_type.get_shape()[input_type.get_shape().len() - 1] != 64
        {
            return Err(runtime_error!(
                "Onehot can only be applied to arrays of uint64's in binary format"
            ));
        }
        let input_shape = input_type.get_shape();

        let graph = context.create_graph()?;
        let input_arr = graph.input(input_type.clone())?;
        let zeros = graph.constant(input_type.clone(), Value::zero_of_type(input_type))?;
        let ones = binary_constant_of_shape(graph.clone(), input_shape, 1)?;

        let mut outputs = vec![];
        for i in 0..self.depth {
            let i_node = binary_constant_of_shape(graph.clone(), vec![64], i)?;
            let equality = graph.custom_op(
                CustomOperation::new(Equal {}),
                vec![input_arr.clone(), i_node],
            )?;
            let mut equality_shape_expanded = equality.get_type()?.get_shape();
            equality_shape_expanded.push(1);
            let equality_expanded = equality.reshape(array_type(equality_shape_expanded, BIT))?;
            outputs.push(graph.custom_op(
                CustomOperation::new(Mux {}),
                vec![equality_expanded, ones.clone(), zeros.clone()],
            )?);
        }
        let output_arr = graph
            .create_vector(outputs[0].get_type()?, outputs)?
            .vector_to_array()?
            .b2a(INT64)?;
        put_in_bits(output_arr)?.set_as_output()?;
        graph.finalize()?;
        Ok(graph)
    }

    fn get_name(&self) -> String {
        format!("Onehot({})", self.depth)
    }
}

fn enumerate(arr: Node) -> Result<Node> {
    let graph = arr.get_graph();
    let shape = arr.get_type()?.get_shape();
    let num_elements = shape[0];
    let mut indices_elements = vec![];
    let mut index_shape = shape[1..shape.len() - 1].to_vec();
    index_shape.push(64);
    for i in 0..num_elements {
        indices_elements.push(binary_constant_of_shape(
            graph.clone(),
            index_shape.clone(),
            i,
        )?);
    }
    let indices = graph.create_vector(indices_elements[0].get_type()?, indices_elements)?;
    graph.zip(vec![arr.array_to_vector()?, indices])
}

pub fn constant_of_type(g: Graph, t: Type, c: u64) -> Result<Node> {
    let zeros = g.constant(t.clone(), Value::zero_of_type(t.clone()))?;
    let scalar = g.constant(
        scalar_type(t.get_scalar_type()),
        Value::from_scalar(c, t.get_scalar_type())?,
    )?;
    // We rely on broadcasting to get a value of the right shape.
    zeros.add(scalar)
}

fn binary_constant_of_shape(g: Graph, shape: ArrayShape, c: u64) -> Result<Node> {
    let output_type = array_type(shape.clone(), BIT);
    let value = Value::zero_of_type(output_type.clone());
    let mut bytes = value.access_bytes(|ref_bytes| Ok(ref_bytes.to_vec()))?;
    for i in 0..shape.iter().product() {
        let index = number_to_index(i, &shape);
        let state_index = index[index.len() - 1];
        let bit = ((c >> state_index) & 1) as u8;
        let position = i / 8;
        let offset = i % 8;
        bytes[position as usize] &= !(1 << offset);
        bytes[position as usize] |= bit << offset;
    }
    let value = Value::from_bytes(bytes);
    g.constant(output_type, value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_ops::run_instantiation_pass;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;

    #[test]
    fn test_argmin() {
        let argmin_helper = |vals: Vec<i64>| -> Result<u64> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let arr = g.input(array_type(vec![5], INT64))?;
            let output = g.custom_op(CustomOperation::new(Argmin {}), vec![arr])?;
            output.b2a(UINT64)?.set_as_output()?;
            g.finalize()?;
            g.set_as_main()?;
            c.finalize()?;
            let context = run_instantiation_pass(c).unwrap().get_context();
            let result = random_evaluate(
                context.get_main_graph()?,
                vec![Value::from_flattened_array(&vals, INT64)?],
            )?;
            result.to_u64(UINT64)
        };
        assert_eq!(argmin_helper(vec![1, 2, 3, 4, 5]).unwrap(), 0);
        assert_eq!(argmin_helper(vec![5, 4, 3, 2, 1]).unwrap(), 4);
        assert_eq!(argmin_helper(vec![3, 2, 1, 5, 4]).unwrap(), 2);
        assert_eq!(argmin_helper(vec![3, 2, 2, 2, 4]).unwrap(), 1);
    }

    #[test]
    fn test_argmin_2d() {
        let argmin_helper = |vals: Vec<Vec<i64>>| -> Result<Vec<u64>> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let input_arr = g.input(array_type(vec![2 * 5], INT64))?;
            let arr = input_arr.reshape(array_type(vec![2, 5], INT64))?;
            let output = g.custom_op(CustomOperation::new(Argmin {}), vec![arr])?;
            output.b2a(UINT64)?.set_as_output()?;
            g.finalize()?;
            g.set_as_main()?;
            c.finalize()?;
            let context = run_instantiation_pass(c).unwrap().get_context();
            let mut flat_vals = vec![];
            for row in vals {
                flat_vals.extend(row);
            }
            let result = random_evaluate(
                context.get_main_graph()?,
                vec![Value::from_flattened_array(&flat_vals, INT64)?],
            )?;
            result.to_flattened_array_u64(array_type(vec![2], UINT64))
        };
        assert_eq!(
            argmin_helper(vec![vec![1, 2, 3, 4, 5], vec![5, 4, 3, 2, 1]]).unwrap(),
            vec![0, 4]
        );
        assert_eq!(
            argmin_helper(vec![vec![2, 1, 1, 2, 2], vec![4, 3, 1, 2, 5]]).unwrap(),
            vec![1, 2]
        );
    }

    #[test]
    fn test_onehot() {
        let onehot_helper = |data: Vec<u64>, shape: Vec<u64>| -> Result<Vec<u64>> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let input_arr = g.input(array_type(vec![data.len() as u64], UINT64))?;
            let arr = input_arr.reshape(array_type(shape, UINT64))?.a2b()?;
            let output = g.custom_op(CustomOperation::new(Onehot { depth: 3 }), vec![arr])?;
            output.set_as_output()?;
            g.finalize()?;
            g.set_as_main()?;
            c.finalize()?;
            let context = run_instantiation_pass(c).unwrap().get_context();
            let result = random_evaluate(
                context.get_main_graph()?,
                vec![Value::from_flattened_array(&data, INT64)?],
            )?;
            result.to_flattened_array_u64(array_type(vec![data.len() as u64 * 3], UINT64))
        };
        assert_eq!(
            onehot_helper(vec![1, 2, 0], vec![3]).unwrap(),
            vec![0, 1, 0, 0, 0, 1, 1, 0, 0],
        );
        assert_eq!(
            onehot_helper(vec![0, 1, 2, 0], vec![2, 2]).unwrap(),
            vec![1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
        );
    }

    #[test]
    fn test_kmeans() {
        let kmeans_helper = |training_data: Vec<Vec<i64>>,
                             test_data: Vec<Vec<i64>>,
                             avoid_divisions: bool|
         -> Result<Vec<u64>> {
            let c = create_context()?;
            let seed = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F";
            let n = training_data.len() as u64;
            let m = test_data.len() as u64;
            let dims = training_data[0].len() as u64;
            // Note: 5 restarts are needed, bad initial center choice is unrecoverable.
            let fit_graph = fit_graph(
                c.clone(),
                KMeansTrainingParams {
                    data_size: n,
                    k: 3,
                    dims,
                    iterations: 5,
                    restarts: 5,
                    seed: *seed,
                    avoid_divisions,
                },
            )?;
            let inference_graph = inference_graph(c.clone(), 3, dims)?;
            let g = c.create_graph()?;
            let training_input = g.input(array_type(vec![n * dims], INT64))?;
            let test_input = g.input(array_type(vec![m * dims], INT64))?;
            let training = training_input.reshape(array_type(vec![n, dims], INT64))?;
            let test = test_input.reshape(array_type(vec![m, dims], INT64))?;
            let centers = g.call(fit_graph, vec![training])?;
            let mut outputs = vec![];
            let test_vecs = test.array_to_vector()?;
            for i in 0..m {
                let vec = test_vecs
                    .vector_get(g.constant(scalar_type(UINT64), Value::from_scalar(i, UINT64)?)?)?;
                let result = g.call(inference_graph.clone(), vec![centers.clone(), vec.clone()])?;
                outputs.push(result.b2a(UINT64)?);
            }
            let output = g
                .create_vector(outputs[0].get_type()?, outputs)?
                .vector_to_array()?;

            output.set_as_output()?;
            g.finalize()?;
            g.set_as_main()?;
            c.finalize()?;

            let context = run_instantiation_pass(c).unwrap().get_context();
            let mut flat_training = vec![];
            for row in training_data {
                flat_training.extend(row);
            }
            let mut flat_test = vec![];
            for row in test_data.clone() {
                flat_test.extend(row);
            }
            let result = random_evaluate(
                context.get_main_graph()?,
                vec![
                    Value::from_flattened_array(&flat_training, INT64)?,
                    Value::from_flattened_array(&flat_test, INT64)?,
                ],
            )?;
            result.to_flattened_array_u64(array_type(vec![test_data.len() as u64], UINT64))
        };
        let training_data = vec![
            vec![0, 0],
            vec![0, 2],
            vec![-1, 0],
            vec![-2, 1],
            vec![-1, -1],
            vec![1, 1],
            vec![100, 0],
            vec![100, 2],
            vec![99, 0],
            vec![98, 1],
            vec![99, -1],
            vec![101, 1],
            vec![0, 100],
            vec![0, 102],
            vec![-1, 100],
            vec![-2, 101],
            vec![-1, 99],
            vec![1, 101],
        ];
        let test_data = vec![
            vec![2, 2],
            vec![-2, -2],
            vec![100, 2],
            vec![100, -2],
            vec![2, 100],
            vec![-2, 100],
        ];
        for avoid_division in vec![true, false] {
            let cluster_inds =
                kmeans_helper(training_data.clone(), test_data.clone(), avoid_division).unwrap();
            assert_eq!(cluster_inds[0], cluster_inds[1]);
            assert_eq!(cluster_inds[2], cluster_inds[3]);
            assert_eq!(cluster_inds[4], cluster_inds[5]);
            assert_ne!(cluster_inds[0], cluster_inds[2]);
            assert_ne!(cluster_inds[0], cluster_inds[4]);
            assert_ne!(cluster_inds[2], cluster_inds[4]);
        }
    }
}
