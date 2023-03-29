extern crate ndarray;

use crate::data_types::{array_type, named_tuple_type, tuple_type, vector_type, ScalarType, Type};
use crate::data_values;
use crate::data_values::Value;
use crate::errors::Result;
use crate::random::PRNG;
use crate::typed_value::{generalized_add, generalized_subtract, TypedValue};
use crate::typed_value_operations::{
    extend_helper, get_helper, get_sub_vector_helper, insert_helper, pop_helper, push_helper,
    remove_helper, FromVectorMode, ToNdarray, TypedValueArrayOperations, TypedValueOperations,
};
use crate::typed_value_secret_shared::TypedValueSecretShared;
#[derive(Clone, Debug)]
pub struct ReplicatedShares {
    shares: Vec<Value>,
    t: Type,
    name: Option<String>,
}
use ndarray::Axis;
impl TypedValueArrayOperations<ReplicatedShares> for ReplicatedShares {
    fn from_ndarray<ST: TryInto<u128> + std::ops::Not<Output = ST> + TryInto<u8> + Copy>(
        a: ndarray::ArrayD<ST>,
        st: ScalarType,
    ) -> Result<ReplicatedShares> {
        let mut shapes = vec![];
        let shares_result: Result<Vec<Value>> = a
            .axis_iter(Axis(0))
            .map(|x| -> Result<Value> {
                shapes.push(x.shape().to_vec());
                Value::from_ndarray(x.to_owned(), st)
            })
            .collect();
        let shares = shares_result?;
        let t = array_type(shapes[0].iter().map(|x| *x as u64).collect(), st);
        Ok(ReplicatedShares {
            shares,
            t,
            name: None,
        })
    }
}

impl<T> ToNdarray<T> for ReplicatedShares
where
    data_values::Value: data_values::ToNdarray<T>,
    T: Copy,
{
    fn to_ndarray(&self) -> Result<ndarray::ArrayD<T>> {
        let array0 = data_values::ToNdarray::<T>::to_ndarray(&self.shares[0], self.t.clone())?;
        let array1 = data_values::ToNdarray::<T>::to_ndarray(&self.shares[1], self.t.clone())?;
        let array2 = data_values::ToNdarray::<T>::to_ndarray(&self.shares[2], self.t.clone())?;
        Ok(ndarray::stack(
            Axis(0),
            &[array0.view(), array1.view(), array2.view()],
        )?)
    }
}
impl TypedValueOperations<ReplicatedShares> for ReplicatedShares {
    /// Returns Type of `self`
    ///
    /// # Returns
    ///
    /// Type of the ReplicatedShares
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::TypedValueOperations;
    /// # use ciphercore_base::typed_value_secret_shared::replicated_shares::ReplicatedShares;
    /// # use ciphercore_base::typed_value_secret_shared::TypedValueSecretShared;
    /// # use ciphercore_base::data_types::{scalar_type,INT32};
    /// # use ciphercore_base::random::PRNG;
    /// let mut prng = PRNG::new(None).unwrap();
    /// let tv = TypedValue::from_scalar(1, INT32).unwrap();
    /// let shared_tv = ReplicatedShares::secret_share_for_local_evaluation(tv, &mut prng).unwrap();
    /// assert!(shared_tv.get_type().eq(&scalar_type(INT32)));
    /// ```
    fn get_type(&self) -> Type {
        self.t.clone()
    }
    /// Checks equality of `self` with another ReplicatedShare.
    ///
    /// # Returns
    ///
    /// True if `self` == `other` false otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::TypedValueOperations;
    /// # use ciphercore_base::typed_value_secret_shared::replicated_shares::ReplicatedShares;
    /// # use ciphercore_base::typed_value_secret_shared::TypedValueSecretShared;
    /// # use ciphercore_base::data_types::{scalar_type,INT32};
    /// # use ciphercore_base::random::PRNG;
    /// let mut prng = PRNG::new(Some([0;16])).unwrap();
    /// let tv1 = TypedValue::from_scalar(1, INT32).unwrap();
    /// let tv1_shared = ReplicatedShares::secret_share_for_local_evaluation(tv1, &mut prng).unwrap();
    /// let mut prng = PRNG::new(Some([0;16])).unwrap();
    /// let tv2 = TypedValue::from_scalar(1, INT32).unwrap();
    /// let tv2_shared = ReplicatedShares::secret_share_for_local_evaluation(tv2, &mut prng).unwrap();
    /// assert!(tv1_shared.is_equal(&tv2_shared).unwrap());
    /// ```
    fn is_equal(&self, other: &Self) -> Result<bool> {
        let self0 = TypedValue::new(self.t.clone(), self.shares[0].clone())?;
        let self1 = TypedValue::new(self.t.clone(), self.shares[1].clone())?;
        let self2 = TypedValue::new(self.t.clone(), self.shares[2].clone())?;
        let other0 = TypedValue::new(other.t.clone(), other.shares[0].clone())?;
        let other1 = TypedValue::new(other.t.clone(), other.shares[1].clone())?;
        let other2 = TypedValue::new(other.t.clone(), other.shares[2].clone())?;
        Ok(self0.is_equal(&other0)? & self1.is_equal(&other1)? & self2.is_equal(&other2)?)
    }
    /// Converts `self` to a vector of ReplicatedShares or return an error if `self` has scalar or array type.
    ///
    /// # Returns
    ///
    /// Extracted vector of ReplicatedShares
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// # use ciphercore_base::typed_value_secret_shared::replicated_shares::ReplicatedShares;
    /// # use ciphercore_base::typed_value_secret_shared::TypedValueSecretShared;
    /// # use ciphercore_base::random::PRNG;
    /// # use ciphercore_base::data_types::{scalar_type,INT32};
    ///
    /// let mut prng = PRNG::new(None).unwrap();
    /// let v = vec![TypedValue::from_scalar(1,INT32).unwrap(),
    ///     TypedValue::from_scalar(2,INT32).unwrap(),
    ///     TypedValue::from_scalar(3,INT32).unwrap()];
    /// let tv1 = TypedValue::from_vector(v.clone(), FromVectorMode::Vector).unwrap();
    /// let tv1_shared = ReplicatedShares::secret_share_for_local_evaluation(tv1, &mut prng).unwrap();
    /// let v_shared = tv1_shared.to_vector().unwrap();
    /// assert!(v.iter().zip(v_shared.iter()).all(|(a,b)| a.is_equal(&b.reveal().unwrap()).unwrap()))
    /// ```
    fn to_vector(&self) -> Result<Vec<ReplicatedShares>> {
        let vec0 = self.shares[0].to_vector()?;
        let vec1 = self.shares[1].to_vector()?;
        let vec2 = self.shares[2].to_vector()?;
        match self.get_type() {
            Type::Tuple(ts) => {
                if ts.len() != vec0.len() {
                    return Err(runtime_error!("Inconsistent number of elements!"));
                }
                let mut res = vec![];
                for (i, t) in ts.iter().enumerate() {
                    res.push(ReplicatedShares {
                        shares: vec![vec0[i].clone(), vec1[i].clone(), vec2[i].clone()],
                        t: t.as_ref().clone(),
                        name: None,
                    });
                }
                Ok(res)
            }
            Type::Vector(n, t) => {
                if n != (vec0.len() as u64) {
                    return Err(runtime_error!("Inconsistent number of elements!"));
                }
                let mut res = vec![];
                for (i, _) in vec0.iter().enumerate() {
                    res.push(ReplicatedShares {
                        shares: vec![vec0[i].clone(), vec1[i].clone(), vec2[i].clone()],
                        t: t.as_ref().clone(),
                        name: None,
                    });
                }
                Ok(res)
            }
            Type::NamedTuple(n_ts) => {
                if n_ts.len() != vec0.len() {
                    return Err(runtime_error!("Inconsistent number of elements!"));
                }
                let mut res = vec![];
                for (i, t) in n_ts.iter().enumerate() {
                    res.push(ReplicatedShares {
                        shares: vec![vec0[i].clone(), vec1[i].clone(), vec2[i].clone()],
                        t: t.1.as_ref().clone(),
                        name: Some(t.0.clone()),
                    });
                }
                Ok(res)
            }
            _ => Err(runtime_error!("Not a vector!")),
        }
    }
    /// Constructs a ReplicatedShare in ciphercore vector or tuple format from a vector of other ReplacatedShare values.
    ///
    /// # Arguments
    ///
    /// `v` - vector of ReplicatedShares
    ///  `mode` - how to decide type of output. if `FromVectorMode::Vector` passed, it constructs a
    ///     ciphercore vector type. If `FromVectorMode::Tuple` passed, it construncts a ciphercore
    ///     tuple type, And if `FromVectorMode::AutoDetection` passed, it decides based on the type
    ///     of inputs, if they are all the same type it constructs a vector, otherwise a tuple.
    ///
    /// # Returns
    ///
    /// New ReplicatedShare constructed from `v`
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// # use ciphercore_base::typed_value_secret_shared::replicated_shares::ReplicatedShares;
    /// # use ciphercore_base::typed_value_secret_shared::TypedValueSecretShared;
    /// # use ciphercore_base::random::PRNG;
    /// # use ciphercore_base::data_types::INT32;
    ///
    /// let mut prng = PRNG::new(None).unwrap();
    ///
    /// let v = vec![
    ///         TypedValue::from_scalar(1, INT32).unwrap(),
    ///         TypedValue::from_scalar(423532, INT32).unwrap(),
    ///         TypedValue::from_scalar(-91, INT32).unwrap()];
    /// let v_of_shared = v.iter().map(|tv| ReplicatedShares::secret_share_for_local_evaluation(tv.clone(), &mut prng).unwrap()).collect();
    /// let v = ReplicatedShares::from_vector(v_of_shared, FromVectorMode::Vector).unwrap();
    /// ```
    fn from_vector(v: Vec<ReplicatedShares>, mode: FromVectorMode) -> Result<ReplicatedShares> {
        match mode {
            FromVectorMode::Vector => vector_from_vector_helper(v),
            FromVectorMode::Tuple => tuple_from_vector_helper(v),
            FromVectorMode::AutoDetection => {
                if v.is_empty() || !v.iter().all(|x| x.t.eq(&v[0].t)) {
                    tuple_from_vector_helper(v)
                } else {
                    vector_from_vector_helper(v)
                }
            }
        }
    }
    /// If `self` is Vector or Tuple or NamedTuple, gets index_th element of the collection.
    /// Retruns error if `self` is Scalar or Array.
    /// Returns error on out of bound access
    ///
    /// # Arguments
    ///
    /// `index` - index of the element in the vector or tuple
    ///
    /// # Returns
    ///
    /// index_th element of the collection
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// # use ciphercore_base::typed_value_secret_shared::replicated_shares::ReplicatedShares;
    /// # use ciphercore_base::typed_value_secret_shared::TypedValueSecretShared;
    /// # use ciphercore_base::random::PRNG;
    /// # use ciphercore_base::data_types::{INT32,BIT};
    ///
    /// let mut prng = PRNG::new(None).unwrap();
    /// let v = vec![TypedValue::from_scalar(1, BIT).unwrap(),
    ///          TypedValue::from_scalar(-91, INT32).unwrap()];
    /// let v_of_shared = v.iter().map(|tv| ReplicatedShares::secret_share_for_local_evaluation(tv.clone(), &mut prng).unwrap()).collect();
    /// let tv = ReplicatedShares::from_vector(v_of_shared, FromVectorMode::Tuple).unwrap();
    /// let tv1 = tv.get(1).unwrap();
    /// assert!(tv1.reveal().unwrap().eq(&TypedValue::from_scalar(-91, INT32).unwrap()));
    /// ```
    fn get(&self, index: usize) -> Result<ReplicatedShares> {
        get_helper::<ReplicatedShares>(self, index)
    }
    /// If `self` is Vector or Tuple or NamedTuple, insert to_insert_element to index_th position of the collection.
    /// Retrun error if `self` is Scalar or Array.
    /// Returns error on out of bound access
    /// If inserting to a vector the Type of to_insert_element should match the Type of exisiting vector's elements
    ///
    /// # Arguments
    ///
    /// `index` - index of the element in the vector or tuple
    /// `to_insert_element` - element to insert
    ///
    /// # Returns
    ///
    /// ()
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// # use ciphercore_base::typed_value_secret_shared::replicated_shares::ReplicatedShares;
    /// # use ciphercore_base::typed_value_secret_shared::TypedValueSecretShared;
    /// # use ciphercore_base::random::PRNG;
    /// # use ciphercore_base::data_types::{INT32,BIT,UINT64};
    ///
    /// let mut prng = PRNG::new(None).unwrap();
    /// let v = vec![TypedValue::from_scalar(1, BIT).unwrap(),
    ///          TypedValue::from_scalar(-91, INT32).unwrap()];
    /// let v_of_shared = v.iter().map(|tv| ReplicatedShares::secret_share_for_local_evaluation(tv.clone(), &mut prng).unwrap()).collect();
    /// let mut tv = ReplicatedShares::from_vector(v_of_shared, FromVectorMode::Tuple).unwrap();
    /// let to_insert_tv = TypedValue::from_scalar(2,UINT64).unwrap();
    /// let to_insert_tv_shared = ReplicatedShares::secret_share_for_local_evaluation(to_insert_tv, &mut prng).unwrap();
    /// tv.insert(to_insert_tv_shared,1).unwrap();
    /// ```
    fn insert(&mut self, to_insert_element: ReplicatedShares, index: usize) -> Result<()> {
        *self = insert_helper::<ReplicatedShares>(self, to_insert_element, index)?;
        Ok(())
    }
    /// If `self` is Vector or Tuple or NamedTuple, push to_insert_element to the end of the collection.
    /// Retrun error if `self` is Scalar or Array.
    /// If pushing to a vector, the Type of to_insert_element should match the Type of exisiting vector's elements
    ///
    /// # Arguments
    ///
    /// `index` - index of the element in the vector or tuple
    /// `to_insert_element` - element to insert
    ///
    /// # Returns
    ///
    /// ()
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// # use ciphercore_base::typed_value_secret_shared::replicated_shares::ReplicatedShares;
    /// # use ciphercore_base::typed_value_secret_shared::TypedValueSecretShared;
    /// # use ciphercore_base::random::PRNG;
    /// # use ciphercore_base::data_types::{INT32,BIT,UINT64};
    ///
    /// let mut prng = PRNG::new(None).unwrap();
    /// let v = vec![TypedValue::from_scalar(1, BIT).unwrap(),
    ///          TypedValue::from_scalar(-91, INT32).unwrap()];
    /// let v_of_shared = v.iter().map(|tv| ReplicatedShares::secret_share_for_local_evaluation(tv.clone(), &mut prng).unwrap()).collect();
    /// let mut tv = ReplicatedShares::from_vector(v_of_shared, FromVectorMode::Tuple).unwrap();
    /// let to_insert_tv = TypedValue::from_scalar(2,UINT64).unwrap();
    /// let to_insert_tv_shared = ReplicatedShares::secret_share_for_local_evaluation(to_insert_tv, &mut prng).unwrap();
    /// tv.push(to_insert_tv_shared).unwrap();
    /// ```
    fn push(&mut self, to_insert_element: ReplicatedShares) -> Result<()> {
        *self = push_helper::<ReplicatedShares>(self, to_insert_element)?;
        Ok(())
    }
    /// If `self` is Vector or Tuple or NamedTuple, it extends to_extend_collection to the end of the `self`
    /// Retrun error if `self` is Scalar or Array.
    /// If extending to a vector the Type of to_extend_collection should match the Type of `self` vector's elements
    ///
    /// # Arguments
    ///
    /// `index` - index of the element in the vector or tuple
    /// `to_extend_collection` - A Vector, Tuple, or NamedTuple to extend to `self`
    ///
    /// # Returns
    ///
    /// ()
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// # use ciphercore_base::typed_value_secret_shared::replicated_shares::ReplicatedShares;
    /// # use ciphercore_base::typed_value_secret_shared::TypedValueSecretShared;
    /// # use ciphercore_base::random::PRNG;
    /// # use ciphercore_base::data_types::{INT32,BIT,UINT64};
    ///
    /// let mut prng = PRNG::new(None).unwrap();
    /// let v = vec![TypedValue::from_scalar(1, BIT).unwrap(),
    ///          TypedValue::from_scalar(-91, INT32).unwrap()];
    /// let v_of_shared = v.iter().map(|tv| ReplicatedShares::secret_share_for_local_evaluation(tv.clone(), &mut prng).unwrap()).collect();
    /// let mut tv = ReplicatedShares::from_vector(v_of_shared, FromVectorMode::Tuple).unwrap();
    /// let  to_extend_tv = TypedValue::from_vector(
    ///          vec![TypedValue::from_scalar(2, UINT64).unwrap(),
    ///          TypedValue::from_scalar(0, BIT).unwrap()], FromVectorMode::Tuple).unwrap();
    /// let to_extend_tv_shared = ReplicatedShares::secret_share_for_local_evaluation(to_extend_tv, &mut prng).unwrap();
    /// tv.extend(to_extend_tv_shared).unwrap();
    /// ```
    fn extend(&mut self, to_extend_collection: ReplicatedShares) -> Result<()> {
        *self = extend_helper(self, to_extend_collection)?;
        Ok(())
    }
    /// If `self` is Vector or Tuple or NamedTuple, it removes index_th item of the `self`
    /// Retrun error if `self` is Scalar or Array.
    /// Returns error on out of bound remove
    ///
    /// # Arguments
    ///
    /// `index` - index of the element in the vector or tuple
    ///
    /// # Returns
    ///
    /// ()
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// # use ciphercore_base::typed_value_secret_shared::replicated_shares::ReplicatedShares;
    /// # use ciphercore_base::typed_value_secret_shared::TypedValueSecretShared;
    /// # use ciphercore_base::random::PRNG;
    /// # use ciphercore_base::data_types::{INT32,BIT,UINT64};
    ///
    /// let mut prng = PRNG::new(None).unwrap();
    /// let v = vec![TypedValue::from_scalar(1, BIT).unwrap(),
    ///          TypedValue::from_scalar(-91, INT32).unwrap()];
    /// let v_of_shared = v.iter().map(|tv| ReplicatedShares::secret_share_for_local_evaluation(tv.clone(), &mut prng).unwrap()).collect();
    /// let mut tv = ReplicatedShares::from_vector(v_of_shared, FromVectorMode::Tuple).unwrap();
    /// tv.remove(0).unwrap();
    /// ```
    fn remove(&mut self, index: usize) -> Result<()> {
        *self = remove_helper::<ReplicatedShares>(self, index)?;
        Ok(())
    }
    /// If `self` is Vector or Tuple or NamedTuple, it gets then removes index_th item of the `self`
    /// Retrun error if `self` is Scalar or Array.
    /// Returns errro on out of bound remove
    ///
    /// # Arguments
    ///
    /// `index` - index of the element in the vector or tuple
    ///
    /// # Returns
    ///
    /// ()
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// # use ciphercore_base::typed_value_secret_shared::replicated_shares::ReplicatedShares;
    /// # use ciphercore_base::typed_value_secret_shared::TypedValueSecretShared;
    /// # use ciphercore_base::random::PRNG;
    /// # use ciphercore_base::data_types::{INT32,BIT,UINT64};
    ///
    /// let mut prng = PRNG::new(None).unwrap();
    /// let v = vec![TypedValue::from_scalar(1, BIT).unwrap(),
    ///          TypedValue::from_scalar(-91, INT32).unwrap()];
    /// let v_of_shared = v.iter().map(|tv| ReplicatedShares::secret_share_for_local_evaluation(tv.clone(), &mut prng).unwrap()).collect();
    /// let mut tv = ReplicatedShares::from_vector(v_of_shared, FromVectorMode::Tuple).unwrap();
    /// let tv1 = tv.pop(0).unwrap();
    /// assert!(tv1.reveal().unwrap().eq(&TypedValue::from_scalar(1, BIT).unwrap()));
    /// ```
    fn pop(&mut self, index: usize) -> Result<ReplicatedShares> {
        let (out1, out2) = pop_helper::<ReplicatedShares>(self, index)?;
        *self = out2;
        Ok(out1)
    }
    /// If `self` is Vector or Tuple or NamedTuple, it gets a same collection of ReplicatedShares with a subset of `self` elements
    /// Retrun error if `self` is Scalar or Array.
    /// Returns error on out of bound remove
    ///
    /// # Arguments
    ///
    /// `start_index_option` - start index of the element in the vector or tuple, default value is `0` if `None` passed
    /// `start_index_option` - end index of the element in the vector or tuple, default value is end of the vector or tuple if `None` passed
    /// `step_option - step, default value is `1` if `None` passed
    /// # Returns
    ///
    /// A ReplicatedShares which is a  sub-vector of input ReplicatedShares
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// # use ciphercore_base::typed_value_secret_shared::replicated_shares::ReplicatedShares;
    /// # use ciphercore_base::typed_value_secret_shared::TypedValueSecretShared;
    /// # use ciphercore_base::random::PRNG;
    /// # use ciphercore_base::data_types::{INT32,BIT,UINT64};
    ///
    /// let mut prng = PRNG::new(None).unwrap();
    /// let v = vec![TypedValue::from_scalar(1, INT32).unwrap(),
    ///          TypedValue::from_scalar(3, INT32).unwrap(),
    ///          TypedValue::from_scalar(4, INT32).unwrap(),
    ///          TypedValue::from_scalar(5, INT32).unwrap(),
    ///          TypedValue::from_scalar(6, INT32).unwrap(),
    ///          TypedValue::from_scalar(7, INT32).unwrap()];
    /// let v_of_shared = v.iter().map(|tv| ReplicatedShares::secret_share_for_local_evaluation(tv.clone(), &mut prng).unwrap()).collect();
    /// let mut tv = ReplicatedShares::from_vector(v_of_shared, FromVectorMode::Tuple).unwrap();
    /// let tv1 = tv.get_sub_vector(Some(1),None,Some(2)).unwrap();
    /// ```
    fn get_sub_vector(
        &self,
        start_index_option: Option<usize>,
        end_index_option: Option<usize>,
        step_option: Option<usize>,
    ) -> Result<ReplicatedShares> {
        get_sub_vector_helper::<ReplicatedShares>(
            self,
            start_index_option,
            end_index_option,
            step_option,
        )
    }
}
impl ReplicatedShares {
    fn shard_to_shares(tv: &TypedValue, prng: &mut PRNG) -> Result<Vec<Value>> {
        let v0 = prng.get_random_value(tv.t.clone())?;
        let v1 = prng.get_random_value(tv.t.clone())?;
        let v2 = generalized_subtract(
            generalized_subtract(tv.value.clone(), v0.clone(), tv.t.clone())?,
            v1.clone(),
            tv.t.clone(),
        )?;
        Ok(vec![v0, v1, v2])
    }
}

impl TypedValueSecretShared<ReplicatedShares> for ReplicatedShares {
    /// Returns a complete ReplicatedShares of a typed value. This ReplicatedShares contains all the information
    /// that is needed to reconstruct the data, and can be used for local evaluation
    ///
    ///
    /// # Arguments
    ///
    /// TypedValue that needs to be shared
    /// mutable reference to prng
    ///
    /// # Returns
    ///
    /// ReplicatedShares of the input TypedValue
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_secret_shared::replicated_shares::ReplicatedShares;
    /// # use ciphercore_base::typed_value_secret_shared::TypedValueSecretShared;
    /// # use ciphercore_base::data_types::{scalar_type,INT32};
    /// # use ciphercore_base::random::PRNG;
    /// let mut prng = PRNG::new(None).unwrap();
    /// let tv = TypedValue::from_scalar(1, INT32).unwrap();
    /// let shared_tv = ReplicatedShares::secret_share_for_local_evaluation(tv, &mut prng).unwrap();
    /// ```
    fn secret_share_for_local_evaluation(
        tv: TypedValue,
        prng: &mut PRNG,
    ) -> Result<ReplicatedShares> {
        let vals = ReplicatedShares::shard_to_shares(&tv, prng)?;
        Ok(ReplicatedShares {
            shares: vals,
            t: tv.t,
            name: tv.name.clone(),
        })
    }
    /// Returns three ReplicatedShares of a typed value to be used for each party in MPC.
    ///
    /// # Arguments
    ///
    /// TypedValue that needs to be shared
    /// mutable reference to prng
    ///
    /// # Returns
    ///
    /// Three ReplicatedShares of the input TypedValue for three parties
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_secret_shared::replicated_shares::ReplicatedShares;
    /// # use ciphercore_base::typed_value_secret_shared::TypedValueSecretShared;
    /// # use ciphercore_base::data_types::{scalar_type,INT32};
    /// # use ciphercore_base::random::PRNG;
    /// let mut prng = PRNG::new(None).unwrap();
    /// let tv = TypedValue::from_scalar(1, INT32).unwrap();
    /// let shared_tv = ReplicatedShares::secret_share_for_parties(tv, &mut prng).unwrap();
    /// ```
    fn secret_share_for_parties(tv: TypedValue, prng: &mut PRNG) -> Result<Vec<ReplicatedShares>> {
        let vals = ReplicatedShares::shard_to_shares(&tv, prng)?;
        let mut garbage = vec![];
        for _ in 0..3 {
            garbage.push(prng.get_random_value(tv.t.clone())?);
        }
        let party0_share = ReplicatedShares {
            shares: vec![vals[0].clone(), vals[1].clone(), garbage[2].clone()],
            t: tv.t.clone(),
            name: tv.name.clone(),
        };
        let party1_share = ReplicatedShares {
            shares: vec![garbage[0].clone(), vals[1].clone(), vals[2].clone()],
            t: tv.t.clone(),
            name: tv.name.clone(),
        };
        let party2_share = ReplicatedShares {
            shares: vec![vals[0].clone(), garbage[1].clone(), vals[2].clone()],
            t: tv.t.clone(),
            name: tv.name.clone(),
        };
        Ok(vec![party0_share, party1_share, party2_share])
    }
    /// Convert ReplicatedShares to a tuple TypedValue.
    /// This tuple includes all three shares and is needed for graph evaluation
    ///
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    /// TypedValue in ciphercore tuple format
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_secret_shared::replicated_shares::ReplicatedShares;
    /// # use ciphercore_base::typed_value_secret_shared::TypedValueSecretShared;
    /// # use ciphercore_base::data_types::{scalar_type,INT32};
    /// # use ciphercore_base::random::PRNG;
    /// # use ciphercore_base::typed_value_operations::TypedValueOperations;
    /// let mut prng = PRNG::new(None).unwrap();
    /// let tv = TypedValue::from_scalar(1, INT32).unwrap();
    /// let shared_tv = ReplicatedShares::secret_share_for_local_evaluation(tv, &mut prng).unwrap();
    /// let tup = shared_tv.to_tuple().unwrap();
    /// ```
    fn to_tuple(&self) -> Result<TypedValue> {
        let shares_type = tuple_type(vec![self.t.clone(), self.t.clone(), self.t.clone()]);
        let shares_value = Value::from_vector(self.shares.clone());
        TypedValue::new(shares_type, shares_value)
    }

    /// Convert a tuple of TypedValue to a ReplicatedShares secret share value.
    /// The input tuple should include all three shares and all shares need to have the same type.
    /// This function is usefull to convert the output of compiled graphs to ReplicatedShares values.
    ///
    ///
    ///  # Arguments
    ///
    ///  TypedValue in ciphercore tuple format
    ///
    ///
    /// # Returns
    ///  
    ///  The ReplicatedShares built from the input tuple
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_secret_shared::replicated_shares::ReplicatedShares;
    /// # use ciphercore_base::typed_value_secret_shared::TypedValueSecretShared;
    /// # use ciphercore_base::data_types::{scalar_type,INT32};
    /// # use ciphercore_base::typed_value_operations::TypedValueOperations;
    /// # use ciphercore_base::random::PRNG;
    /// let mut prng = PRNG::new(None).unwrap();
    /// let tv = TypedValue::from_scalar(1, INT32).unwrap();
    /// let shared_tv = ReplicatedShares::secret_share_for_local_evaluation(tv, &mut prng).unwrap();
    /// let tup = shared_tv.to_tuple().unwrap();
    /// let shared_tv2 = ReplicatedShares::from_tuple(tup).unwrap();
    /// assert!(shared_tv2.is_equal(&shared_tv).unwrap());
    /// ```
    fn from_tuple(tv: TypedValue) -> Result<ReplicatedShares> {
        let t = tv.get_type();
        let v = tv.value;
        if let Type::Tuple(a) = t {
            let types_vec: Vec<Type> = a.iter().map(|x| x.as_ref().clone()).collect();
            let first_type = types_vec[0].clone();
            if !types_vec.iter().all(|x| *x == first_type) {
                return Err(runtime_error!("tuple elements should have same type"));
            }
            let value_vector = v.to_vector()?;
            Ok(ReplicatedShares {
                shares: value_vector,
                t: first_type,
                name: None,
            })
        } else {
            Err(runtime_error!("from_tuple called for non tuple value"))
        }
    }

    /// Reconstruct the TypedValue from a complete ReplicatedShares.
    ///
    /// # Arguments
    ///
    /// # Returns
    ///
    /// Reconstructed TypedValue
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_secret_shared::replicated_shares::ReplicatedShares;
    /// # use ciphercore_base::typed_value_operations::TypedValueOperations;
    /// # use ciphercore_base::typed_value_secret_shared::TypedValueSecretShared;
    /// # use ciphercore_base::data_types::{scalar_type,INT32};
    /// # use ciphercore_base::random::PRNG;
    /// let mut prng = PRNG::new(None).unwrap();
    /// let tv = TypedValue::from_scalar(1, INT32).unwrap();
    /// let shared_tv = ReplicatedShares::secret_share_for_local_evaluation(tv.clone(), &mut prng).unwrap();
    /// let revealed = shared_tv.reveal().unwrap();
    /// assert!(revealed.is_equal(&tv).unwrap());
    /// ```
    fn reveal(&self) -> Result<TypedValue> {
        let v_output = generalized_add(
            generalized_add(
                self.shares[0].clone(),
                self.shares[1].clone(),
                self.t.clone(),
            )?,
            self.shares[2].clone(),
            self.t.clone(),
        )?;
        Ok(TypedValue {
            t: self.t.clone(),
            value: v_output,
            name: self.name.clone(),
        })
    }
}

fn vector_from_vector_helper(v: Vec<ReplicatedShares>) -> Result<ReplicatedShares> {
    let val_type = if v.is_empty() {
        tuple_type(vec![])
    } else {
        v[0].t.clone()
    };
    let t = vector_type(v.len() as u64, val_type.clone());
    let mut val0_vec = vec![];
    let mut val1_vec = vec![];
    let mut val2_vec = vec![];
    for val in v {
        if !val.t.eq(&val_type) {
            return Err(runtime_error!(
                "Can not distinguish the type: vector has different types"
            ));
        }
        val0_vec.push(val.shares[0].clone());
        val1_vec.push(val.shares[1].clone());
        val2_vec.push(val.shares[2].clone());
    }
    Ok(ReplicatedShares {
        shares: vec![
            Value::from_vector(val0_vec),
            Value::from_vector(val1_vec),
            Value::from_vector(val2_vec),
        ],
        t,
        name: None,
    })
}

fn tuple_from_vector_helper(v: Vec<ReplicatedShares>) -> Result<ReplicatedShares> {
    let named = v.iter().all(|x| x.name.is_some());
    let unnamed = v.iter().all(|x| x.name.is_none());
    if unnamed {
        let t = tuple_type(v.iter().map(|v| v.t.clone()).collect());
        let val0_vec = v.iter().map(|v| v.shares[0].clone()).collect();
        let val1_vec = v.iter().map(|v| v.shares[1].clone()).collect();
        let val2_vec = v.iter().map(|v| v.shares[2].clone()).collect();
        Ok(ReplicatedShares {
            shares: vec![
                Value::from_vector(val0_vec),
                Value::from_vector(val1_vec),
                Value::from_vector(val2_vec),
            ],
            t,
            name: None,
        })
    } else if named {
        let t = named_tuple_type(
            v.iter()
                .map(|v| (v.name.clone().unwrap(), v.t.clone()))
                .collect(),
        );
        let val0_vec = v.iter().map(|v| v.shares[0].clone()).collect();
        let val1_vec = v.iter().map(|v| v.shares[1].clone()).collect();
        let val2_vec = v.iter().map(|v| v.shares[2].clone()).collect();
        Ok(ReplicatedShares {
            shares: vec![
                Value::from_vector(val0_vec),
                Value::from_vector(val1_vec),
                Value::from_vector(val2_vec),
            ],
            t,
            name: None,
        })
    } else {
        Err(runtime_error!("Can not mix named and unnamed tuple"))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::{scalar_type, tuple_type, BIT, INT32, UINT16, UINT32, UINT64, UINT8};
    use crate::typed_value_secret_shared::TypedValueSecretShared;
    use std::ops::Add;

    #[test]
    fn test_secret_sharing() {
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            for bv in 0..2 {
                let tv = TypedValue::new(scalar_type(BIT), Value::from_bytes(vec![bv]))?;
                for _ in 0..10 {
                    let tv1 =
                        ReplicatedShares::secret_share_for_local_evaluation(tv.clone(), &mut prng)?;
                    let mut result = vec![];
                    for v in tv1.shares.clone() {
                        v.access_bytes(|b| {
                            assert_eq!(b.len(), 1);
                            result.push(b[0]);
                            Ok(())
                        })?;
                    }
                    assert_eq!(result[0] ^ result[1] ^ result[2], bv);
                    assert_eq!(tv1.reveal()?, tv);
                }
            }
            for _ in 0..10 {
                let tv = TypedValue::new(
                    scalar_type(INT32),
                    prng.get_random_value(scalar_type(INT32))?,
                )?;
                for _ in 0..10 {
                    let tv1 =
                        ReplicatedShares::secret_share_for_local_evaluation(tv.clone(), &mut prng)?;
                    let mut result = vec![];
                    for v in tv1.shares.clone() {
                        result.push(v.to_i32(INT32)?);
                    }
                    assert_eq!(
                        result[0]
                            .overflowing_add(result[1])
                            .0
                            .overflowing_add(result[2])
                            .0,
                        tv.value.to_i32(INT32)?
                    );
                    assert_eq!(tv1.reveal()?, tv);
                }
            }
            for _ in 0..10 {
                let t = tuple_type(vec![scalar_type(INT32), scalar_type(BIT)]);
                let tv = TypedValue::new(t.clone(), prng.get_random_value(t.clone())?)?;
                for _ in 0..10 {
                    let tv1 =
                        ReplicatedShares::secret_share_for_local_evaluation(tv.clone(), &mut prng)?;
                    let mut result = vec![];
                    for v in tv1.shares.clone() {
                        v.access_vector(|v1| {
                            assert_eq!(v1.len(), 2);
                            result.push((v1[0].to_i32(INT32)?, v1[1].to_u8(BIT)?));
                            Ok(())
                        })?;
                    }
                    let ov = tv
                        .value
                        .access_vector(|v| Ok((v[0].to_i32(INT32)?, v[1].to_u8(BIT)?)))?;
                    assert_eq!(
                        result[0]
                            .0
                            .overflowing_add(result[1].0)
                            .0
                            .overflowing_add(result[2].0)
                            .0,
                        ov.0
                    );
                    assert_eq!(result[0].1 ^ result[1].1 ^ result[2].1, ov.1);
                    assert_eq!(tv1.reveal()?, tv);
                }
            }
            for _ in 0..10 {
                let t = named_tuple_type(vec![
                    ("field1".to_owned(), scalar_type(INT32)),
                    ("field2".to_owned(), scalar_type(BIT)),
                ]);
                let tv = TypedValue::new(t.clone(), prng.get_random_value(t.clone())?)?;
                for _ in 0..10 {
                    let tv1 =
                        ReplicatedShares::secret_share_for_local_evaluation(tv.clone(), &mut prng)?;
                    let mut result = vec![];
                    for v in tv1.shares.clone() {
                        v.access_vector(|v1| {
                            assert_eq!(v1.len(), 2);
                            result.push((v1[0].to_i32(INT32)?, v1[1].to_u8(BIT)?));
                            Ok(())
                        })?;
                    }
                    let ov = tv
                        .value
                        .access_vector(|v| Ok((v[0].to_i32(INT32)?, v[1].to_u8(BIT)?)))?;
                    assert_eq!(
                        result[0]
                            .0
                            .overflowing_add(result[1].0)
                            .0
                            .overflowing_add(result[2].0)
                            .0,
                        ov.0
                    );
                    assert_eq!(result[0].1 ^ result[1].1 ^ result[2].1, ov.1);
                    assert_eq!(tv1.reveal()?, tv);
                }
            }
            for _ in 0..10 {
                let t = vector_type(2, scalar_type(INT32));
                let tv = TypedValue::new(t.clone(), prng.get_random_value(t.clone())?)?;
                for _ in 0..10 {
                    let tv1 =
                        ReplicatedShares::secret_share_for_local_evaluation(tv.clone(), &mut prng)?;
                    let mut result = vec![];
                    for v in tv1.shares.clone() {
                        v.access_vector(|v1| {
                            assert_eq!(v1.len(), 2);
                            result.push((v1[0].to_i32(INT32)?, v1[1].to_i32(INT32)?));
                            Ok(())
                        })?;
                    }
                    let ov = tv
                        .value
                        .access_vector(|v| Ok((v[0].to_i32(INT32)?, v[1].to_i32(INT32)?)))?;
                    assert_eq!(
                        result[0]
                            .0
                            .overflowing_add(result[1].0)
                            .0
                            .overflowing_add(result[2].0)
                            .0,
                        ov.0
                    );
                    assert_eq!(
                        result[0]
                            .1
                            .overflowing_add(result[1].1)
                            .0
                            .overflowing_add(result[2].1)
                            .0,
                        ov.1
                    );
                    assert_eq!(tv1.reveal()?, tv);
                }
            }
            for _ in 0..10 {
                let t = array_type(vec![2], INT32);
                let tv = TypedValue::new(t.clone(), prng.get_random_value(t.clone())?)?;
                for _ in 0..10 {
                    let tv1 =
                        ReplicatedShares::secret_share_for_local_evaluation(tv.clone(), &mut prng)?;
                    let mut result = vec![];
                    for v in tv1.shares.clone() {
                        result.push(v.to_flattened_array_i32(t.clone())?);
                    }
                    let ov = tv.value.to_flattened_array_i32(t.clone())?;
                    assert_eq!(
                        result[0][0]
                            .overflowing_add(result[1][0])
                            .0
                            .overflowing_add(result[2][0])
                            .0,
                        ov[0]
                    );
                    assert_eq!(
                        result[0][1]
                            .overflowing_add(result[1][1])
                            .0
                            .overflowing_add(result[2][1])
                            .0,
                        ov[1]
                    );
                    assert_eq!(tv1.reveal()?, tv);
                }
            }
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_secret_sharing_for_parties() {
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            let data = vec![12, 34, 56];
            let value = TypedValue::from_ndarray(
                ndarray::Array::from_vec(data.clone()).into_dyn(),
                UINT64,
            )?;
            let shares = ReplicatedShares::secret_share_for_parties(value, &mut prng)?;
            let shares0 = shares[0].to_tuple()?.to_vector()?;
            let shares1 = shares[1].to_tuple()?.to_vector()?;
            let shares2 = shares[2].to_tuple()?.to_vector()?;
            assert_eq!(shares0[0], shares2[0]);
            assert_eq!(shares0[1], shares1[1]);
            assert_eq!(shares1[2], shares2[2]);
            let m = UINT32.get_modulus().unwrap();
            let v0 = ToNdarray::<u128>::to_ndarray(&shares0[0])? % m;
            let v1 = ToNdarray::<u128>::to_ndarray(&shares1[1])? % m;
            let v2 = ToNdarray::<u128>::to_ndarray(&shares2[2])? % m;
            let new_data = v0.add(v1).add(v2) % m;
            assert_eq!(new_data, ndarray::Array::from_vec(data).into_dyn());
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_get() {
        //test get from tuple success/failure
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            let t0 = tuple_type(vec![scalar_type(UINT8), scalar_type(UINT16)]);
            let zero0 = TypedValue::zero_of_type(t0);
            let zero0_shared =
                ReplicatedShares::secret_share_for_local_evaluation(zero0, &mut prng)?;
            let res_ok = zero0_shared.get(1);
            //ok case
            assert!(res_ok.is_ok());
            let value_got = res_ok?;
            let tv_expected = TypedValue::from_scalar(0, UINT16)?;
            assert!(tv_expected.is_equal(&value_got.reveal()?)?);
            // out of bound access case
            let res_err = zero0_shared.get(4);
            assert!(res_err.is_err());
            Ok(())
        }()
        .unwrap();
        //test get from vector success/failure
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            let t0 = vector_type(2, scalar_type(UINT8));
            let zero0 = TypedValue::zero_of_type(t0);
            let zero0_shared =
                ReplicatedShares::secret_share_for_local_evaluation(zero0, &mut prng)?;
            let res_ok = zero0_shared.get(1);
            //ok case
            assert!(res_ok.is_ok());
            let value_got = res_ok?;
            let tv_expected = TypedValue::from_scalar(0, UINT8)?;
            assert!(tv_expected.is_equal(&value_got.reveal()?)?);
            // out of bound access case
            let res_err = zero0_shared.get(4);
            assert!(res_err.is_err());
            Ok(())
        }()
        .unwrap();
    }
    #[test]
    fn test_insert() {
        //test insert to tuple success/failure
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            let t0 = tuple_type(vec![scalar_type(UINT8), scalar_type(UINT16)]);
            let zero0 = TypedValue::zero_of_type(t0);
            let mut zero0_shared =
                ReplicatedShares::secret_share_for_local_evaluation(zero0, &mut prng)?;
            let to_insert = TypedValue::from_scalar(20, UINT32)?;
            let to_insert_shared =
                ReplicatedShares::secret_share_for_local_evaluation(to_insert, &mut prng)?;
            let res_ok = zero0_shared.insert(to_insert_shared.clone(), 1);
            //ok case
            assert!(res_ok.is_ok());
            let revealed = zero0_shared.reveal()?;
            let t_expected = tuple_type(vec![
                scalar_type(UINT8),
                scalar_type(UINT32),
                scalar_type(UINT16),
            ]);
            let value_expected = Value::from_vector(vec![
                Value::from_scalar(0, UINT8).unwrap(),
                Value::from_scalar(20, UINT32).unwrap(),
                Value::from_scalar(0, UINT16).unwrap(),
            ]);
            let tv_expected = TypedValue::new(t_expected, value_expected)?;
            assert!(tv_expected.is_equal(&revealed)?);
            // out of bound insert case
            let res_err = zero0_shared.insert(to_insert_shared, 4);
            assert!(res_err.is_err());
            Ok(())
        }()
        .unwrap();
        //test insert to vector success/failure
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            let t0 = vector_type(2, scalar_type(UINT8));
            let zero0 = TypedValue::zero_of_type(t0);
            let mut zero0_shared =
                ReplicatedShares::secret_share_for_local_evaluation(zero0, &mut prng)?;
            let to_insert = TypedValue::from_scalar(20, UINT8)?;
            let to_insert_shared =
                ReplicatedShares::secret_share_for_local_evaluation(to_insert, &mut prng)?;
            let res_ok = zero0_shared.insert(to_insert_shared.clone(), 1);
            //ok case
            assert!(res_ok.is_ok());
            let revealed = zero0_shared.reveal()?;
            let t_expected = vector_type(3, scalar_type(UINT8));
            let value_expected = Value::from_vector(vec![
                Value::from_scalar(0, UINT8).unwrap(),
                Value::from_scalar(20, UINT8).unwrap(),
                Value::from_scalar(0, UINT8).unwrap(),
            ]);
            let tv_expected = TypedValue::new(t_expected, value_expected)?;
            assert!(tv_expected.is_equal(&revealed)?);
            // out of bound insert case
            let res_err = zero0_shared.insert(to_insert_shared.clone(), 4);
            assert!(res_err.is_err());
            // wrong type insert case
            let to_insert = TypedValue::from_scalar(20, UINT16)?;
            let to_insert_shared =
                ReplicatedShares::secret_share_for_local_evaluation(to_insert, &mut prng)?;
            let res_err = zero0_shared.insert(to_insert_shared.clone(), 1);
            assert!(res_err.is_err());

            Ok(())
        }()
        .unwrap();
        //test insert to incompatible types
        || -> Result<()> {
            //insert to scalar
            let mut prng = PRNG::new(None)?;
            let t0 = scalar_type(UINT8);
            let zero0 = TypedValue::zero_of_type(t0);
            let mut zero0_shared =
                ReplicatedShares::secret_share_for_local_evaluation(zero0, &mut prng)?;
            let to_insert = TypedValue::from_scalar(20, UINT8)?;
            let to_insert_shared =
                ReplicatedShares::secret_share_for_local_evaluation(to_insert, &mut prng)?;
            let res_err = zero0_shared.insert(to_insert_shared.clone(), 1);
            assert!(res_err.is_err());
            //insert to array
            let t0 = array_type(vec![2, 2], UINT8);
            let zero0 = TypedValue::zero_of_type(t0);
            let mut zero0_shared =
                ReplicatedShares::secret_share_for_local_evaluation(zero0, &mut prng)?;
            let to_insert = TypedValue::from_scalar(20, UINT8)?;
            let to_insert_shared =
                ReplicatedShares::secret_share_for_local_evaluation(to_insert, &mut prng)?;
            let res_err = zero0_shared.insert(to_insert_shared.clone(), 1);
            assert!(res_err.is_err());
            Ok(())
        }()
        .unwrap();
    }
    #[test]
    fn test_push() {
        //test push to tuple success
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            let t0 = tuple_type(vec![scalar_type(UINT8), scalar_type(UINT16)]);
            let zero0 = TypedValue::zero_of_type(t0);
            let mut zero0_shared =
                ReplicatedShares::secret_share_for_local_evaluation(zero0, &mut prng)?;
            let to_insert = TypedValue::from_scalar(20, UINT32)?;
            let to_insert_shared =
                ReplicatedShares::secret_share_for_local_evaluation(to_insert, &mut prng)?;
            let res_ok = zero0_shared.push(to_insert_shared.clone());
            //ok case
            assert!(res_ok.is_ok());
            let revealed = zero0_shared.reveal()?;
            let tv_expected = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(0, UINT16).unwrap(),
                    TypedValue::from_scalar(20, UINT32).unwrap(),
                ],
                FromVectorMode::Tuple,
            )
            .unwrap();
            assert!(tv_expected.is_equal(&revealed)?);
            Ok(())
        }()
        .unwrap();
        //test push to tuple fail
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            let t0 = tuple_type(vec![scalar_type(UINT8), scalar_type(UINT16)]);
            let zero0 = TypedValue::zero_of_type(t0);
            let mut zero0_shared =
                ReplicatedShares::secret_share_for_local_evaluation(zero0, &mut prng)?;
            let mut to_insert = TypedValue::from_scalar(20, UINT32)?;
            to_insert.name = Some("name".to_owned());
            let to_insert_shared =
                ReplicatedShares::secret_share_for_local_evaluation(to_insert, &mut prng)?;
            let res_ok = zero0_shared.push(to_insert_shared.clone());
            //err case
            assert!(res_ok.is_err());
            Ok(())
        }()
        .unwrap();
        //test push to named tuple success
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            let t0 = named_tuple_type(vec![
                ("name1".to_owned(), scalar_type(UINT8)),
                ("name2".to_owned(), scalar_type(UINT16)),
            ]);
            let zero0 = TypedValue::zero_of_type(t0);
            let mut zero0_shared =
                ReplicatedShares::secret_share_for_local_evaluation(zero0, &mut prng)?;
            let mut to_insert = TypedValue::from_scalar(20, UINT32)?;
            to_insert.name = Some("name".to_owned());
            let to_insert_shared =
                ReplicatedShares::secret_share_for_local_evaluation(to_insert.clone(), &mut prng)?;
            let res_ok = zero0_shared.push(to_insert_shared.clone());
            //ok case
            assert!(res_ok.is_ok());
            let revealed = zero0_shared.reveal()?;
            let mut tv1 = TypedValue::from_scalar(0, UINT8).unwrap();
            tv1.name = Some("name1".to_owned());
            let mut tv2 = TypedValue::from_scalar(0, UINT16).unwrap();
            tv2.name = Some("name2".to_owned());
            let tv3 = to_insert.clone();
            let tv_expected =
                TypedValue::from_vector(vec![tv1, tv2, tv3], FromVectorMode::Tuple).unwrap();
            assert!(tv_expected.is_equal(&revealed)?);
            Ok(())
        }()
        .unwrap();
    }
    #[test]
    fn test_remove() {
        //test remove from tuple
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            let tv = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(0, UINT16).unwrap(),
                    TypedValue::from_scalar(20, UINT32).unwrap(),
                ],
                FromVectorMode::Tuple,
            )
            .unwrap();
            let mut tv_shared = ReplicatedShares::secret_share_for_local_evaluation(tv, &mut prng)?;
            let res = tv_shared.remove(2);
            assert!(res.is_ok());
            let reveled = tv_shared.reveal()?;
            let tv_expected = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(0, UINT16).unwrap(),
                ],
                FromVectorMode::Tuple,
            )
            .unwrap();
            assert!(tv_expected.is_equal(&reveled)?);
            //failure case : out of bound
            assert!(tv_shared.remove(2).is_err());
            Ok(())
        }()
        .unwrap();
    }
    #[test]
    fn test_pop() {
        //test pop from vector
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            let tv = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(20, UINT8).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            let mut tv_shared = ReplicatedShares::secret_share_for_local_evaluation(tv, &mut prng)?;
            let element_shared = tv_shared.pop(2)?;
            let tv_revealed = tv_shared.reveal()?;
            let element_revealed = element_shared.reveal()?;
            let tv_expected = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            assert!(tv_expected.is_equal(&tv_revealed)?);
            assert!(element_revealed.is_equal(&TypedValue::from_scalar(20, UINT8).unwrap())?);
            //failure case : out of bound
            assert!(tv_shared.pop(2).is_err());
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_extend() {
        //test extend vector to vector
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            let tv = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(10, UINT8).unwrap(),
                    TypedValue::from_scalar(20, UINT8).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            let mut tv_shared = ReplicatedShares::secret_share_for_local_evaluation(tv, &mut prng)?;
            let tv_to_extend = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(30, UINT8).unwrap(),
                    TypedValue::from_scalar(40, UINT8).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            let tv_to_extend_shared =
                ReplicatedShares::secret_share_for_local_evaluation(tv_to_extend, &mut prng)?;
            let tv_expected = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(10, UINT8).unwrap(),
                    TypedValue::from_scalar(20, UINT8).unwrap(),
                    TypedValue::from_scalar(30, UINT8).unwrap(),
                    TypedValue::from_scalar(40, UINT8).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            tv_shared.extend(tv_to_extend_shared)?;
            let tv_revealed = tv_shared.reveal()?;
            assert!(tv_expected.is_equal(&tv_revealed)?);
            Ok(())
        }()
        .unwrap();
        //test extend incompatible vector to vector: failure
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            let tv = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(10, UINT8).unwrap(),
                    TypedValue::from_scalar(20, UINT8).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            let mut tv_shared = ReplicatedShares::secret_share_for_local_evaluation(tv, &mut prng)?;
            let tv_to_extend = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(30, UINT16).unwrap(),
                    TypedValue::from_scalar(40, UINT16).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            let tv_to_extend_shared =
                ReplicatedShares::secret_share_for_local_evaluation(tv_to_extend, &mut prng)?;
            let res = tv_shared.extend(tv_to_extend_shared);
            assert!(res.is_err());
            Ok(())
        }()
        .unwrap();
        //test extend incompatible vector to tuple: success
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            let tv = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(10, UINT8).unwrap(),
                    TypedValue::from_scalar(20, UINT8).unwrap(),
                ],
                FromVectorMode::Tuple,
            )
            .unwrap();
            let mut tv_shared = ReplicatedShares::secret_share_for_local_evaluation(tv, &mut prng)?;
            let tv_to_extend = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(30, UINT16).unwrap(),
                    TypedValue::from_scalar(40, UINT16).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            let tv_to_extend_shared =
                ReplicatedShares::secret_share_for_local_evaluation(tv_to_extend, &mut prng)?;
            let res = tv_shared.extend(tv_to_extend_shared);
            let tv_revealed = tv_shared.reveal()?;
            assert!(res.is_ok());
            let tv_expected = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(10, UINT8).unwrap(),
                    TypedValue::from_scalar(20, UINT8).unwrap(),
                    TypedValue::from_scalar(30, UINT16).unwrap(),
                    TypedValue::from_scalar(40, UINT16).unwrap(),
                ],
                FromVectorMode::Tuple,
            )
            .unwrap();
            assert!(tv_revealed.is_equal(&tv_expected)?);
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_get_sub_vector() {
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            let tv = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT16).unwrap(),
                    TypedValue::from_scalar(1, UINT16).unwrap(),
                    TypedValue::from_scalar(2, UINT16).unwrap(),
                    TypedValue::from_scalar(3, UINT16).unwrap(),
                    TypedValue::from_scalar(4, UINT16).unwrap(),
                    TypedValue::from_scalar(5, UINT16).unwrap(),
                    TypedValue::from_scalar(6, UINT16).unwrap(),
                    TypedValue::from_scalar(7, UINT16).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            let tv_shared = ReplicatedShares::secret_share_for_local_evaluation(tv, &mut prng)?;
            let sub_v = tv_shared.get_sub_vector(None, None, Some(2))?;
            let sub_v_revealed = sub_v.reveal()?;
            let tv_expected = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT16).unwrap(),
                    TypedValue::from_scalar(2, UINT16).unwrap(),
                    TypedValue::from_scalar(4, UINT16).unwrap(),
                    TypedValue::from_scalar(6, UINT16).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            assert!(sub_v_revealed.is_equal(&tv_expected)?);

            let res = tv_shared.get_sub_vector(None, Some(12), Some(1));
            assert!(res.is_err());

            let sub_v2 = tv_shared.get_sub_vector(Some(1), None, Some(20))?;
            let tv_expected2 = TypedValue::from_vector(
                vec![TypedValue::from_scalar(1, UINT16).unwrap()],
                FromVectorMode::Vector,
            )
            .unwrap();
            let sub_v2_revealed = sub_v2.reveal()?;
            assert!(sub_v2_revealed.is_equal(&tv_expected2)?);
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_ndarray() {
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            let t = array_type(vec![2, 5], INT32);
            let tv = TypedValue::new(t.clone(), prng.get_random_value(t.clone())?)?;
            let shared_tv =
                ReplicatedShares::secret_share_for_local_evaluation(tv.clone(), &mut prng)?;
            let arr_shared = ToNdarray::<i32>::to_ndarray(&shared_tv)?;
            let shared_tv_double_converted = ReplicatedShares::from_ndarray(arr_shared, INT32)?;
            let tv_doudle_converted = shared_tv_double_converted.reveal()?;
            assert!(tv.is_equal(&tv_doudle_converted)?);
            assert!(shared_tv.is_equal(&shared_tv_double_converted)?);
            Ok(())
        }()
        .unwrap();
    }
}
