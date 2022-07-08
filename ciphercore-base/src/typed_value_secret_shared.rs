pub mod replicated_shares;
use crate::errors::Result;
use crate::random::PRNG;
use crate::typed_value::TypedValue;
use crate::typed_value_operations::TypedValueOperations;

pub trait TypedValueSecretShared<T>: TypedValueOperations<T> {
    // It secret shares a TypedValue and returns one TypeValueSecretShared that can be converted to
    // a tuple of TypedValue and be used in local evaluation of MPC compiled graphs outside of runtime.
    fn secret_share_for_local_evaluation(tv: TypedValue, prng: &mut PRNG) -> Result<T>;
    // It secret shares a TypedValue and returns a vector of TypedValueSecretShared,
    // one TypedValueSecretShared for each party. Parties can use these shares for evaluation of
    // compiled graphs in the runtime.
    fn secret_share_for_parties(tv: TypedValue, prng: &mut PRNG) -> Result<Vec<T>>;
    fn to_tuple(&self) -> Result<TypedValue>;
    fn from_tuple(tv: TypedValue) -> Result<T>;
    fn reveal(&self) -> Result<TypedValue>;
}
