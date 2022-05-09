#ifndef CIPHERCORE_DATA_TYPES
#define CIPHERCORE_DATA_TYPES

#include "ciphercore_structs.h"
#include "ciphercore_auxiliary_structs.h"
struct CResultVal_bool scalar_type_get_signed(const struct ScalarType *st_ptr);

struct CResultVal_u64 scalar_type_get_modulus(const struct ScalarType *st_ptr);

struct CResultVal_bool type_is_scalar(struct Type *type_ptr);

struct CResultVal_bool type_is_array(struct Type *type_ptr);

struct CResultVal_bool type_is_vector(struct Type *type_ptr);

struct CResultVal_bool type_is_tuple(struct Type *type_ptr);

struct CResultVal_bool type_is_named_tuple(struct Type *type_ptr);

struct CResult_ScalarType type_get_scalar_type(struct Type *type_ptr);

struct CResult_CVecVal_u64 type_get_shape(struct Type *type_ptr);

struct CResult_CVecVal_u64 type_get_dimensions(struct Type *type_ptr);

struct CResult_Type scalar_type(const struct ScalarType *st_ptr);

struct CResult_Type array_type(struct CVecVal_u64 shape, const struct ScalarType *st_ptr);

struct CResult_Type vector_type(uint64_t n, struct Type *t_ptr);

struct CResult_Type tuple_type(struct CVec_Type type_ptrs_cvec);

struct CResult_Type named_tuple_type(struct CVecVal_CStr cstr_cvec, struct CVec_Type type_ptrs_cvec);

struct CResultVal_u64 scalar_size_in_bits(const struct ScalarType *st_ptr);

struct CResultVal_u64 get_size_in_bits(struct Type *t_ptr);

struct CResult_CVec_Type get_types_vector(struct Type *t_ptr);

void scalar_type_destroy(struct ScalarType *st_ptr);

struct CResultVal_CStr scalar_type_to_string(const struct ScalarType *st_ptr);

struct CResultVal_CStr type_to_string(struct Type *type_ptr);

void type_destroy(struct Type *t_ptr);
#endif //CIPHERCORE_DATA_TYPES

