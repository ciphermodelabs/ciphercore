#ifndef C_RESULT_H
#define C_RESULT_H
#include "Error.h"
#include <stdint.h>

#define IS_ERR(cres) (cres.tag == Err)
#define IS_OK(cres) (cres.tag == Ok)
#define UNWRAP(cres) ({if(IS_ERR(cres)) PANIC(cres.err); cres.ok;})
#define GET_ERR(cres) ({if(IS_ERR(cres)) cres.err;})

typedef uint64_t u64;

#define CRESULT(T) \
typedef struct CResult_##T {\
	  CResult_Tag tag; \
  union { \
    struct { \
      struct T *ok; \
    }; \
    struct { \
      struct CiphercoreError err; \
    }; \
  }; \
} CResult_##T\

#define CRESULT_VAL(T) \
typedef struct CResultVal_##T {\
	  CResult_Tag tag; \
  union { \
    struct { \
      T ok; \
    }; \
    struct { \
      struct CiphercoreError err; \
    }; \
  }; \
} CResultVal_##T\


typedef enum CResult_Tag {
  Ok,
  Err,
} CResult_Tag;


CRESULT(Context);
CRESULT(Graph);
CRESULT(Node);
CRESULT(Type);
CRESULT(ScalarType);
CRESULT(CVecVal_u64);
CRESULT(CVec_Type);
CRESULT(CVec_Node);
CRESULT(CVec_Graph);
CRESULT(COperation);

CRESULT_VAL(u64);
CRESULT_VAL(bool);
CRESULT_VAL(CStr);


#endif // CRESULT_H

