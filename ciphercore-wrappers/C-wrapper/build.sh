set -x
gcc $1.c ../../target/release/libcadapter.so -ldl -lpthread -lssl -lcrypto -lm -o $1.o
set +x

