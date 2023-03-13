# Changes in CipherCore 0.1.3

* **Better SMPC protocol for Private set intersection** The new protocol follows the description of the Join protocol from <https://eprint.iacr.org/2019/518.pdf>. The operation is renamed from `SetIntersection` to `Join`, and supports four join flavors: Inner, Left, Union and Full.
* **Concatenate** Primitive operation to concatenate arrays.
* **Apply permutation** Primitive operation that permutes private or public data using a private or public permutation.
* **General matrix multiplication** Primitive operation `Gemm` that generalizes `Matmul`.
* **Division** Two algorithms implemented as custom operations: Approximate GoldschmidtDivision and slow but exact LongDivision.

# Changes in CipherCore 0.1.2

* **Runtime documentation + examples** We now provide [detailed documentation](https://github.com/ciphermodelabs/ciphercore/blob/main/reference/runtime.md) for CipherCore runtime, which can be used to execute a secure protocol produced by CipherCore compiler between actual parties over the network. [E-mail us](mailto:ciphercore@ciphermode.tech) to request access to play with runtime.
* **Private set intersection** We added a simple implementation of [private set intersection](https://en.wikipedia.org/wiki/Private_set_intersection) based on sorting (available as a binary `ciphercore_set_intersection`).
* **Exponent, Sigmoid, GeLU** We implemented efficiently several non-linear transformations that are crucial for machine learning (e.g., transformers).
* **Better SMPC protocol for Truncate** The newly implemented protocol for the Truncate operation is more stable and never makes large errors, which makes it more suitable for fixed-point arithmetic necessary for training machine learning models.
* **MixedMultiply + improvements in InverseSqrt and NewtonInversion** We added a new operation and the corresponding secure protocol for multiplying a bit and an integer avoid an expensive conversion. This allows to significantly improve the performance of various operations including ReLU and division.
* **Sorting: custom operation + support for signed integers** Now sorting is implemented as a custom operation, and it supports signed integers inputs.
* **Efficiency improvements: optimizer, log-depth inlining** We added several crucial improvements to the optimizing and inlining passes of the compiler.
* **More complete and documented Python wrapper** Finally, we overhauled and improved our Python wrapper pretty significantly. In particular, almost all the classes and functions now have readable docstrings.