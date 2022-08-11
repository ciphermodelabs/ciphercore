# CipherCore

CipherCore is a user-friendly secure computation engine based on [secure multi-party computation](https://en.wikipedia.org/wiki/Secure_multi-party_computation).

To install CipherCore, run one of the following commands:
* `pip install ciphercore` -- installs the Python wrapper for CipherCore computation graph API
* `cargo install ciphercore-base` -- builds and installs the CipherCore compiler and other CLI tools from source (requires [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html))
* `docker pull ciphermodelabs/ciphercore:latest` -- pulls a Docker image with binary distribution of CipherCore
* `docker pull ciphermodelabs/runtime_example:latest` -- pulls a Docker image that contains CipherCore runtime (requires an access token, [e-mail us](mailto:ciphercore@ciphermode.tech) to request access).

Check out the [complete documentation](https://github.com/ciphermodelabs/ciphercore/blob/main/reference/main.md), which includes a tutorial, several examples and a comprehensive guide to CipherCore.

If you have any questions, or, more generally, would like to discuss CipherCore, please [join the Slack community](https://join.slack.com/t/slack-r5s9809/shared_invite/zt-1901t4ec3-W4pk~nsTl2dY8Is5HFWT4w).

# Five-minute intro

Suppose that three parties, Alice, Bob and Charlie, would like to perform the following computation:
1. Alice and Bob each have one 32-bit integer, `x` and `y`, respectively, which is kept secret.
2. Charlie wants to know if `x` is greater than `y`, but, crucially, Alice and Bob don't trust Charlie, each other, or any other party with their secrets.

This is an instance of a general problem of secure multi-party computation (SMPC), where several parties would like to jointly compute some function (comparison, in the above case) of their inputs in a way that no information about the inputs (other than what can be inferred from the output) leaks to any other party. Currently, CipherCore supports [the ABY3 protocol](https://eprint.iacr.org/2018/403.pdf), which works for three parties and is one of the most efficient available protocols.

First, let us formulate our problem as a CipherCore computation graph:
```Python
import ciphercore as cc

c = cc.create_context()
with c:
    g = c.create_graph()
    with g:
        x = g.input(cc.scalar_type(cc.INT32)) # Alice's input
        y = g.input(cc.scalar_type(cc.INT32)) # Bob's input
        output = x.a2b() > y.a2b() # Charlie's output
        # (`a2b` converts integers into bits, which is necessary for comparisons)
        output.set_as_output()
    g.set_as_main()
print(c)
```

and serialize it by running the above script with Python and redirecting its output to the file `a.json`.

Next, let's compile the computation graph using CipherCore CLI compiler as follows:
```
ciphercore_compile a.json simple 0,1 2 > b.json
```
Here `0,1` means that the inputs belong to party `0` (Alice) and party `1` (Bob), respectively. And `2` means that the output should be revealed to party `2` (Charlie).

The file `b.json` contains a *full description* of the secure protocol for our comparison problem. We can peek into it by running the inspection tool as follows:
```
ciphercore_inspect b.json
```
which outputs various useful statistics, including the number of network rounds that is necessary to perform the computation securely (45, in our case), and the total amount of traffic that needs to be exchanged (233 bytes).

To check that the functionality of the secure protocol stayed intact, let's run it locally. Let's first create a file `inputs.json` with Alice's and Bob's inputs:
```JSON
[
  {"kind": "scalar", "type": "i32", "value": 32},
  {"kind": "scalar", "type": "i32", "value": 12}
]
```

To evaluate the secure protocol on the inputs 32 and 12, we run
```
ciphercore_evaluate b.json inputs.json
```
and correctly get:
```JSON
{"kind": "scalar", "type": "b", "value": 1}
```
since the Alice's number is larger than Bob's.

However, even though the description of the secure protocol for comparison is *fully contained* within `b.json`, this is just the local simulation of the protocol. To execute the secure protocol between actual parties over the network reliably and with high performance, one needs CipherCore runtime, which we provide [upon request](mailto:ciphercore@ciphermode.tech).

The above example is a toy one, however, one can use CipherCore to perform machine learning training and inference and analytics at a large scale. Stay tuned!