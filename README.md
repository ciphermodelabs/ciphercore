# What is CipherCore?

CipherCore is an open-source general purpose library for processing encrypted data. It’s a state-of-the-art platform for building customized applications that can run directly over encrypted data without decrypting it first. CipherCore can be used to run tasks on multiple distributed datasets owned by multiple organizations within the same enterprise or even different enterprises without disclosing the data to other parties. The library is based on a technology called secure computation.

# What is Secure Computation?

Secure Multip-Party Computation (SMPC) is a cutting-edge subfield of cryptography that provides various types of protocols allowing the execution of certain programs over encrypted data ([read more](https://en.wikipedia.org/wiki/Secure_multi-party_computation)). SMPC protocols take as input a restricted form of computation called [circuit representation](https://en.wikipedia.org/wiki/Boolean_circuit). Translating high-level programs into circuit representation is a complicated, error-prone and time-consuming process. CipherCore Transpiler drastically simplifies the process by automatically translating and compiling high-level programs directly into the SMPC protocols, thus, allowing any software developer to use secure computation without requiring any knowledge of cryptography.

# CipherCore and Intermediate Representation

CipherCore’s ease of use is due to introducing a new intermediate representation layer between the application layer and the protocol layer. Applications are mapped to a computation graph first and then to a (or set of) SMPC protocols. This architecture allows for rapid integration of various SMPC protocols as new cryptographic backends.

![CipherCore architecture](https://i.imgur.com/yBm69N1.png)


# Stay tuned for the release!

Meanwhile, contact us at [info@ciphermode.com](mailto:info@ciphermode.com).
