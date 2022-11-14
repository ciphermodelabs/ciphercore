import argparse
import numpy as np
import itertools

def matrix_to_bytes(mat):
    ''' Convert a given matrix to the list of bytes
    '''
    bytes_list = []
    bit_counter = 0
    byte = 0
    for bit in np.nditer(mat):
        byte += bit << bit_counter
        if bit_counter == 7:
            bytes_list.append(byte)
            byte = 0
        bit_counter = (bit_counter + 1) % 8
    if bit_counter > 0:
        bytes_list.append(byte)

    return bytes(bytes_list)

def main():
    ''' Use the global parameters `blocksize`, `keysize` and `rounds`
        to create the set of matrices and constants for the corresponding
        LowMC instance.
    '''

    program_description = 'Create a set of matrices and constants for the corresponding LowMC instance. Save those in files named `linear_layer_matrices.dat`, `key_matrices.dat`, `round_constant.dat`.'
    parser = argparse.ArgumentParser(description=program_description)
    parser.add_argument('blocksize', type=int, help='specifies block size in bits')
    parser.add_argument('keysize', type=int, help='specifies key size in bits')
    parser.add_argument('rounds', type=int, help='specifies the number of rounds')

    arguments = parser.parse_args()
    blocksize = arguments.blocksize
    keysize = arguments.keysize
    rounds = arguments.rounds

    gen = grain_ssg()
    linlayers = np.array([instantiate_matrix(blocksize, blocksize, gen) for _ in range(rounds)])
    round_constants = np.array([[next(gen) for _ in range(blocksize)] for _ in range(rounds)])
    roundkey_matrices = np.array([instantiate_matrix(blocksize, keysize, gen) for _ in range(rounds + 1)])

    with open('linear_layer_matrices.dat', 'wb') as linfile:
        linfile.write(matrix_to_bytes(linlayers))

    with open('key_matrices.dat', 'wb') as keyfile:
        keyfile.write(matrix_to_bytes(roundkey_matrices))

    with open('round_constants.dat', 'wb') as constfile:
        constfile.write(matrix_to_bytes(round_constants))

    with open('matrices_and_constants.dat', 'w') as matfile:
        s = 'LowMC matrices and constants\n'\
            '============================\n'\
            'Block size: ' + str(blocksize) + '\n'\
            'Key size: ' + str(keysize) + '\n'\
            'Rounds: ' + str(rounds) + '\n\n'
        matfile.write(s)
        s = 'Linear layer matrices\n'\
            '---------------------'
        matfile.write(s)
        for r in range(rounds):
            s = '\nLinear layer ' + str(r + 1) + ':\n'
            for row in linlayers[r]:
                s += str(row) + '\n'
            matfile.write(s)

        s = '\nRound constants\n'\
              '---------------------'
        matfile.write(s)
        for r in range(rounds):
            s = '\nRound constant ' + str(r + 1) + ':\n'
            s += str(round_constants[r]) + '\n'
            matfile.write(s)

        s = '\nRound key matrices\n'\
              '---------------------'
        matfile.write(s)
        for r in range(rounds + 1):
            s = '\nRound key matrix ' + str(r) + ':\n'
            for row in roundkey_matrices[r]:
                s += str(row) + '\n'
            matfile.write(s)

def instantiate_matrix(n, m, gen):
    ''' Instantiate a matrix of maximal rank using bits from the
        generatator `gen`.
    '''
    while True:
        mat  = np.fromiter(itertools.islice(gen, n*m), dtype=int).reshape((n,m))
        if np.linalg.matrix_rank(mat) >= min(n, m):
            return mat


def grain_ssg():
    ''' A generator for using the Grain LSFR in a self-shrinking generator. '''
    state = [1 for _ in range(80)]
    index = 0
    # Discard first 160 bits
    for _ in range(160):
        state[index] ^= state[(index + 13) % 80] ^ state[(index + 23) % 80]\
                        ^ state[(index + 38) % 80] ^ state[(index + 51) % 80]\
                        ^ state[(index + 62) % 80]
        index += 1
        index %= 80
    choice = False
    while True:
        state[index] ^= state[(index + 13) % 80] ^ state[(index + 23) % 80]\
                        ^ state[(index + 38) % 80] ^ state[(index + 51) % 80]\
                        ^ state[(index + 62) % 80]
        choice = state[index]
        index += 1
        index %= 80
        state[index] ^= state[(index + 13) % 80] ^ state[(index + 23) % 80]\
                        ^ state[(index + 38) % 80] ^ state[(index + 51) % 80]\
                        ^ state[(index + 62) % 80]
        if choice == 1:
            yield state[index]
        index += 1
        index %= 80


if __name__ == '__main__':
    main()
