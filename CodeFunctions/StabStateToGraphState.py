import numpy as np
from copy import deepcopy
import random

##### GAUSSIAN ELIMINATION ON BINARY MATRICES WITH SUM MODULO 2 ###############

# gauss elimination of rows
def gauss(A):
    output =deepcopy(A)
    (m ,n) = np.shape(A)
    if m == 0 or n == 0:
        raise NameError('empty matrix')
        output = np.zeros((m, n))
    row = -1
    for j in range(n):
        pivot = -1
        for i in range(row + 1, m):
            if output[i][j]:
                if pivot == -1:
                    pivot = i
                    row += 1
                    output[[row, pivot]] = output[[pivot, row]]
                else:
                    output[i] = np.logical_xor(output[i], output[row])
    return output


# rank of a matrix A. A must be in gauss form
def rank(A):
    m, n = np.shape(A)
    rank = 0
    for i in range(m):
        for j in range(i, n):
            if A[i][j]:
                rank += 1
                break
    return rank


# inverse of a matrix
def inverse(A):
    (m, n) = np.shape(A)
    if m != n:
        raise NameError('a not square matrix does not have inverse')
        output = np.empty((m, 0), bool)
    A = np.block([[A, np.eye(m, dtype=bool)]])
    A = gauss(A)
    if rank(A[:, :m]) < m:
        raise Exception('singular matrix')
    A = np.array([A[-i - 1] for i in range(m)]).T
    A = np.array([A[m - i - 1] for i in range(m)] + [A[-i - 1] for i in range(m)]).T
    output = gauss(A)[:, m:]
    output = np.array([[output[-i - 1, -j - 1] for j in range(m)] for i in range(m)])
    return output


def convert(stabs, control=None, target=None, shuffle=False):
    """
    Returns a graph state that is local Clifford unitary equivalent to the given stabilizer state.

    :param stabs: list of N strings ['XYYXZ...','IIXXZ...',...,'stabilizerN'] defining the stabilizer operators
                  'stabilizer': string ['IXIYZ...'] of N elements from the set of Pauli operators 'I', 'X', 'Y', 'Z'
    :param control: list of control qubits [0,43,2,1,...] labelled from 0 to N-1
    :param target: list of target qubits [3,23,42,4...] labelled from 0 to N-1
    :param shuffle: True for obtaining a random output from the set of valid outputs.

    G, c, t, z, R = convert(s, control=None, target=None, shuffle=False)

    :return G: NxN numpy array composed only by 0s and 1s, that represents the adjacency matrix of the resulting graph
    :return c: final list of control qubits labelled from 0 to N-1. Contains the control qubits given in the input
               `control` and other control qubits assigned by the function.
    :return t: final list of target qubits labelled from 0 to N-1. Contains the target qubits given in the input
               `control` and other target qubits assigned by the function.
    :return z: list of control qubits where a pi/2 z-rotation is applied. It is a subset of the output c
    :return R: NxN numpy array composed only by 0s and 1s, that represents the recombinations of the stabilizers
               performed to obtain the stabilizers of the final graph state.
    """

    if control is None:
        control = []
    if target is None:
        target = []
    # number of qubits N and number of stabilizers Ns
    N = len(stabs[0])
    Ns = len(stabs)
    # binary representation
    A = np.array([[stabs[j][i] in {'Y', 'Z'} for j in range(N)] for i in range(Ns)] +
                 [[stabs[j][i] in {'Y', 'X'} for j in range(N)] for i in range(Ns)], dtype=int)
    # raise Exception if there are not enough stabilizers
    if Ns < N:
        raise Exception('The number of stabilizers in S can not be smaller than the number of qubits')
    # raise Exception if rank(p) is not N
    if rank(gauss(A)) != N:
        raise Exception('S must contain the same number of independent stabilizers than qubits')
    # raise Exception if stabilizers do not commute
    for i in range(Ns-1):
        for j in range(i+1, Ns):
            if A[:N, i].dot(A[N:, j]) % 2 != A[:N, j].dot(A[N:, i]) % 2:
                raise Exception('generators ' + stabs[i] + ' and ' + stabs[j] + ' do not commute')
    # raise Exception if control and target qubits have non empty intersection
    if set(control).intersection(set(target)):
        raise Exception('c and t must have empty intersection')
    # if a control or target qubit has a label bigger than N
    for i in control:
        if i >= N:
            raise Exception('control qubits must be labelled from 0 to N-1')
    for i in target:
        if i >= N:
            raise Exception('target qubits must be labelled from 0 to N-1')
    # shuffle rows
    if shuffle:
        remaining = set(range(N))-set(control).union(set(target))
        qubits = control + random.sample(sorted(remaining), N-len(control)-len(target)) + target
    else:
        remaining = set(range(N)) - set(control).union(set(target))
        qubits = control + list(remaining) + target
    # reorder rows in A
    A = np.array([A[qubits[i]] for i in range(N)] + [A[qubits[i]+N] for i in range(N)])
    # put A in the form that allows to make Gauss elimination in the right way add the identity to monitor the
    # recombinations performed by the Gaussian elimination
    A = np.array(list(A[N:, :])+list(A[:N, :])+list(np.eye(N, dtype=int))).T
    # perform the Gaussian elimination and identify the matrix R that performs that recombination
    A = gauss(A)
    R = A[:, -N:].T
    A = A[:, :-N]
    n = rank(A[:, :N])
    if len(control) > n:
        raise Exception('too many control qubits selected')
    if len(target) > N - n:
        raise Exception('too many target qubits selected')
    if rank(A[:n, :len(control)]) < len(control):
        raise Exception('wrong selection of control qubits')
    # select control and target qubits
    for i in range(len(control), n):
        for j in range(i, N):
            if A[i, j] and qubits[j] not in target and qubits[j] not in control:
                control.append(qubits[j])
                break
            elif qubits[j] not in target and qubits[j] not in control:
                target.append(qubits[j])
    target = target + list(set(range(N))-set(control).union(set(target)))  # add missing qubits
    if len(control) != n:
        raise Exception('wrong selection of control and/or target qubits')
    # put A back to the original form
    A = A.T
    A = np.array(list(A[N:, :])+list(A[:N, :]))
    A = np.array([A[qubits.index(i)] for i in control] + [A[qubits.index(i)] for i in target] +
                 [A[qubits.index(i)+N] for i in control] + [A[qubits.index(i)+N] for i in target])
    # update the order of the qubits in the rows
    qubits = control+target
    # build xlc and the inverse
    xlc = A[N:n+N, :n]
    if n != 0:
        xlc_inv = inverse(xlc)
    else:
        xlc_inv = np.zeros((0, 0), dtype=int)
    # identify the rest of blocks
    xlt = A[n+N:, :n]
    zlc = A[:n, :n]
    zlt = A[n:N, :n]
    zrt = A[n:N, n:]
    # compute r
    if n != N:
        zrt_inv = inverse(zrt)
    else:
        zrt_inv = np.zeros((0, 0), dtype=int)
    R = R.dot(np.block([[xlc_inv, np.zeros((n, N-n), dtype=int)], [zrt_inv.dot(zlt).dot(xlc_inv) % 2, zrt_inv]])) % 2
    # compute C and B and obtain the list z of the qubits where the z-rotation is performed
    B = xlt.dot(xlc_inv) % 2
    C = (zlc+B.T.dot(zlt)).dot(xlc_inv) % 2
    z = [qubits[i] for i in range(n) if C[i, i]]
    C = (C + np.diag(np.diagonal(C))) % 2
    # Adjacency matrix
    G = np.block([[C, B.T], [B, np.zeros((N-n, N-n), dtype=int)]])
    G = np.array([G[qubits.index(i)] for i in range(N)])
    G = np.array([G[:, qubits.index(i)] for i in range(N)]).T
    return G, sorted(control), sorted(target), sorted(z), R

if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt
    from StabilizerStateClass import *

    # path = r"C:\Users\Admin\Desktop\HeraldedResourceStateGen\codetableHmatrices\5_qbt_H_matrix_1_logical.txt"
    # dim_row = 4
    # dim_col = 5 * 2
    # num_logicals = 1

    # path = r"C:\Users\Admin\Desktop\HeraldedResourceStateGen\codetableHmatrices\8_qbt_H_matrix_3_logicals.txt"
    # dim_row = 5
    # dim_col = 8 * 2
    # num_logicals = 3

    # path = r"C:\Users\Admin\Desktop\HeraldedResourceStateGen\codetableHmatrices\20_qbt_H_matrix_10_logicals.txt"
    # dim_row = 10
    # dim_col = 20 * 2
    # num_logicals = 10

    path = r"C:\Users\Admin\Desktop\HeraldedResourceStateGen\codetableHmatrices\28_qbt_H_matrix_10_logicals.txt"
    dim_row = 18
    dim_col = 28 * 2
    num_logicals = 10

    # parse_H_matrix(path, dim_row, dim_col)
    Parser = CodetableCodeParsing(path, dim_row, dim_col, num_logicals)
    stabilizers = Parser.parse_stabilizers_to_SG_converter()
    for logical in stabilizers[dim_row:]:
        print(logical)
    print("")
    G, control, target, z, R = convert(stabilizers)
    print(G)
    print(z)
    print()
    print(control)
    print(target)
    print(control + target)
    graph = nx.from_numpy_matrix(G)
    nx.draw(graph, with_labels=True)
    plt.show()

    edge_list = list(graph.edges())
    for edge in edge_list:
        u, v = edge
        print(str(u+1) + "," + str(v+1) + ";")


    stab_gens = []
    nodes = list(graph.nodes())
    nqubits = len(nodes)

    import qecc as q
    sorted_nodes = list(sorted(graph.nodes()))
    for node_ix in sorted_nodes:
        stab_dict = {sorted_nodes.index(node_ix): 'X'}
        for ngb_node_ix in sorted(graph.neighbors(node_ix)):
            stab_dict[sorted_nodes.index(ngb_node_ix)] = 'Z'
        this_stab = q.Pauli.from_sparse(stab_dict, nq=nqubits)
        stab_gens.append(this_stab)
    for stab in stab_gens:
        print(stab)