import numpy as np
from itertools import chain, combinations, product
from CodeFunctions.linear_algebra_inZ2 import find_kernel_basis_inZ2
import networkx as nx

#### Binary forms of single-qubit Clifford, using the (Z|X) notation.
symClifford_to_gate = {(1, 0, 0, 1): 'I', (0, 1, 1, 0): 'H', (1, 1, 0, 1): 'S',
                       (0, 1, 1, 1): 'HS', (1, 1, 1, 0): 'SH', (1, 0, 1, 1): 'HSH'}

#### Binary forms of single-qubit Clifford, using the (X|Z) notation.
# symClifford_to_gate = {(1, 0, 0, 1): 'I', (0, 1, 1, 0): 'H', (1, 0, 1, 1): 'S',
#                        (1, 1, 1, 0): 'HS', (0, 1, 1, 1): 'SH', (1, 1, 0, 1): 'HSH'}


##### MISCELANNEA FUNCTIONS

def powerset_noempty(iterable):
    """Calculates the powerset of an iterable, excluding the empty set"""
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


##### FUNCTIONS TO SET UP THE LINEAR SYSTEM OF EQUATIONS FOR LC EQUIVALENCE

def get_VanDenNest_rows(j, k, adj1, adj2, nnodes):
    """ Obtains the rows for the  VanDenNest matrix, used in the get_VanDenNest_matrix function.
    see Van den Nest, Dehaene, De Moor, PHYSICAL REVIEW A 70, 034302 (2004),
    and Hein, M. et al. arXiv:0602096 [quant-ph] (2006).
    """
    this_row = np.zeros(4 * nnodes)
    # set values relative to (a)s
    this_row[0 * nnodes + k] = adj1[j, k]
    # set values relative to (b)s
    if j == k:
        this_row[1 * nnodes + k] = 1
    # set values relative to (c)s
    for ix in range(nnodes):
        this_row[2 * nnodes + ix] = adj1[j, ix] * adj2[ix, k]
    # set values relative to (d)s
    this_row[3 * nnodes + j] = adj2[j, k]

    return this_row


def get_VanDenNest_matrix(adj1, adj2):
    """ Obtains that describes the system of linear equations Γ′BΓ + DΓ + Γ′A + C = 0
    see Van den Nest, Dehaene, De Moor, PHYSICAL REVIEW A 70, 034302 (2004),
    and Hein, M. et al. arXiv:0602096 [quant-ph] (2006).
    """
    if not (len(adj1) == len(adj2)):
        raise ValueError("Graphs need to have same number of nodes.")

    nqubits = len(adj1)
    return np.array(
        [get_VanDenNest_rows(j, k, adj1, adj2, nqubits) for j, k in product(range(nqubits), range(nqubits))], dtype=int)


def check_symplectic_constraint_single(v, nqubits):
    """ Function to be used in check_VanDenNest_constraint
    see Van den Nest, Dehaene, De Moor, PHYSICAL REVIEW A 70, 034302 (2004)
    """
    v_shaped = np.reshape(v, (4, nqubits))
    return (np.multiply(v_shaped[0], v_shaped[3]) + np.multiply(v_shaped[1], v_shaped[2])) % 2


def check_symplectic_constraint(basis, nqubits, return_all=False):
    """ Checks if a combination of the basis vectors satisfies the conditions required for LC equivalence.
    see Van den Nest, Dehaene, De Moor, PHYSICAL REVIEW A 70, 034302 (2004)
    """
    if len(basis) > 4:
        # print('dim>4, doing Test 1')
        test_basis = [(sol1 + sol2) % 2 for sol1, sol2 in product(basis, basis)]
    else:
        # print('dim<=4, doing Test 2')
        test_basis = list(map(lambda x: sum(x) % 2, powerset_noempty(basis)))

    check_costraints = [check_symplectic_constraint_single(this_v, nqubits) for this_v in test_basis]

    if return_all:
        valid_idx = [idx for idx, this_solution in enumerate(check_costraints) if np.all(this_solution == 1.)]
        valid_solutions = [test_basis[idx] for idx in valid_idx]
        if len(valid_solutions) > 0:
            # removes duplicates
            valid_solutions_noduplicates = list({array.tostring(): array for array in valid_solutions}.values())

            return True, [get_Clifford_unitary(this_sol, nqubits) for this_sol in valid_solutions_noduplicates]
        else:
            return False, []
    else:
        for idx, this_solution in enumerate(check_costraints):
            if np.all(this_solution == 1.):
                return True, get_Clifford_unitary(test_basis[idx], nqubits)
        return False, []


##### FUNCTIONS TO CONVERT BINARY <---> UNITARY REPRESENTATIONS OF CLIFFORD OPERATIONS
def get_Clifford_unitary(cliff_binvect, nqbs):
    """ Returns the single-qubit Clifford operations on n qubits from the binary representation
     """
    cliff_binvect_onqubits = list(map(tuple, np.reshape(cliff_binvect, (4, nqbs)).T))
    cliff_ops = [symClifford_to_gate[this_binvec] for this_binvec in cliff_binvect_onqubits]
    return cliff_ops


##### FINAL FUNCTION TO CHECK LC EQUIVALENCE OF TWO GRAPHS

def check_LCequiv(graph1, graph2, return_all=True):
    r""" Function that checks whether the graphs graph1 and graph2 are locally equivalent and, if they are equivalent,
    provides the local operations to be performed to convert one into the other.
    The algorithm runs in O(V^4) (V: #vertices).
    Based on: Van den Nest, Dehaene, De Moor, PHYSICAL REVIEW A 70, 034302 (2004)

    :param graph1: The first graph.
    :type graph1: :class:`nx.Graph`
    :param graph2: The second graph.
    :type graph2: :class:`nx.Graph`
    :param bool return_all: When True, the algorithm returns all possible Clifford unitaries that provide a
    LC equivalence between the two graphs. When False it only returns the first it finds.
    """

    # get the number of qubits for each graph
    nqb1 = graph1.number_of_nodes()
    nqb2 = graph2.number_of_nodes()

    if nqb1 != nqb2:
        return False, []  # If the graphs have different number of nodes they are not equivalent
    else:
        nqbs = nqb1

    # get adjacency matrices
    adj1 = nx.convert_matrix.to_numpy_matrix(graph1, nodelist=sorted(graph1.nodes()), dtype=int)
    adj2 = nx.convert_matrix.to_numpy_matrix(graph2, nodelist=sorted(graph2.nodes()), dtype=int)

    # Here starts the algorithm
    # initialise linear system of binary equations
    getMat = get_VanDenNest_matrix(adj1, adj2)
    # solve equations in the F2 field
    solution_basis = find_kernel_basis_inZ2(getMat)
    # check which solutions provide symplectic matrices, and calculate the associated unitaries.
    return check_symplectic_constraint(solution_basis, nqbs, return_all=return_all)


if __name__ == '__main__':
    ########### DEFINE GRAPHS TO COMPARE

    #### STAR vs FULLY CONNECTED
    nqb = 4
    G1 = nx.Graph()
    G1.add_nodes_from([0, 1, 2, 3])
    G1.add_edges_from([(0, 1), (0, 2), (0, 3)])
    G2 = nx.Graph()
    G2.add_nodes_from([0, 1, 2, 3])
    G2.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])

    ########### PERFORM ALGORITHM

    check_equiv, results = check_LCequiv(G1, G2, return_all=True)

    print(check_equiv)
    print(results)

    import matplotlib.pyplot as plt

    plt.subplot(211)
    nx.draw(G1, with_labels=True)
    plt.subplot(212)
    nx.draw(G2, with_labels=True)
    plt.show()
