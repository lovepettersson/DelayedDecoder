## IMPORTS ##

import networkx as nx
# TODO: use graph-tool instead of networkx...much faster for graphs (and better graphics), but only for Mac/Linux

# See comments in GraphState.graph_vecstate to see why/if cirq is required
# TODO: avoid CIRQ once the vector calculation from Griffiths is implemented

import qecc as q

import matplotlib.pyplot as plt

## import graph-state specific functions
from CodeFunctions.lc_equivalence import check_LCequiv


## CLASSES ##


class GraphState(object):
    r"""
    Class representing a Graph state on :math:`n` qubits.

    :param graph: The graph representing the state, where each node represents a qubit and edges are entangling gates
    :type graph: :class:`nx.Graph`
    """

    def __init__(self, graph):
        # graph is an object from the Graph class of networkx

        # Check that the graph is the correct object
        if not isinstance(graph, nx.Graph):
            raise ValueError("Input graph needs to be a Graph object of NetworkX.")

        self.graph = graph

        # calculates the stabilizer generators of the graph
        self.stab_gens = stabilizer_generators_from_graph(graph)

    def __hash__(self):
        # We need a hash function to store GraphStates as dict keys or in sets.
        return hash(self.graph)

    def __len__(self):
        """
        Yields the number of qubits in the graph.
        """
        return len(self.graph.nodes())

    ## PRINTING ##
    def image(self, with_labels=True, font_weight='bold', position_nodes=None, input_qubits=[]):
        """
        Produces a matplotlib image of the graph associated to the graph state.
        """
        pos_nodes = position_nodes
        if pos_nodes is None:
            # checks if all nodes have attribute "layer", and, if they do, plot the graph using multilayer_layout
            if all(["layer" in this_node[1] for this_node in self.graph.nodes(data=True)]):
                pos_nodes = nx.multipartite_layout(self.graph, subset_key="layer")
            else:
                pos_nodes = nx.spring_layout(self.graph)
        color_map = ['red' if this_node in input_qubits else 'blue' for this_node in self.graph.nodes]
        nx.draw(self.graph, pos_nodes, node_color=color_map, with_labels=with_labels, font_weight=font_weight)


    ##

    ## REPRESENTATION ##

    def as_stabstate(self):
        """
        Converts the graph state into a Stablizer state.
        """
        return StabState(self.stab_gens)


    ## Equivalence under Local Complementation ##

    def is_LC_equiv(self, other, return_all=True):
        r""" Function that checks whether the graph state is locally equivalent to another one,
        If they are, it provides the local operations to be performed to convert one into the other.
        The algorithm runs in O(V^4) (V: #vertices).
        Based on: Van den Nest, Dehaene, De Moor, PHYSICAL REVIEW A 70, 034302 (2004)

        :param other: The second graph.
        :type other: :class:`GraphState`
        :param bool return_all: When True, the algorithm returns all possible Clifford unitaries that provide a
        LC equivalence between the two graphs. When False it only returns the first it finds.
        """
        return check_LCequiv(self.graph, other.graph, return_all=return_all)

    def adj_mat(self):
        r"""
        :return: The adjecency matrix of the graph state
        """
        return nx.adjacency_matrix(self.graph, nodelist=sorted(self.graph.nodes())).todense()


## STABILIZER FUNCTIONS ##

def stabilizer_generators_from_graph(graph):
    r"""
    Calculates the stabilizer generators associated to a graph .

    :param graph: The graph representing the state, where each node represents a qubit and edges are entangling gates
    :type graph: :class:`nx.Graph`
    :returns: An iterator over the list of :math:`[K_1,...,K_n]` stabilizer generators
    """
    stab_gens = []
    nodes = list(graph.nodes())
    nqubits = len(nodes)

    sorted_nodes = list(sorted(graph.nodes()))
    for node_ix in sorted_nodes:
        stab_dict = {sorted_nodes.index(node_ix): 'X'}
        for ngb_node_ix in sorted(graph.neighbors(node_ix)):
            stab_dict[sorted_nodes.index(ngb_node_ix)] = 'Z'
        this_stab = q.Pauli.from_sparse(stab_dict, nq=nqubits)
        stab_gens.append(this_stab)
    return stab_gens


if __name__ == '__main__':
    ## Additional useful functions and classes
    from CodesFunctions.local_transformations import local_cliffords_on_stab_list
    from CodesFunctions.StabStateClass import StabState
    import time
    from CodesFunctions.graphs import gen_linear_graph, gen_ring_graph

    ########### DEFINE GRAPHS TO USE

    #### Linear vs fully connected graphs - 3 qubits
    # nqb = 3
    # G = gen_linear_graph(nqb)
    # G_2 = gen_fullyconnected_graph(nqb)

    #### Linear vs fully connected graphs - 4 qubits
    # nqb = 4
    # G = gen_linear_graph(nqb)
    # G_2 = gen_fullyconnected_graph(nqb)

    #### Star vs fully connected graphs - 4 qubits
    # nqb = 4
    # G = gen_star_graph(nqb)
    # G_2 = gen_fullyconnected_graph(nqb)

    #### Star vs Star with relabeling - 4 qubits
    # nqb = 4
    # G = gen_star_graph(nqb)
    # G_2 = gen_star_graph(nqb, central_qubit=1)

    #### Linear vs standard ring - 4 qubits
    # nqb = 4
    # G = gen_linear_graph(nqb)
    # G_2 = gen_ring_graph(nqb)

    #### Linear vs standard ring + relabeling - 4 qubits
    nqb = 4
    G = gen_linear_graph(nqb)
    temp_G_2 = gen_ring_graph(nqb)
    G_2 = nx.relabel_nodes(temp_G_2, {1: 2, 2: 1})

    #### Star vs Fully connected, large graphs - n qubits
    # nqb = 22
    # G = gen_fullyconnected_graph(nqb)
    # G_2 = gen_star_graph(nqb)

    gstate = GraphState(G)
    gstate_2 = GraphState(G_2)

    ### Obtain the stabilizer generators of the graph G
    print('Stabilizer generators of G:')
    print(gstate.adj_mat())
    print(gstate.stab_gens)
    print()
    print(gstate_2.adj_mat())
    print(gstate_2.stab_gens)

    ### Checks the amount of time to calculate the state vector of the graph state (and prints it, if uncommented)
    start = time.time()
    state_from_cirq = gstate.graph_vecstate()
    end = time.time()
    print('Time required to calculate the state vector:', end - start, 's')
    ## print(np.around(state_from_cirq, 3))

    ### Checks if the two graph states are equivalent under local complementation,
    ### and which local operations are required
    start = time.time()
    check_equiv, unitaries = gstate.is_LC_equiv(gstate_2, return_all=True)
    end = time.time()
    print('Are the two graphs locally equivalent? Which Clifford operators transform them into each other?')
    print(check_equiv)
    print(unitaries)
    print('Time required to check LC equivalence:', end - start, 's')

    ################ CHECK GRAPHS MAP INTO EACH OTHER UNDER LOCAL CLIFFORDS

    if check_equiv:
        loca_Cliff_transf = unitaries[0]
        print()
        print('Test equivalence under transformation:', loca_Cliff_transf)
        starting_gens = gstate.stab_gens
        print('Initial stabilizers:')
        print(starting_gens)
        update_gen = local_cliffords_on_stab_list(loca_Cliff_transf, starting_gens)
        print('Transformed stabilizers:')
        print(update_gen)

        ## Put the generators of the transformed graph state in canonical form
        G, adj_mat, clifford_transf, basis_change_mat = StabState(update_gen).as_graph()
        graphstate_transf = GraphState(G)
        update_gen_canonical = graphstate_transf.stab_gens
        print('Transformed stabilizers in canonical form:')
        print(update_gen_canonical)

        targ_stabs = gstate_2.stab_gens
        print('Found stabilizers match target stabilizers?', update_gen_canonical == targ_stabs)
        print(targ_stabs)

        ### Print the graphs associated to the states
        plt.subplot(221)
        gstate.image(with_labels=True)
        plt.subplot(222)
        gstate_2.image(with_labels=True)
        plt.subplot(224)
        graphstate_transf.image(with_labels=True)
        plt.show()

################ OTHER TESTS

# nqb = 5
# G = gen_linear_graph(nqb)
# # G = gen_ring_graph(nqb)
#
# gstate = GraphState(G)
# print(gstate.graph_vecstate())
# print(gstate.stab_gens)
#
# gstate.image(with_labels=True)
# plt.show()
