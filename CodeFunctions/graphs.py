import networkx as nx
# TODO: use graph-tool instead of networkx...much faster for graphs (and better graphics), but only for Mac/Linux

import numpy as np
from itertools import product, combinations

# from CodeFunctions.GraphStateClass import GraphState

# quick graph state definition
def graph_from_nodes_and_edges(graph_nodes, graph_edges):
    graph = nx.Graph()
    graph.add_nodes_from(graph_nodes)
    graph.add_edges_from(graph_edges)
    return graph


# def graphstate_from_nodes_and_edges(graph_nodes, graph_edges):
#     return GraphState(graph_from_nodes_and_edges(graph_nodes, graph_edges))

## GRAPH INITIALIZATION FUNCTIONS ##

def gen_empty_graph(nqubits):
    graph = nx.Graph()
    graph.add_nodes_from(range(nqubits))
    return graph

def gen_linear_graph(nqubits):
    graph = nx.Graph()
    graph.add_nodes_from(range(nqubits))
    these_edges = [(node_ix, node_ix + 1) for node_ix in range(nqubits - 1)]
    graph.add_edges_from(these_edges)
    return graph


def gen_ring_graph(nqubits):
    graph = nx.Graph()
    graph.add_nodes_from(range(nqubits))
    these_edges = [(node_ix, (node_ix + 1) % nqubits) for node_ix in range(nqubits)]
    graph.add_edges_from(these_edges)
    return graph


def gen_star_graph(nqubits, central_qubit=0):
    graph = nx.Graph()
    nodes = range(nqubits)
    graph.add_nodes_from(nodes)
    graph.add_edges_from(
        product([central_qubit], [other_nodes for other_nodes in nodes if other_nodes != central_qubit]))
    return graph


def gen_fullyconnected_graph(nqubits):
    graph = nx.Graph()
    nodes = range(nqubits)
    graph.add_nodes_from(nodes)
    graph.add_edges_from(combinations(nodes, 2))
    return graph


def gen_crazy_graph(nrows, nlayers):
    graph = nx.Graph()
    nodes_mat = np.arange(nrows * nlayers).reshape((nlayers, nrows))
    for layer_ix in range(nlayers):
        for row_ix in range(nrows):
            graph.add_node(layer_ix * nrows + row_ix, layer=layer_ix)
    for layer_ix in range(nlayers - 1):
        these_edges = product(nodes_mat[layer_ix], nodes_mat[layer_ix + 1])
        graph.add_edges_from(these_edges)
    return graph


def gen_multiwire_graph(nrows, nlayers):
    graph = nx.Graph()
    nodes_mat = np.arange(nrows * nlayers).reshape((nlayers, nrows))
    for layer_ix in range(nlayers):
        for row_ix in range(nrows):
            graph.add_node(layer_ix * nrows + row_ix, layer=layer_ix)
    for layer_ix in range(nlayers - 1):
        these_edges = zip(nodes_mat[layer_ix], nodes_mat[layer_ix + 1])
        graph.add_edges_from(these_edges)
    return graph


def gen_square_lattice_graph(nrows, nlayers):
    graph = nx.Graph()
    nodes_mat = np.arange(nrows * nlayers).reshape((nlayers, nrows))
    for layer_ix in range(nlayers):
        for row_ix in range(nrows):
            graph.add_node(layer_ix * nrows + row_ix, layer=layer_ix)
    for layer_ix in range(nlayers - 1):
        # Horizontal edges
        these_edges = list(zip(nodes_mat[layer_ix], nodes_mat[layer_ix + 1]))
        graph.add_edges_from(these_edges)
    for layer_ix in range(nlayers):
        # Vertical edges
        these_edges = [tuple([nodes_mat[layer_ix, row_ix], nodes_mat[layer_ix, row_ix + 1]])
                       for row_ix in range(nrows - 1)]
        graph.add_edges_from(these_edges)
    return graph


def gen_triangular_lattice_graph(nrows, nlayers):
    graph = nx.Graph()
    nodes_mat = np.arange(nrows * nlayers).reshape((nlayers, nrows))
    for layer_ix in range(nlayers):
        for row_ix in range(nrows):
            graph.add_node(layer_ix * nrows + row_ix, layer=layer_ix)
    for layer_ix in range(nlayers - 1):
        # Horizontal edges
        these_edges = list(zip(nodes_mat[layer_ix], nodes_mat[layer_ix + 1]))
        graph.add_edges_from(these_edges)
    for layer_ix in range(nlayers):
        # Vertical edges
        these_edges = [tuple([nodes_mat[layer_ix, row_ix], nodes_mat[layer_ix, row_ix + 1]])
                       for row_ix in range(nrows - 1)]
        graph.add_edges_from(these_edges)
    for layer_ix in range(nlayers - 1):
        # transversal edges
        if (layer_ix % 2) == 0:
            these_edges = [tuple([nodes_mat[layer_ix, row_ix], nodes_mat[layer_ix + 1, row_ix - 1]])
                           for row_ix in range(1, nrows)]
        else:
            these_edges = [tuple([nodes_mat[layer_ix, row_ix], nodes_mat[layer_ix + 1, row_ix + 1]])
                           for row_ix in range(nrows - 1)]
        graph.add_edges_from(these_edges)
    return graph


def gen_hexagonal_lattice_graph(nrows, nlayers):
    graph = nx.Graph()
    nodes_mat = np.arange(nrows * nlayers).reshape((nlayers, nrows))
    for layer_ix in range(nlayers):
        for row_ix in range(nrows):
            graph.add_node(layer_ix * nrows + row_ix, layer=layer_ix)
    for layer_ix in range(nlayers - 1):
        # Horizontal edges
        these_edges = list(zip(nodes_mat[layer_ix], nodes_mat[layer_ix + 1]))
        graph.add_edges_from(these_edges)
    for layer_ix in range(nlayers):
        # Vertical edges
        if (layer_ix % 2) == 0:
            these_edges = [tuple([nodes_mat[layer_ix, 2 * row_ix], nodes_mat[layer_ix, 2 * row_ix + 1]])
                           for row_ix in range(int(nrows / 2))]
        else:
            these_edges = [tuple([nodes_mat[layer_ix, 2 * row_ix + 1], nodes_mat[layer_ix, 2 * row_ix + 2]])
                           for row_ix in range(int((nrows - 1) / 2))]
        graph.add_edges_from(these_edges)
    return graph


def gen_tree_graph(branching_list):
    graph = nx.Graph()
    temp_num_qubits = 0
    tot_qubit_num = temp_num_qubits
    old_qubits = []
    for layer_num in range(len(branching_list) + 1):
        temp_num_qubits = temp_num_qubits + int(np.prod(branching_list[:layer_num]))
        new_qubits = list(range(tot_qubit_num, temp_num_qubits))
        graph.add_nodes_from(range(tot_qubit_num, temp_num_qubits))
        if layer_num > 0:
            for old_qubit_ix, old_qubit in enumerate(old_qubits):
                these_edges = [(old_qubit, node_ix) for node_ix in
                               new_qubits[old_qubit_ix * branching_list[layer_num - 1]:
                                          (old_qubit_ix + 1) * branching_list[layer_num - 1]]]
                graph.add_edges_from(these_edges)
        old_qubits = new_qubits
        tot_qubit_num = temp_num_qubits
    return graph


def gen_random_graph(n, p):
    # generates Erdős–Rényi random with n nodes and probability p for each edge to exist
    return nx.fast_gnp_random_graph(n, p)


def gen_random_connected_graph(n):
    # generates Erdős–Rényi random with n nodes which is, with high probability, connected.
    # The condition for connectedness with high probability is p>ln(n)/n
    p_randomgraph = np.random.uniform(np.log(n)/n, 1)
    temp_graph = nx.fast_gnp_random_graph(n, p_randomgraph)
    while not nx.is_connected(temp_graph):
        p_randomgraph = np.random.uniform(np.log(n) / n, 1)
        temp_graph = nx.fast_gnp_random_graph(n, p_randomgraph)
    return temp_graph


def gen_random_connected_fixed_numb_edges_graph(n):
    # generates Erdős–Rényi random with n nodes which is, with high probability, connected.
    # The condition for connectedness with high probability is p>ln(n)/n
    loop_flag = False
    output_graph = 0
    while loop_flag is False:
        p_randomgraph = np.random.uniform(np.log(n)/n, 1)
        temp_graph = nx.fast_gnp_random_graph(n, p_randomgraph)
        while not nx.is_connected(temp_graph):
            p_randomgraph = np.random.uniform(np.log(n) / n, 1)
            temp_graph = nx.fast_gnp_random_graph(n, p_randomgraph)
        edges = temp_graph.number_of_edges()
        if edges >= 2 * n and edges <= 2.5 * n:
            output_graph = temp_graph
            loop_flag = True
    return output_graph


def gen_random_disconnected_graph(n):
    # generates Erdős–Rényi random with n nodes which is, with high probability, connected.
    # The condition for connectedness with high probability is p>ln(n)/n
    G2 = nx.Graph()
    G2.add_nodes_from([7, 8])
    p_randomgraph = np.random.uniform(np.log(n)/n, 1)
    G1 = nx.fast_gnp_random_graph(n, p_randomgraph)
    G = nx.compose(G1, G2)
    return G


if __name__ == '__main__':
    graph = gen_random_connected_fixed_numb_edges_graph(10)
    print(graph.number_of_edges())
