import numpy as np


def row_echelon_inZ2(m):
    """ Return Reduced Row Echelon Form of matrix A in the field F_2={0,1}.
    Based on the pseudocode from https://en.wikipedia.org/wiki/Row_echelon_form.
    Returns REF_m, the Reduced Row Echelon Form of the input m, and the transformation matrix inverse_mat such that
    inverse_mat @ m = REF_m  (mod 2), which, if m is invertible (i.e. REF_m is Id), represents the left-inverse of m.

    :param m: The matrix to be put in Reduced Row Echelon Form
    :type m: :class:`np.matrix`
    """
    nr, nc = m.shape
    inverse_mat = np.eye(nr, dtype=np.int)

    REF_m = m.copy()

    lead = 0
    for r in range(nr):
        if nc <= lead:
            return REF_m, inverse_mat
        i = r
        while REF_m[i, lead] == 0:
            i = i+1
            if nr == i:
                i = r
                lead = lead + 1
                if nc == lead:
                    return REF_m, inverse_mat
        if i != r:
            # swap rows i and r of the matrix (REF_m), and the i and r rows of transf_mat
            REF_m[[i, r]] = REF_m[[r, i]]
            inverse_mat[[i, r]] = inverse_mat[[r, i]]

        for i in range(nr):
            if i != r:
                # subtract row r to row i of the matrix (REF_m), and the i and r rows of transf_mat
                if REF_m[i, lead] != 0:
                    REF_m[i] = (REF_m[i] - REF_m[r]) % 2
                    inverse_mat[i] = (inverse_mat[i] - inverse_mat[r]) % 2
        lead = lead + 1
    return REF_m, inverse_mat


def find_kernel_basis_inZ2(m):
    """ A basis for the Kernel of matrix M is found via row_echelon in the field F_2={0,1}
    see e.g. https://en.wikipedia.org/wiki/Kernel_(linear_algebra)
    """
    n_rows0, n_cols0 = np.shape(m)
    expanded_M = np.block([[m], [np.eye(n_cols0)]])
    u, _ = row_echelon_inZ2(expanded_M.T)
    kern_basis = [this_row[n_rows0:] for this_row in u if np.all(this_row[:n_rows0] == 0)]
    return np.array(kern_basis)


if __name__ == '__main__':
    import networkx as nx

    nqb = 4
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (1, 3), (0, 2)])

    mymat = nx.adjacency_matrix(G).todense()

    mymat = np.bmat([[mymat], [np.eye(nqb, dtype = np.int)]])
    print(mymat)

    test_mat, inv_mat = row_echelon_inZ2(mymat)
    print(test_mat)
    # print(inv_mat)
    # print((test_mat - inv_mat.dot(mymat))%2)
