import stim
import numpy as np

class CodetableCodeParsing(object):
    r"""
    """

    def __init__(self, path, num_row, num_col, num_logical_qbts, printing=False):
        # Initialize the decoder
        self.path = path
        self.num_row = num_row
        self.num_col = num_col
        self.num_logical_qbt = num_logical_qbts
        self.printing = printing

        self.H_matrix = self.parse_H_matrix()

        self.pauli_stabilizers = []
        self.parse_H_matrix_to_stim()
        self.logicals = self.get_logicals()
        self.logicals_in_Z_2 = self.parse_logicals_to_Z_2()
        if self.printing:
            for logical in self.logicals_in_Z_2:
                print("logical: ", logical)
        self.H_X, self.H_Z = self.combine_logicals_and_stab_gens()



    def binary_to_str(self, stab):
        n_qbts = int(self.num_col / 2)
        pauli = ""
        for idx in range(n_qbts):
            if stab[idx] == 1 and stab[idx + n_qbts] == 0:
                pauli += "X"
            elif stab[idx] == 0 and stab[idx + n_qbts] == 1:
                pauli += "Z"
            elif stab[idx] == 1 and stab[idx + n_qbts] == 1:
                pauli += "Y"
            else:
                pauli += "I"
        return pauli

    def parse_H_matrix(self):
        H_matrix = np.zeros((self.num_row, self.num_col), dtype=int)
        idx_col_cnt = 0
        idx_row_cnt = 0
        f = open(self.path, "r")
        for word in f.read().split():
            if len(word) == 2 and idx_col_cnt == 0:
                pauli = int(word[-1])
                H_matrix[idx_row_cnt][idx_col_cnt] = pauli
                idx_col_cnt += 1
            elif len(word) == 3:
                pauli = int(word[0])
                H_matrix[idx_row_cnt][idx_col_cnt] = pauli
                idx_col_cnt += 1
                pauli = int(word[-1])
                H_matrix[idx_row_cnt][idx_col_cnt] = pauli
                idx_col_cnt += 1
            elif len(word) == 1:
                pauli = int(word)
                H_matrix[idx_row_cnt][idx_col_cnt] = pauli
                idx_col_cnt += 1
            elif len(word) == 2 and idx_col_cnt == self.num_col - 1:
                pauli = int(word[0])
                H_matrix[idx_row_cnt][idx_col_cnt] = pauli
                idx_col_cnt = 0
                idx_row_cnt += 1
        return H_matrix


    def parse_H_matrix_to_stim(self):
        n_qbts = int(self.num_col / 2)
        for row in self.H_matrix:
            pauli_string = ""
            for idx_col in range(n_qbts):
                if row[idx_col] == 1 and row[idx_col + n_qbts] == 1:
                    pauli_string += "Y"
                elif row[idx_col] == 1 and row[idx_col + n_qbts] == 0:
                    pauli_string += "X"
                elif row[idx_col] == 0 and row[idx_col + n_qbts] == 1:
                    pauli_string += "Z"
                else:
                    pauli_string += "_"
            self.pauli_stabilizers.append(stim.PauliString(pauli_string))


    def get_logicals(self):
        completed_tableau = stim.Tableau.from_stabilizers(
            self.pauli_stabilizers,
            allow_redundant=True,
            allow_underconstrained=True,
        )
        observables = []
        for k in range(len(completed_tableau))[::-1]:
            z = completed_tableau.z_output(k)
            if z in self.pauli_stabilizers:
                break
            x = completed_tableau.x_output(k)
            observables.append((x, z))
        return observables


    def parse_logicals_to_Z_2(self):
        n_qbts = int(self.num_col / 2)
        parsed_logs = []
        for log_pair in self.logicals:
            pauli_string_pair = []
            for logical in log_pair:
                log_pauli = ""
                logical_array = np.zeros(self.num_col, dtype=int)
                for idx_qbt, qbt in enumerate(logical):
                    if qbt == 1:
                        logical_array[idx_qbt] = 1
                        log_pauli += "X"
                    elif qbt == 2:
                        logical_array[idx_qbt] = 1
                        logical_array[idx_qbt + n_qbts] = 1
                        log_pauli += "Y"
                    elif qbt == 3:
                        logical_array[idx_qbt + n_qbts] = 1
                        log_pauli += "Z"
                    else:
                        log_pauli += "I"
                pauli_string_pair.append(log_pauli)
                parsed_logs.append(logical_array)
            if self.printing:
                print("Pauli strings: ", pauli_string_pair)
        return parsed_logs

    def combine_logicals_and_stab_gens(self):
        n_qbts = int(self.num_col / 2)
        H_matrix_X = np.zeros((self.num_row + self.num_logical_qbt * 2, n_qbts), dtype=int)
        H_matrix_Z = np.zeros((self.num_row + self.num_logical_qbt * 2, n_qbts), dtype=int)
        for idx_row in range(self.num_row):
            for idx_col in range(n_qbts):
                H_matrix_X[idx_row][idx_col] = self.H_matrix[idx_row][idx_col]
                H_matrix_Z[idx_row][idx_col] = self.H_matrix[idx_row][idx_col + n_qbts]

        for idx_row in range(self.num_logical_qbt * 2):
            for idx_col in range(n_qbts):
                H_matrix_X[idx_row + self.num_row][idx_col] = self.logicals_in_Z_2[idx_row][idx_col]
                H_matrix_Z[idx_row + self.num_row][idx_col] = self.logicals_in_Z_2[idx_row][idx_col + n_qbts]
        return H_matrix_X, H_matrix_Z


    def parse_stabilizers_to_SG_converter(self):
        stabilizers = []
        H = np.concatenate((self.H_X, self.H_Z), axis=1)
        idx = 0
        for idx in range(self.num_row):
            row = H[idx]
            stab_parsed = ""
            for _ in range(self.num_logical_qbt):
                stab_parsed += "I"
            stab_parsed_paulis = self.binary_to_str(row)
            for pauli in stab_parsed_paulis:
                stab_parsed += pauli
            stabilizers.append(stab_parsed)
        logical_counter = 0
        for i in range(self.num_logical_qbt):
            logical_parsed_X_paulis = self.binary_to_str(H[self.num_row + logical_counter])
            logical_parsed_Z_paulis = self.binary_to_str(H[self.num_row + logical_counter + 1])
            logical_parsed_X = ""
            logical_parsed_Z = ""
            logical_counter += 2
            for j in range(self.num_logical_qbt):
                if i == j:
                    logical_parsed_X += "X"
                    logical_parsed_Z += "Z"
                else:
                    logical_parsed_X += "I"
                    logical_parsed_Z += "I"
            for qbt_idx in range(len(logical_parsed_X_paulis)):
                logical_parsed_X += logical_parsed_X_paulis[qbt_idx]
                logical_parsed_Z += logical_parsed_Z_paulis[qbt_idx]
            stabilizers.append(logical_parsed_X)
            stabilizers.append(logical_parsed_Z)
            idx += 1
        return stabilizers



if __name__ == '__main__':
    # TODO: Check here that stuff only anticommutes on one qubit!
    path = r"C:\Users\Admin\Desktop\HeraldedResourceStateGen\codetableHmatrices\5_qbt_H_matrix_1_logical.txt"
    dim_row = 4
    dim_col = 5 * 2
    num_logicals = 1

    # parse_H_matrix(path, dim_row, dim_col)
    Parser = CodetableCodeParsing(path, dim_row, dim_col, num_logicals, printing=True)
    print(Parser.parse_H_matrix())
    print(Parser.pauli_stabilizers)
    print()
    log_X = Parser.logicals[0][0]
    for qbt in log_X:
        print(qbt)
    print()
    H = np.concatenate((Parser.H_X, Parser.H_Z), axis=1)
    for row in H:
        print(Parser.binary_to_str(row))
