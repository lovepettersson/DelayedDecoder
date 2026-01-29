import json
import networkx as nx
import pandas as pd


def get_meas_order_list(string):
    meas_list = []
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    flag_app = True
    for idx_char, char in enumerate(string):
        if char in numbers and string[idx_char + 1] not in numbers:
            if flag_app:
                meas_list.append(int(char))
            else:
                flag_app = True
        elif char in numbers and string[idx_char + 1] in numbers:
            meas_list.append(int(char + string[idx_char + 1]))
            flag_app = False
        # else:
        #     flag_app = False
    return meas_list


def get_edge_list(string):
    edge_list = []
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    current_edge = []
    flag_app = True
    for idx_char, char in enumerate(string):
        if len(current_edge) == 2:
            edge_list.append(tuple(current_edge))
            current_edge = []
        if char in numbers and string[idx_char + 1] not in numbers:
            if len(current_edge) < 2 and flag_app:
                current_edge.append(int(char))
            else:
                flag_app = True
        elif char in numbers and string[idx_char + 1] in numbers:
            if len(current_edge) < 2:
                current_edge.append(int(char + string[idx_char + 1]))
                flag_app = False
    return edge_list



def parse_gefen_data(path):
    df = pd.read_csv(path)
    edge_list = []
    meas_pattern = []
    for idx in range(len(df['Permutation'])):
        # edges = df['UpdatedEdgeListPython'][idx]
        perm = df["Permutation"][idx]
        # edge_list.append(edges)
        meas_pattern.append(perm)
        # matter_qbts = df['MatterQubits'][idx]
    return meas_pattern



def get_full_m_patt_list(filename):
    meas_patt = parse_gefen_data(filename)
    list_m_patt = []
    for m_patt in meas_patt:
        m = get_meas_order_list(m_patt)
        list_m_patt.append(m)
    return list_m_patt


if __name__ == '__main__':

    dic = r"C:\Users\bdt697\Downloads"
    # filename = r"\9_1_3_graph_results.csv"
    filename = r"\final_best_permutations_13_1_5_a.csv"
    # run_specific_graph(dic + filename, save_name="6_1_3_qbt.json", n_qbts=9, distance=2, erasure_decoder_flag=False)

    meas_patt = parse_gefen_data(dic + filename)
    print("Meas: ", meas_patt[:10])
    for m_patt in meas_patt:
        print(get_meas_order_list(m_patt))