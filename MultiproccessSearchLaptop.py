from CodeFunctions.graphs import *
from HybridDelayedMeasDecoderFixedMeasPatt import *
from itertools import permutations
import json
from multiprocessing import Pool




def parse_data(idx):
    directory = r"C:\Users\bdt697\Documents\HarvardAdaptiveDecoder\edge_files_new"
    filename = r"\10_qbt_graph_" + str(idx) + ".json"
    f = open(directory+filename)
    edge_patt_dict = json.load(f)
    patterns = []
    edge_list = []
    for idx in edge_patt_dict.keys():
        edge_list.append(edge_patt_dict[idx][0])
        patterns.append(edge_patt_dict[idx][1])
    f.close()
    return patterns, edge_list



n_qbts = 10
in_qubit = 0
distance = 3
graph_nodes = list(range(11))

def run_patterns(meas_pattern, graph_edges, meas_idx):
    saved_inst = {str(meas_idx): 0}
    meas_pattern_list = [qbt for qbt in meas_pattern]
    # graph_edges = edge_list[meas_idx]
    gstate = graph_from_nodes_and_edges(graph_nodes, graph_edges)
    # input_strats = AllPossStrats(graph_nodes, gstate).get_possible_decoding_strats()
    erasure_decoder = LT_Erasure_decoder_All_Strats(n_qbts, distance, gstate, in_qbt=in_qubit)
    input_strats = erasure_decoder.strategies_ordered
    no_anti_com_flag = False
    adaptive_decoder = LT_FullHybridDecoderNew(gstate, input_strats, measurement_order=meas_pattern_list,
                                            no_anti_com_flag=no_anti_com_flag, printing=False)
    numb_matt_qbts = adaptive_decoder.max_number_of_m_dec
    saved_inst[str(meas_idx)] = numb_matt_qbts
    return saved_inst


def fun_wrapper_pattern_loop(pattern_info):
    return run_patterns(*pattern_info)







if __name__ == '__main__':


    # save_dir = r"C:\Users\bdt697\Documents\HarvardAdaptiveDecoder\10_qbt_graph_saved_data"
    save_dir = r"C:\Users\bdt697\Documents\HarvardAdaptiveDecoder\10_qbt_graph_saved_data_all"


    split_range = 16  # 32
    pool = Pool()
    for file_idx in range(199):
    # for file_idx in range(84, 199):
        print("At idx: ", file_idx)
        patterns, edge_list = parse_data(file_idx)
        loop_range = 1134
        results = []
        for idx in range(loop_range):
            start_range = split_range * idx
            end_range = split_range * (idx + 1)

            results.append(pool.map(fun_wrapper_pattern_loop, [(patterns[ix], edge_list[ix], ix) for ix in range(start_range, end_range)]))
            saved_dict = {"saved_data": results}
        filename_save = r"\g_"
        with open(save_dir + filename_save + str(file_idx) + ".json", "w") as f:
            json.dump(saved_dict, f)
    print(results)
    pool.close()
    pool.join()
