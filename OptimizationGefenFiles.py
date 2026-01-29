from CodeFunctions.graphs import *

from HybridDelayedMeasDecoderFixedMeasPatt import *
from itertools import permutations
import json
from multiprocessing import Pool


filename = r"C:\Users\bdt697\Documents\HarvardAdaptiveDecoder\edges_files\10_qbt_graph_0.json"  # r"/users/lpetters/HarvardAdaptiveDecoder/10_qbt_graph.json"
f = open(filename)
edge_patt_dict = json.load(f)
patterns = []
edge_list = []
for idx in edge_patt_dict.keys():
    edge_list.append(edge_patt_dict[idx][0])
    patterns.append(edge_patt_dict[idx][1])
    if idx == 100:
        break

f.close()

n_qbts = 10
in_qubit = 0
distance = 3
graph_nodes = list(range(11))

def run_patterns(meas_pattern, meas_idx):
    saved_inst = {str(meas_idx): 0}
    meas_pattern_list = [qbt for qbt in meas_pattern]
    graph_edges = edge_list[meas_idx]
    gstate = graph_from_nodes_and_edges(graph_nodes, graph_edges)

    erasure_decoder = LT_Erasure_decoder(n_qbts, distance, gstate, in_qbt=in_qubit)
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

    #######################################################################
    ########################### ELEVEN QBTS ###############################
    #######################################################################

    graph_edges = [(10, 1), (10, 2), (10, 5), (10, 4), (10, 3), (0, 10), (10, 6), (10, 8),
                   (1, 2), (1, 5), (1, 9), (1, 8), (2, 7), (2, 5), (3, 4), (3, 5),
                   (3, 9), (3, 6), (4, 5), (4, 7), (4, 0), (6, 8), (6, 9), (6, 7),
                   (7, 0), (8, 9), (9, 0)]
    gstate = graph_from_nodes_and_edges(graph_nodes, graph_edges)
    erasure_decoder = LT_Erasure_decoder(n_qbts, distance, gstate, in_qbt=in_qubit)
    input_strats = erasure_decoder.strategies_ordered
    list_matt_qbts = []
    patterns = list(permutations(range(1, n_qbts + 1), n_qbts))
    outer_patterns_range = 3628800
    start = 0
    end = int(302400 / 2)
    split_range = 16  # 32
    pool = Pool()

    for out_idx in range(14, 15):
        # start_in = 1 * 7659 # 245088
        # end_in = 15 * 7659 # 245087
        start_in = 0  # 245088
        end_in = 4 # 245087
        results = []
        for idx in range(start_in, end_in):
            start_range = split_range * idx
            end_range = split_range * (idx + 1)

            results.append(pool.map(fun_wrapper_pattern_loop, [(patterns[ix], ix) for ix in range(start_range, end_range)]))
            saved_dict = {"saved_data": results}
            with open("eleven_qubit_optimization" + str(out_idx) + ".json", "w") as f:
                json.dump(saved_dict, f)
    print(results)
# for out_idx in range(14, 15):
#     start_in = 14 * 7659 # 245088
#     end_in = 15 * 7659 # 245087
#     results = []
#     for idx in range(start_in, end_in):
#         start_range = split_range * idx
#         end_range = split_range * (idx + 1)
#
#         results.append(pool.map(fun_wrapper_pattern_loop, [(patterns[ix], ix) for ix in range(start_range, end_range)]))
#         saved_dict = {"saved_data": results}
#         with open("eleven_qubit_optimization" + str(out_idx) + ".json", "w") as f:
#             json.dump(saved_dict, f)
