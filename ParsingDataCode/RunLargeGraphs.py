from RunningDecoderLaptop import *
from CodeFunctions.graphs import *
from itertools import permutations
from ErasureDecoder import *
from multiprocessing import Pool

def run_patterns(meas_pattern, meas_idx):
    saved_inst = {str(meas_idx): 0}
    meas_pattern_list = [qbt for qbt in meas_pattern]

    no_anti_com_flag = False
    adaptive_decoder = LT_FullHybridDecoderNew(gstate, input_strats, measurement_order=meas_pattern_list,
                                            no_anti_com_flag=no_anti_com_flag, printing=False)
    numb_matt_qbts = adaptive_decoder.max_number_of_m_dec
    saved_inst[str(meas_idx)] = numb_matt_qbts
    return saved_inst


def fun_wrapper_pattern_loop(pattern_info):
    return run_patterns(*pattern_info)

n_qbts = 16
last_node = 16
graph_edges = [(0, 2), (0, 6), (0, 8), (0, 10), (0, 14), (0, 16), (1, 3), (1, 4), (1, 6), (1, 9), (1, 10), (1, 13), (1, 15), (1, 16), (2, 4), (2, 5), (2, 7), (2, 10), (2, 11), (2, 14), (2, 16), (3, 8), (3, 9), (3, 10), (3, 11), (3, 16), (4, 6), (4, 7), (4, 9), (4, 12), (4, 13), (4, 16), (5, 6), (5, 7), (5, 8), (5, 9), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (6, 8), (6, 10), (6, 11), (6, 12), (6, 14), (6, 16), (7, 9), (7, 10), (7, 12), (7, 15), (7, 16), (8, 10), (8, 11), (8, 13), (8, 16), (9, 10), (9, 11), (9, 14), (9, 15), (9, 16), (10, 13), (10, 16), (11, 13), (11, 14), (11, 16), (12, 14), (12, 16), (13, 15), (13, 16), (14, 16), (15, 16)]
distance = 5
graph_nodes = list(range(n_qbts + 1))
in_qubit = 0
graph_edges = interchange_nodes(last_node, graph_edges)
gstate = graph_from_nodes_and_edges(graph_nodes,
                                    graph_edges)
erasure_decoder = LT_Erasure_decoder(n_qbts, distance, gstate, in_qbt=in_qubit)
input_strats = erasure_decoder.strategies_ordered
print(input_strats[0])
if __name__ == '__main__':
    save_name = "16_1_6_qbt_graph_data"
    filename = r"C:\Users\bdt697\Downloads\final_best_permutations_16_1_6.csv"
    split_range = 16
    pool = Pool()

    loop_range = 21186
    results = []
    patterns = get_full_m_patt_list(filename)
    for idx in range(loop_range):
        if idx % 100 == 0:
            print("at idx: ", idx)
        start_range = split_range * idx
        end_range = split_range * (idx + 1)
        results.append(pool.map(fun_wrapper_pattern_loop,
                                [(patterns[ix], ix) for ix in range(start_range, end_range)]))
        saved_dict = {"saved_data": results}
    print(results)
    with open(save_dir + r"\graph_" + save_name + ".json", 'w') as fp:
        json.dump(save_dict, fp)