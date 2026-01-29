import numpy as np
import json

def parse_single_file(filename):
    f = open(filename)
    results = json.load(f)
    numb_matt_qbts = []
    for list_out in results["saved_data"]:
        for item in list_out:
            matt_qbts = list(item.values())[0]
            numb_matt_qbts.append(matt_qbts)
    return numb_matt_qbts

if __name__ == '__main__':
    import matplotlib.pyplot as plt


    file_dir = r"C:\Users\bdt697\Documents\HarvardAdaptiveDecoder\10_qbt_graph_saved_data"
    loop_range = 18144
    full_output = []
    output_dict = {}
    min_matt_qbt = 10
    min_cnt = 0
    for idx in range(199):
        filename = file_dir + r"\g_" + str(idx) + ".json"
        output = parse_single_file(filename)
        for ix in range(loop_range):
            matt_qbt = output[ix]
            full_output.append(matt_qbt)
            output_dict[ix * loop_range + ix] = matt_qbt
            if matt_qbt < min_matt_qbt:
                min_cnt = 1
                min_matt_qbt = matt_qbt
            elif matt_qbt == min_matt_qbt:
                min_cnt += 1
    print("Minimum number of emitters: ", min_matt_qbt, ", with number of hits: ", min_cnt)
    lii_unique = list(set(full_output))

    # This is the corresponding count for each value
    counts = [full_output.count(value) for value in lii_unique]

    fig, ax = plt.subplots()
    bar_container = ax.bar(lii_unique, counts)
    print(lii_unique, counts)
    print(sum(counts))
    ax.set(ylabel='Counts', title='Number of matter qubits')
    ax.set_yscale("log")
    ax.legend()
    plt.show()
