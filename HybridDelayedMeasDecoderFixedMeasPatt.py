import numpy as np
import math
from itertools import product
from ErasureDecoder import *


###################################
####### FULL DECODER CLASS ########
###################################

class LT_FullHybridDecoder(object):
    r"""
    """

    def __init__(self, gstate, strats, measurement_order=[x for x in range(1, 8)], first_output_qbt=0, in_qubit=0,  no_anti_com_flag=False, printing=False):
        # Initialize the decoder
        self.gstate = gstate
        self.in_qubit = in_qubit
        self.no_anti_com_flag = no_anti_com_flag
        self.printing = printing
        self.first_output_qbt = first_output_qbt
        self.tree_branches = []
        self.measurement_order = measurement_order

        self.min_loss_pattern = [x for x in range(len(self.gstate) + 1)]
        self.all_min_loss_patterns = []
        self.poss_strat_list = strats
        self.output_qbt_counter = {}
        for strat in self.poss_strat_list:
            output_qbt = strat[0][1]
            if output_qbt not in self.output_qbt_counter.keys():
                self.output_qbt_counter[output_qbt] = [strat]
            else:
                self.output_qbt_counter[output_qbt].append(strat)




        self.anti_support_list_hybrid = []
        self.anti_support_dict_hybrid = {}

        ##################################################
        #### START BY ENUMERATING POSSIBLE STARTEGIES ####
        ##################################################

        self.calc_qubit_support()


        # print(self.output_qbt_counter)

        #####################
        #### RUN DECODER ####
        #####################

        self.analytic_exp = self.decode_hybrid(self.poss_strat_list)

        ############################################
        #### FIND THE DELAYED MEASUREMENT PATHS ####
        ############################################

        self.max_number_of_m_dec = 0
        for key in self.anti_support_dict_hybrid.keys():
            for anti_list in self.anti_support_dict_hybrid[key]:
                if len(anti_list) > self.max_number_of_m_dec:
                    self.max_number_of_m_dec = len(anti_list)
        if self.printing:
            for key in self.anti_support_dict_hybrid.keys():
                print("Key: ", key, ", anticom. support ", self.anti_support_dict_hybrid[key])

            print("Smallest loss pattern: ", self.min_loss_pattern)
            # print("All smallest loss patterns: ", self.all_min_loss_patterns)




    def bundle_all_strats(self, input_strat_dict):
        # Order the strategies after their support weigth
        list_of_all_strats = []
        for key in input_strat_dict.keys():
            for strat in input_strat_dict[key][:-1]:
                list_of_all_strats.append(strat)
        list_of_all_strats.sort(key=lambda x: x[4])
        return list_of_all_strats

    def binom_coeff(self, n, k):
        return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


    def filter_strat_lost_qubit(self, strats, lost_qbt_ix):
        # Filter strategies to the loss pattern
        return [these_stabs for these_stabs in strats if these_stabs[3][lost_qbt_ix] == 'I']



    def calc_qubit_support(self):
        for key in self.output_qbt_counter.keys():
            support = []
            for strat in self.output_qbt_counter[key]:
                meas_supp = strat[1]
                for qbt in meas_supp:
                    if qbt not in support:
                        support.append(qbt)
            self.output_qbt_counter[key].append(support)



    def update_output_counter_dict(self, strats):
        self.output_qbt_counter = {}
        for strat in strats:
            output_qbt = strat[0][1]
            if output_qbt not in self.output_qbt_counter.keys():
                self.output_qbt_counter[output_qbt] = [strat]
            else:
                self.output_qbt_counter[output_qbt].append(strat)
        # Update qbt support as well
        self.calc_qubit_support()



    def pick_output_qbt(self, output_cnt, available_strats):
        # Here there could be a fancier heuristic. At the momemnt this is the same as SPF
        target_output = self.measurement_order[output_cnt]
        for strat in available_strats:
            output_qbt = strat[0][1]
            if output_qbt == target_output:
                return strat
        return available_strats[0]
        # return self.output_qbt_counter[self.measurement_order[output_cnt]][0]

    def decode_hybrid(self, input_strats):
        # Want analytic expression from all poss strats given output qbt, i.e., I assume that all
        # strats in the dict can be realized.
        current_poss_strats = copy.deepcopy(input_strats)
        analytic_exp = []
        numb_lost_output_qbts = 0
        tracking_lost_output_qbts = []
        output_cnt = 0
        while len(current_poss_strats) != 0:
            strat = self.pick_output_qbt(output_cnt, current_poss_strats)
            anti_com_pair = strat[0]
            output_qbt = anti_com_pair[1]
            poss_further_strats = self.output_qbt_counter[output_qbt]
            loss_patts = self.decode_specific_output_qbt(poss_further_strats[:-1], output_qbt, tracking_lost_output_qbts)
            for l_patt in loss_patts:
                trans, loss = l_patt
                analytic_exp.append([loss, trans])

            # Updating strats if output qbt is lost
            numb_lost_output_qbts += 1
            tracking_lost_output_qbts.append(output_qbt)
            current_poss_strats = self.filter_strat_lost_qubit(current_poss_strats, output_qbt)
            self.measurement_order.remove(self.measurement_order[0])
            if self.printing:
                print("Just measured output qubit: ", output_qbt)
                print("Number of tolerable loss patterns so far is ", len(analytic_exp), ", and number of surviving strategies is ", len(current_poss_strats))
            if len(current_poss_strats) != 0:
                current_poss_strats.sort(key=lambda x: x[4])  # Do I need to do thsi again ?
                self.update_output_counter_dict(current_poss_strats)

        if self.printing:
            print("Finished, with said number of loss patterns ", len(analytic_exp))
        return analytic_exp




    def get_next_strat(self, strats, meas_before_qbt_and_pauli, next_measure_qbt, previous_output_qbt=0):
        meas_before = []
        meas_pauli = {}
        for outcome in meas_before_qbt_and_pauli:
            qbt, pauli = outcome
            meas_pauli[qbt] = pauli
            meas_before.append(qbt)

        min_anti_com_supp = 1000
        min_supp = 1000
        min_idx = 0
        min_output_qbt = -1
        for idx, strat in enumerate(strats):
            support = strat[1]
            pauli_string = strat[3]
            output_qbt = strat[0][1]
            support_size = 0
            anti_comm_size = 0
            for qbt in support:
                if qbt not in meas_before:
                    support_size += 1
                else:
                    pauli_op = pauli_string[qbt]
                    meas_op = meas_pauli[qbt]
                    if qbt == output_qbt:
                        if output_qbt != previous_output_qbt:
                            anti_comm_size += 1
                        if pauli_op != "I" and pauli_op != meas_op:
                            anti_comm_size += 1
                    else:
                        if pauli_op != "I" and pauli_op != meas_op:
                            anti_comm_size += 1

            if next_measure_qbt in support:
                if support_size <= min_supp and output_qbt == previous_output_qbt:
                    min_supp = support_size
                    min_anti_com_supp = anti_comm_size
                    min_idx = idx
                    min_output_qbt = output_qbt
                elif support_size < min_supp:
                    min_supp = support_size
                    min_anti_com_supp = anti_comm_size
                    min_idx = idx
                    min_output_qbt = output_qbt
                elif support_size == min_supp:
                    if anti_comm_size < min_anti_com_supp:
                        min_anti_com_supp = anti_comm_size
                        min_supp = support_size
                        min_idx = idx
                        min_output_qbt = output_qbt
        return strats[min_idx]



    def get_next_strat_version2(self, strats, meas_before_qbt_and_pauli, previous_output_qbt=0):
        meas_before = []
        meas_pauli = {}
        for outcome in meas_before_qbt_and_pauli:
            qbt, pauli = outcome
            meas_pauli[qbt] = pauli
            meas_before.append(qbt)

        min_anti_com_supp = 1000
        min_supp = 1000
        min_idx = 0
        min_output_qbt = -1
        for idx, strat in enumerate(strats):
            support = strat[1]
            pauli_string = strat[3]
            output_qbt = strat[0][1]
            support_size = 0
            anti_comm_size = 0
            for qbt in support:
                if qbt not in meas_before:
                    support_size += 1
                else:
                    pauli_op = pauli_string[qbt]
                    meas_op = meas_pauli[qbt]
                    if qbt == output_qbt:
                        if output_qbt != previous_output_qbt:
                            anti_comm_size += 1
                        if pauli_op != "I" and pauli_op != meas_op:
                            anti_comm_size += 1
                    else:
                        if pauli_op != "I" and pauli_op != meas_op:
                            anti_comm_size += 1

            if support_size <= min_supp and output_qbt == previous_output_qbt:
                # if min_output_qbt == previous_output_qbt:
                if anti_comm_size < min_anti_com_supp:
                    min_anti_com_supp = anti_comm_size
                    min_supp = support_size
                    min_idx = idx
                    min_output_qbt = output_qbt
                # else:
                #     min_supp = support_size
                #     min_anti_com_supp = anti_comm_size
                #     min_idx = idx
                #     min_output_qbt = output_qbt
            elif support_size < min_supp:
                min_supp = support_size
                min_anti_com_supp = anti_comm_size
                min_idx = idx
                min_output_qbt = output_qbt
            elif support_size == min_supp:
                if anti_comm_size < min_anti_com_supp:
                    min_anti_com_supp = anti_comm_size
                    min_supp = support_size
                    min_idx = idx
                    min_output_qbt = output_qbt
        # print("Picking")
        # print(strats[min_idx], min_anti_com_supp)
        # print()
        return strats[min_idx]



    def filter_anticom_strats_TBH(self, strats, meas_before_qbt_and_pauli, num_qbts=11):
        meas_before = []
        meas_pauli = {}
        for outcome in meas_before_qbt_and_pauli:
            qbt, pauli = outcome
            meas_pauli[qbt] = pauli
            meas_before.append(qbt)
        pauli_op = []
        for qbt in range(num_qbts):
            if qbt not in meas_before:
                pauli_op.append("I")
            else:
                pauli_op.append(meas_pauli[qbt])
        new_strats = []
        for strat in strats:
            pauli_op_check = strat[3]
            flag = True
            for qbt in range(num_qbts):
                commute_val = single_qubit_commute(pauli_op, pauli_op_check, qbt)
                if commute_val != 0:
                    flag = False
            if flag:
                new_strats.append(strat)
        return new_strats



    def decode_specific_output_qbt(self, input_strats, output_qbt, lost_output_qbt=[]):
        # Does the SPF decoding for a given output qbt
        # print("INPUT: ", input_strats)
        lt_finished = True
        tree_dict = {}
        beginning_order = []
        for qbt in lost_output_qbt:
            beginning_order.append(qbt)
        beginning_order.append(output_qbt)

        start_key = "S"
        for ix, qbt in enumerate(lost_output_qbt):
            start_key += "," + str(qbt)
        # start_key += str(output_qbt)

        tree_dict[start_key] = [copy.deepcopy(input_strats), [[output_qbt, "Z"]], beginning_order]
        current_keys = list(tree_dict.keys())
        succes_measured_stabs = {}
        while lt_finished:
            new_keys = []
            for key in current_keys:
                discovered_qbts = []
                meas_order = [tree_dict[key][2][idx] for idx in range(len(tree_dict[key][2]))]
                # meas_qbt = [output_qbt]
                meas_qbt = []
                meas_before = [tree_dict[key][1][idx][0] for idx in range(len(tree_dict[key][1]))]
                for item in tree_dict[key][1]:
                    meas_qbt.append(item)
                strats = copy.deepcopy(tree_dict[key][0])

                next_meas_qbt = 0
                for qbt in self.measurement_order:
                    if qbt not in meas_qbt:
                        next_meas_qbt = qbt
                        break
                # print("Meas bef: ", tree_dict[key][1], output_qbt)

                starting_stab = self.get_next_strat(tree_dict[key][0], tree_dict[key][1], next_meas_qbt, output_qbt)
                # succes_measured_stabs[key] = [starting_stab, copy.deepcopy(meas_qbt), meas_order, copy.deepcopy(lost_output_qbt)]
                pot_new_output_qbt = starting_stab[0][1]
                # print("pot new output: ", pot_new_output_qbt)
                starting_qbts_init = [next_meas_qbt]
                if pot_new_output_qbt != output_qbt and pot_new_output_qbt not in meas_qbt and pot_new_output_qbt != next_meas_qbt:
                    starting_qbts_init.append(pot_new_output_qbt)
                starting_qbts_init_extras = starting_stab[1]  # Only considers the supp of the attempted qbt
                for qbt in starting_qbts_init_extras:
                    starting_qbts_init.append(qbt)
                starting_qbts = []

                lost_qbts = self.from_key_get_list_of_lost_qbt(key)
                for qbt in lost_qbts:
                    discovered_qbts.append(qbt)
                # Dont measure the same qubits twice!
                # for qbt in starting_qbts_init:
                #     if qbt not in meas_before:
                #         starting_qbts.append(qbt)
                #     else:
                #         discovered_qbts.append(qbt)

                for qbt in self.measurement_order:
                    if qbt not in meas_before and qbt not in lost_qbts and qbt not in lost_output_qbt:
                        starting_qbts.append(qbt)
                    else:
                        discovered_qbts.append(qbt)
                succes_measured_stabs[key] = [starting_stab, copy.deepcopy(meas_qbt), meas_order,
                                              copy.deepcopy(lost_output_qbt), starting_qbts]
                # starting_qbts = self.pick_qbt_to_measure(tree_dict[key][0][0], tree_dict[key][0], starting_qbts)
                # print("Starting start: ", starting_stab, lost_qbts, starting_qbts, meas_qbt)
                # print(strats)
                # print()
                for qbt in starting_qbts:
                    update_strats = self.filter_strat_lost_qubit(strats, qbt)
                    save_key = key + "," + str(qbt)
                    if self.no_anti_com_flag:
                        # This is a flag indicating if one allows for delayed measurements or not
                        update_strats = self.filter_anticom_strats_TBH(update_strats, copy.deepcopy(meas_qbt))
                    else:
                        if len(update_strats) < 2:
                            # If there is only one strategy left for a given output qbt branch, either it is 1) the last valid strategy, or 2) we switch output qbt
                            update_strats = self.update_valid_strategies(self.in_qubit, output_qbt, key, qbt, update_strats)
                            save_key = key + "," + str(qbt)

                    if qbt not in beginning_order:
                        meas_order.append(qbt)
                    tree_dict[save_key] = [copy.deepcopy(update_strats), copy.deepcopy(meas_qbt), copy.deepcopy(meas_order)]
                    if len(update_strats) > 1:
                        # Continue the loop, there are more valid strategies
                        new_keys.append(save_key)
                    elif len(update_strats) == 1:
                        # succes_measured_stabs[save_key] = [update_strats, copy.deepcopy(meas_qbt), copy.deepcopy(meas_order), copy.deepcopy(lost_output_qbt)]
                        succes_measured_stabs[save_key] = [update_strats, copy.deepcopy(meas_qbt),
                                                           copy.deepcopy(meas_order), copy.deepcopy(lost_output_qbt), starting_qbts]
                    else:

                        # This lead to a logical loss, add the loss pattern to keep track of distance
                        lost_qbts = self.from_key_get_list_of_lost_qbt(key + "," + str(qbt))
                        total_lost = [x for x in lost_qbts]
                        for x in lost_output_qbt:
                            total_lost.append(x)
                        if len(total_lost) < len(self.min_loss_pattern):
                            self.min_loss_pattern = total_lost
                            self.all_min_loss_patterns = []
                            if self.printing:
                                print("New smallest los pattern: ", key, lost_qbts, lost_output_qbt, key + "," + str(qbt))
                        elif len(total_lost) == len(self.min_loss_pattern):
                            self.all_min_loss_patterns.append(total_lost)
                    pauli_op = starting_stab[3][qbt]
                    meas_qbt.append([qbt, pauli_op])
                    # meas_order.append(qbt)
                    # print(meas_qbt, meas_order)
            if len(new_keys) == 0:
                lt_finished = False
            else:
                current_keys = copy.deepcopy(new_keys)
        succ_patterns = []
        anti_support = []
        list_of_anti_support = []
        for key in succes_measured_stabs.keys():
            succ_patt = succes_measured_stabs[key]
            # print("Succ pat: ", succ_patt)
            tree_branch = [item for item in succ_patt]
            tree_branch.append(key)
            if len(succ_patt[0]) == 1:
                this_succ_patt_output_qbt = succ_patt[0][0][0][1]
            else:
                this_succ_patt_output_qbt = succ_patt[0][0][1]

            self.get_anticommuting_support(succ_patt, anti_support, list_of_anti_support, this_succ_patt_output_qbt)
            tree_branch.append(list_of_anti_support[-1])
            self.tree_branches.append(tree_branch)

            # self.get_anticommuting_support(succ_patt, anti_support, list_of_anti_support, output_qbt)
            lost = self.from_key_get_lost_qbt(key)
            if self.printing:
                print("Key, succ_patt, and lost", key, succ_patt, lost)
                print("Anti support: ", list_of_anti_support)

            s = succes_measured_stabs[key][0]
            if len(s) > 1:
                # pauli = succes_measured_stabs[key][0][3]
                # measured = succes_measured_stabs[key][1]
                # qbts_measured = [measured[idx][0] for idx in range(len(measured))]
                # trans = self.get_numb_of_meas_qbts(pauli, qbts_measured, output_qbt)
                measured = succes_measured_stabs[key][1]
                qbts_measured = [measured[idx][0] for idx in range(len(measured))]
                trans = len(qbts_measured) + len(succes_measured_stabs[key][-1])
            else:
                # pauli = succes_measured_stabs[key][0][0][3]
                # measured = succes_measured_stabs[key][1]
                # qbts_measured = [measured[idx][0] for idx in range(len(measured))]
                # trans = self.get_numb_of_meas_qbts(pauli, qbts_measured, output_qbt)
                measured = succes_measured_stabs[key][1]
                qbts_measured = [measured[idx][0] for idx in range(len(measured))]
                trans = len(qbts_measured) + len(succes_measured_stabs[key][-1])
            succ_patterns.append([trans, lost])
            # print(succ_patt)
            # print("trans and lost: ", trans, lost, ", and key", key)
            # print(list_of_anti_support[-1])
            # print()
        if self.printing:
            print("succ patterns, [trans, lost]: ", succ_patterns)

        self.anti_support_dict_hybrid[output_qbt] = list_of_anti_support  # anti_support
        for qbt in anti_support:
            if qbt not in self.anti_support_list_hybrid:
                self.anti_support_list_hybrid.append(qbt)
        # print("Succ patterns: ", succ_patterns)
        return succ_patterns

    def update_valid_strategies(self, in_qubit, output_qbt, key, qbt, in_strats):
        new_strats = []
        for s in in_strats:
            new_strats.append(s)
        for qbt_out in range(len(self.gstate)):
            if qbt_out != in_qubit and qbt_out != output_qbt and qbt_out in list(self.output_qbt_counter.keys()):
                poss_further_strats = self.output_qbt_counter[qbt_out][:-1]
                lost_qbts = self.from_key_get_list_of_lost_qbt(key + "," + str(qbt))
                for l_qbt in lost_qbts:
                    poss_further_strats = self.filter_strat_lost_qubit(poss_further_strats, l_qbt)
                if len(poss_further_strats) > 0:
                    for strat in poss_further_strats:
                        new_strats.append(strat)
        return new_strats

    def get_numb_of_meas_qbts(self, pauli, qbts_meas, output_qbt, printing=False):
        numb = len(qbts_meas)
        if output_qbt not in qbts_meas:
            numb += 1
        for qbt, p in enumerate(pauli):
            if qbt != self.in_qubit and qbt != output_qbt and qbt not in qbts_meas:
            # if qbt != self.in_qubit and qbt not in qbts_meas:
                if p != "I":
                    numb += 1
                    if printing:
                        print(qbt, p)
        return numb


    def get_anticommuting_support(self, outcome_list, anti_support, list_of_anti_support, init_output_qbt):
        meas_qbts = outcome_list[1]
        this_traj_anti = []
        if len(outcome_list[0]) == 1:
            pauli = outcome_list[0][0][3]
            output_qbt = outcome_list[0][0][0][1]
        else:
            pauli = outcome_list[0][3]
            output_qbt = outcome_list[0][0][1]
        this_traj_anti.append(init_output_qbt)
        if output_qbt not in this_traj_anti:
            this_traj_anti.append(output_qbt)
        if len(meas_qbts) != 0:
            if output_qbt not in anti_support:
                anti_support.append(output_qbt)
            for meas_pair in meas_qbts:
                qbt, meas_pauli = meas_pair
                if pauli[qbt] == "I" or pauli[qbt] == meas_pauli or qbt == output_qbt:
                    continue
                else:
                    this_traj_anti.append(qbt)
                    if qbt not in anti_support:
                        anti_support.append(qbt)

        list_of_anti_support.append(this_traj_anti)


    def from_key_get_lost_qbt(self, key):
        lost = 0
        for idx, char in enumerate(key):
            if idx != len(key) - 1:
                if char != "S" and char != "," and key[idx + 1] == ",":
                    lost += 1
            else:
                if char != "S" and char != ",":  # Double digit numbers is fixed here
                    lost += 1
        return lost



    def from_key_get_list_of_lost_qbt(self, key):
        lost = []
        for idx, char in enumerate(key):
            if idx != len(key) - 1:
                if char != "S" and char != ",": # and key[idx + 1] == ",":
                    if key[idx + 1] != ",":
                        lost.append(int(char + key[idx + 1]))
                    elif key[idx - 1] != ",":
                        continue
                    else:
                        lost.append(int(char))
            else:
                if key[idx - 1] != ",":
                    continue
                else:  # Double digit numbers is fixed here
                    lost.append(int(char))
        return lost


    def pick_qbt_to_measure(self, measuring_strat, input_all_strats, qbts_to_measure):
        largest_surv_strats = 0
        largest_surv_qbt = measuring_strat[1][0]
        flag_continue = True
        measure_order = []
        all_strats = copy.deepcopy(input_all_strats)
        while flag_continue:
            cnt = 0
            for qbt in qbts_to_measure:
                copied_all_strats = copy.deepcopy(all_strats)
                surv_strat = self.filter_strat_lost_qubit(copied_all_strats, qbt)
                if len(surv_strat) > largest_surv_strats:
                    largest_surv_strats = len(surv_strat)
                    largest_surv_qbt = qbt
                    cnt += 1
            if cnt > 0:
                measure_order.append(largest_surv_qbt)
                all_strats = self.filter_strat_lost_qubit(all_strats, largest_surv_qbt)
                largest_surv_qbt = 0
                if len(all_strats) == 0:
                    flag_continue = False

            else:
                flag_continue = False
        for qbt in qbts_to_measure:
            if qbt not in measure_order:
                measure_order.append(qbt)
        return measure_order



    def calculate_log_transmisison(self, transmission, analytic_exp):
        log_trans = []
        for trans in transmission:
            tot_trans = 0
            for trajectory in analytic_exp:
                loss_qbt, trans_qbt = trajectory
                tot_trans += (((1 - trans) ** loss_qbt) * (trans ** trans_qbt))
            log_trans.append(tot_trans)
        return log_trans


    def calculate_log_transmisison_from_min_loss_patt(self, transmission):
        numb_lost_patts = len(self.all_min_loss_patterns)
        min_qbts_lost = len(self.min_loss_pattern)
        n_total = len(self.gstate) - 1
        log_trans = []
        for trans in transmission:
            tot_trans = 0
            for n_qbt in range(min_qbts_lost + 1):
                pref = self.binom_coeff(n_total, n_qbt)
                if n_qbt == min_qbts_lost:
                    pref -= numb_lost_patts
                tot_trans += pref * (((1 - trans) ** n_qbt) * (trans ** (n_total - n_qbt)))
            log_trans.append(tot_trans)
        return log_trans



class LT_FullHybridDecoderNew(object):
    r"""
    """

    def __init__(self, gstate, strats, measurement_order=[x for x in range(1, 8)], first_output_qbt=0, in_qubit=0,  no_anti_com_flag=False, printing=False):
        # Initialize the decoder
        self.gstate = gstate
        self.in_qubit = in_qubit
        self.no_anti_com_flag = no_anti_com_flag
        self.printing = printing
        self.first_output_qbt = first_output_qbt
        self.tree_branches = []
        self.measurement_order = measurement_order

        self.min_loss_pattern = [x for x in range(len(self.gstate) + 1)]
        self.all_min_loss_patterns = []
        self.poss_strat_list = strats
        self.output_qbt_counter = {}
        for strat in self.poss_strat_list:
            output_qbt = strat[0][1]
            if output_qbt not in self.output_qbt_counter.keys():
                self.output_qbt_counter[output_qbt] = [strat]
            else:
                self.output_qbt_counter[output_qbt].append(strat)




        self.anti_support_list_hybrid = []
        self.anti_support_dict_hybrid = {}

        ##################################################
        #### START BY ENUMERATING POSSIBLE STARTEGIES ####
        ##################################################

        self.calc_qubit_support()


        # print(self.output_qbt_counter)

        #####################
        #### RUN DECODER ####
        #####################

        self.analytic_exp = self.decode_hybrid(self.poss_strat_list)

        ############################################
        #### FIND THE DELAYED MEASUREMENT PATHS ####
        ############################################

        self.max_number_of_m_dec = 0
        for key in self.anti_support_dict_hybrid.keys():
            for anti_list in self.anti_support_dict_hybrid[key]:
                if len(anti_list) > self.max_number_of_m_dec:
                    self.max_number_of_m_dec = len(anti_list)
        if self.printing:
            for key in self.anti_support_dict_hybrid.keys():
                print("Key: ", key, ", anticom. support ", self.anti_support_dict_hybrid[key])

            print("Smallest loss pattern: ", self.min_loss_pattern)
            # print("All smallest loss patterns: ", self.all_min_loss_patterns)




    def bundle_all_strats(self, input_strat_dict):
        # Order the strategies after their support weigth
        list_of_all_strats = []
        for key in input_strat_dict.keys():
            for strat in input_strat_dict[key][:-1]:
                list_of_all_strats.append(strat)
        list_of_all_strats.sort(key=lambda x: x[4])
        return list_of_all_strats

    def binom_coeff(self, n, k):
        return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


    def filter_strat_lost_qubit(self, strats, lost_qbt_ix):
        # Filter strategies to the loss pattern
        return [these_stabs for these_stabs in strats if these_stabs[3][lost_qbt_ix] == 'I']



    def calc_qubit_support(self):
        for key in self.output_qbt_counter.keys():
            support = []
            for strat in self.output_qbt_counter[key]:
                meas_supp = strat[1]
                for qbt in meas_supp:
                    if qbt not in support:
                        support.append(qbt)
            self.output_qbt_counter[key].append(support)



    def update_output_counter_dict(self, strats):
        self.output_qbt_counter = {}
        for strat in strats:
            output_qbt = strat[0][1]
            if output_qbt not in self.output_qbt_counter.keys():
                self.output_qbt_counter[output_qbt] = [strat]
            else:
                self.output_qbt_counter[output_qbt].append(strat)
        # Update qbt support as well
        self.calc_qubit_support()



    def pick_output_qbt(self, output_cnt, available_strats):
        # Here there could be a fancier heuristic. At the momemnt this is the same as SPF
        target_output = self.measurement_order[output_cnt]
        for strat in available_strats:
            output_qbt = strat[0][1]
            if output_qbt == target_output:
                return strat
        return available_strats[0]
        # return self.output_qbt_counter[self.measurement_order[output_cnt]][0]

    def decode_hybrid(self, input_strats):
        # Want analytic expression from all poss strats given output qbt, i.e., I assume that all
        # strats in the dict can be realized.
        current_poss_strats = copy.deepcopy(input_strats)
        analytic_exp = []
        numb_lost_output_qbts = 0
        tracking_lost_output_qbts = []
        output_cnt = 0
        while len(current_poss_strats) != 0:
            strat = self.pick_output_qbt(output_cnt, current_poss_strats)
            anti_com_pair = strat[0]
            output_qbt = anti_com_pair[1]
            poss_further_strats = self.output_qbt_counter[output_qbt]
            loss_patts = self.decode_specific_output_qbt(poss_further_strats[:-1], output_qbt, tracking_lost_output_qbts)
            for l_patt in loss_patts:
                trans, loss = l_patt
                analytic_exp.append([loss, trans])

            # Updating strats if output qbt is lost
            numb_lost_output_qbts += 1
            tracking_lost_output_qbts.append(output_qbt)
            current_poss_strats = self.filter_strat_lost_qubit(current_poss_strats, output_qbt)
            self.measurement_order.remove(self.measurement_order[0])
            if self.printing:
                print("Just measured output qubit: ", output_qbt)
                print("Number of tolerable loss patterns so far is ", len(analytic_exp), ", and number of surviving strategies is ", len(current_poss_strats))
            if len(current_poss_strats) != 0:
                current_poss_strats.sort(key=lambda x: x[4])  # Do I need to do thsi again ?
                self.update_output_counter_dict(current_poss_strats)

        if self.printing:
            print("Finished, with said number of loss patterns ", len(analytic_exp))
        return analytic_exp




    def get_next_strat(self, strats, meas_before_qbt_and_pauli, next_measure_qbt, previous_output_qbt=0):
        meas_before = []
        meas_pauli = {}
        for outcome in meas_before_qbt_and_pauli:
            qbt, pauli = outcome
            meas_pauli[qbt] = pauli
            meas_before.append(qbt)

        min_anti_com_supp = 1000
        min_supp = 1000
        min_idx = 0
        min_output_qbt = -1
        for idx, strat in enumerate(strats):
            support = strat[1]
            pauli_string = strat[3]
            output_qbt = strat[0][1]
            support_size = 0
            anti_comm_size = 0
            for qbt in support:
                if qbt not in meas_before:
                    support_size += 1
                else:
                    pauli_op = pauli_string[qbt]
                    meas_op = meas_pauli[qbt]
                    if qbt == output_qbt:
                        if output_qbt != previous_output_qbt:
                            anti_comm_size += 1
                        if pauli_op != "I" and pauli_op != meas_op:
                            anti_comm_size += 1
                    else:
                        if pauli_op != "I" and pauli_op != meas_op:
                            anti_comm_size += 1

            if next_measure_qbt in support:
                if support_size <= min_supp and output_qbt == previous_output_qbt:
                    min_supp = support_size
                    min_anti_com_supp = anti_comm_size
                    min_idx = idx
                    min_output_qbt = output_qbt
                elif support_size < min_supp:
                    min_supp = support_size
                    min_anti_com_supp = anti_comm_size
                    min_idx = idx
                    min_output_qbt = output_qbt
                elif support_size == min_supp:
                    if anti_comm_size < min_anti_com_supp:
                        min_anti_com_supp = anti_comm_size
                        min_supp = support_size
                        min_idx = idx
                        min_output_qbt = output_qbt
        return strats[min_idx]



    def get_next_strat_version2(self, strats, meas_before_qbt_and_pauli, previous_output_qbt=0):
        meas_before = []
        meas_pauli = {}
        for outcome in meas_before_qbt_and_pauli:
            qbt, pauli = outcome
            meas_pauli[qbt] = pauli
            meas_before.append(qbt)

        min_anti_com_supp = 1000
        min_supp = 1000
        min_idx = 0
        min_output_qbt = -1
        for idx, strat in enumerate(strats):
            support = strat[1]
            pauli_string = strat[3]
            output_qbt = strat[0][1]
            support_size = 0
            anti_comm_size = 0
            for qbt in support:
                if qbt not in meas_before:
                    support_size += 1
                else:
                    pauli_op = pauli_string[qbt]
                    meas_op = meas_pauli[qbt]
                    if qbt == output_qbt:
                        if output_qbt != previous_output_qbt:
                            anti_comm_size += 1
                        if pauli_op != "I" and pauli_op != meas_op:
                            anti_comm_size += 1
                    else:
                        if pauli_op != "I" and pauli_op != meas_op:
                            anti_comm_size += 1

            if support_size <= min_supp and output_qbt == previous_output_qbt:
                # if min_output_qbt == previous_output_qbt:
                if anti_comm_size < min_anti_com_supp:
                    min_anti_com_supp = anti_comm_size
                    min_supp = support_size
                    min_idx = idx
                    min_output_qbt = output_qbt
                # else:
                #     min_supp = support_size
                #     min_anti_com_supp = anti_comm_size
                #     min_idx = idx
                #     min_output_qbt = output_qbt
            elif support_size < min_supp:
                min_supp = support_size
                min_anti_com_supp = anti_comm_size
                min_idx = idx
                min_output_qbt = output_qbt
            elif support_size == min_supp:
                if anti_comm_size < min_anti_com_supp:
                    min_anti_com_supp = anti_comm_size
                    min_supp = support_size
                    min_idx = idx
                    min_output_qbt = output_qbt
        # print("Picking")
        # print(strats[min_idx], min_anti_com_supp)
        # print()
        return strats[min_idx]



    def filter_anticom_strats_TBH(self, strats, meas_before_qbt_and_pauli, num_qbts=11):
        meas_before = []
        meas_pauli = {}
        for outcome in meas_before_qbt_and_pauli:
            qbt, pauli = outcome
            meas_pauli[qbt] = pauli
            meas_before.append(qbt)
        pauli_op = []
        for qbt in range(num_qbts):
            if qbt not in meas_before:
                pauli_op.append("I")
            else:
                pauli_op.append(meas_pauli[qbt])
        new_strats = []
        for strat in strats:
            pauli_op_check = strat[3]
            flag = True
            for qbt in range(num_qbts):
                commute_val = single_qubit_commute(pauli_op, pauli_op_check, qbt)
                if commute_val != 0:
                    flag = False
            if flag:
                new_strats.append(strat)
        return new_strats



    def decode_specific_output_qbt(self, input_strats, output_qbt, lost_output_qbt=[]):
        # Does the SPF decoding for a given output qbt
        # print("INPUT: ", input_strats)
        lt_finished = True
        tree_dict = {}
        beginning_order = []
        for qbt in lost_output_qbt:
            beginning_order.append(qbt)
        beginning_order.append(output_qbt)

        start_key = "S"
        for ix, qbt in enumerate(lost_output_qbt):
            start_key += "," + str(qbt)
        # start_key += str(output_qbt)

        tree_dict[start_key] = [copy.deepcopy(input_strats), [[output_qbt, "A"]], beginning_order]  # Changed to A
        current_keys = list(tree_dict.keys())
        succes_measured_stabs = {}
        while lt_finished:
            new_keys = []
            for key in current_keys:
                discovered_qbts = []
                meas_order = [tree_dict[key][2][idx] for idx in range(len(tree_dict[key][2]))]
                # meas_qbt = [output_qbt]
                meas_qbt = []
                meas_before = [tree_dict[key][1][idx][0] for idx in range(len(tree_dict[key][1]))]
                for item in tree_dict[key][1]:
                    meas_qbt.append(item)
                # print("Meas before: ", meas_before, meas_qbt)
                strats = copy.deepcopy(tree_dict[key][0])

                next_meas_qbt = 0
                for qbt in self.measurement_order:
                    if qbt not in meas_qbt:
                        next_meas_qbt = qbt
                        break
                # print("Meas bef: ", tree_dict[key][1], output_qbt)

                starting_stab = self.get_next_strat(tree_dict[key][0], tree_dict[key][1], next_meas_qbt, output_qbt)
                # print("New stab: ", starting_stab)

                # Update tree with a Pauli operator of a previous output qubit measurement "A".
                tree_dict = self.update_tree_meas_pattern(meas_qbt, starting_stab, tree_dict)
                tree_dict = self.update_identity_in_tree_meas_pattern(meas_qbt, starting_stab, tree_dict)
                # succes_measured_stabs[key] = [starting_stab, copy.deepcopy(meas_qbt), meas_order, copy.deepcopy(lost_output_qbt)]
                pot_new_output_qbt = starting_stab[0][1]
                # print("pot new output: ", pot_new_output_qbt)
                starting_qbts_init = [next_meas_qbt]
                if pot_new_output_qbt != output_qbt and pot_new_output_qbt not in meas_qbt and pot_new_output_qbt != next_meas_qbt:
                    starting_qbts_init.append(pot_new_output_qbt)
                starting_qbts_init_extras = starting_stab[1]  # Only considers the supp of the attempted qbt
                for qbt in starting_qbts_init_extras:
                    starting_qbts_init.append(qbt)
                starting_qbts = []

                lost_qbts = self.from_key_get_list_of_lost_qbt(key)
                for qbt in lost_qbts:
                    discovered_qbts.append(qbt)

                for qbt in self.measurement_order:
                    if qbt not in meas_before and qbt not in lost_qbts and qbt not in lost_output_qbt:
                        starting_qbts.append(qbt)
                    else:
                        discovered_qbts.append(qbt)
                succes_measured_stabs[key] = [starting_stab, copy.deepcopy(meas_qbt), meas_order,
                                              copy.deepcopy(lost_output_qbt), starting_qbts]
                # starting_qbts = self.pick_qbt_to_measure(tree_dict[key][0][0], tree_dict[key][0], starting_qbts)
                # print("Starting start: ", starting_stab, lost_qbts, starting_qbts, meas_qbt)
                # print(strats)
                # print()
                for qbt in starting_qbts:
                    update_strats = self.filter_strat_lost_qubit(strats, qbt)
                    save_key = key + "," + str(qbt)
                    if self.no_anti_com_flag:
                        # This is a flag indicating if one allows for delayed measurements or not
                        update_strats = self.filter_anticom_strats_TBH(update_strats, copy.deepcopy(meas_qbt))
                    else:
                        if len(update_strats) < 2:
                            # If there is only one strategy left for a given output qbt branch, either it is 1) the last valid strategy, or 2) we switch output qbt
                            update_strats = self.update_valid_strategies(self.in_qubit, output_qbt, key, qbt, update_strats)
                            save_key = key + "," + str(qbt)

                    if qbt not in beginning_order:
                        meas_order.append(qbt)
                    tree_dict[save_key] = [copy.deepcopy(update_strats), copy.deepcopy(meas_qbt), copy.deepcopy(meas_order)]
                    if len(update_strats) > 1:
                        # Continue the loop, there are more valid strategies
                        new_keys.append(save_key)
                    elif len(update_strats) == 1:
                        # succes_measured_stabs[save_key] = [update_strats, copy.deepcopy(meas_qbt), copy.deepcopy(meas_order), copy.deepcopy(lost_output_qbt)]
                        # print("FININSHED: ", update_strats, meas_qbt, qbt)
                        meas_qbt = self.update_meas_qbts_in_loop(meas_qbt, update_strats[0])
                        meas_qbt = self.update_identity_in_meas_qbts_in_loop(meas_qbt, update_strats[0])
                        succes_measured_stabs[save_key] = [update_strats, copy.deepcopy(meas_qbt),
                                                           copy.deepcopy(meas_order), copy.deepcopy(lost_output_qbt), starting_qbts]
                    else:
                        # This lead to a logical loss, add the loss pattern to keep track of distance
                        lost_qbts = self.from_key_get_list_of_lost_qbt(key + "," + str(qbt))
                        total_lost = [x for x in lost_qbts]
                        for x in lost_output_qbt:
                            total_lost.append(x)
                        if len(total_lost) < len(self.min_loss_pattern):
                            self.min_loss_pattern = total_lost
                            self.all_min_loss_patterns = []
                            if self.printing:
                                print("New smallest los pattern: ", key, lost_qbts, lost_output_qbt, key + "," + str(qbt))
                        elif len(total_lost) == len(self.min_loss_pattern):
                            self.all_min_loss_patterns.append(total_lost)
                    pauli_op = starting_stab[3][qbt]
                    meas_qbt.append([qbt, pauli_op])
                    # meas_order.append(qbt)
                    # print(meas_qbt, meas_order)
            if len(new_keys) == 0:
                lt_finished = False
            else:
                current_keys = copy.deepcopy(new_keys)
        succ_patterns = []
        anti_support = []
        list_of_anti_support = []
        for key in succes_measured_stabs.keys():
            succ_patt = succes_measured_stabs[key]
            # print("Succ pat: ", succ_patt)
            tree_branch = [item for item in succ_patt]
            tree_branch.append(key)
            if len(succ_patt[0]) == 1:
                this_succ_patt_output_qbt = succ_patt[0][0][0][1]
            else:
                this_succ_patt_output_qbt = succ_patt[0][0][1]

            self.get_anticommuting_support(succ_patt, anti_support, list_of_anti_support, this_succ_patt_output_qbt)
            tree_branch.append(list_of_anti_support[-1])
            # print("Anti support: ", list_of_anti_support[-1])
            self.tree_branches.append(tree_branch)

            # self.get_anticommuting_support(succ_patt, anti_support, list_of_anti_support, output_qbt)
            lost = self.from_key_get_lost_qbt(key)
            if self.printing:
                print("Key, succ_patt, and lost", key, succ_patt, lost)
                print("Anti support: ", list_of_anti_support)

            s = succes_measured_stabs[key][0]
            if len(s) > 1:
                # pauli = succes_measured_stabs[key][0][3]
                # measured = succes_measured_stabs[key][1]
                # qbts_measured = [measured[idx][0] for idx in range(len(measured))]
                # trans = self.get_numb_of_meas_qbts(pauli, qbts_measured, output_qbt)
                measured = succes_measured_stabs[key][1]
                qbts_measured = [measured[idx][0] for idx in range(len(measured))]
                trans = len(qbts_measured) + len(succes_measured_stabs[key][-1])
            else:
                # pauli = succes_measured_stabs[key][0][0][3]
                # measured = succes_measured_stabs[key][1]
                # qbts_measured = [measured[idx][0] for idx in range(len(measured))]
                # trans = self.get_numb_of_meas_qbts(pauli, qbts_measured, output_qbt)
                measured = succes_measured_stabs[key][1]
                qbts_measured = [measured[idx][0] for idx in range(len(measured))]
                trans = len(qbts_measured) + len(succes_measured_stabs[key][-1])
            succ_patterns.append([trans, lost])
            # print(succ_patt)
            # print("trans and lost: ", trans, lost, ", and key", key)
            # print(list_of_anti_support[-1])
            # print()
        if self.printing:
            print("succ patterns, [trans, lost]: ", succ_patterns)

        self.anti_support_dict_hybrid[output_qbt] = list_of_anti_support  # anti_support
        for qbt in anti_support:
            if qbt not in self.anti_support_list_hybrid:
                self.anti_support_list_hybrid.append(qbt)
        # print("Succ patterns: ", succ_patterns)
        return succ_patterns


    def update_tree_meas_pattern(self, meas_qbts, new_strat, tree):
        new_pauli_measurement_qubits = new_strat[1]
        new_pauli_measurements = new_strat[3]
        new_pauli_meas = ""
        new_pauli_meas_qbt = 0
        flag = False
        new_tree = {}
        for meas_pair in meas_qbts:
            qbt, meas_outcome = meas_pair
            if meas_outcome == "A":
                if qbt in new_pauli_measurement_qubits:
                    new_pauli_meas = new_pauli_measurements[qbt]
                    new_pauli_meas_qbt = qbt
                    flag = True
                    break
        if flag:
            for key in tree.keys():
                new_tree[key] = []
                first_item, meas_qbts_prev, third_item = tree[key]
                new_tree[key].append(first_item)
                new_meas_qbts_prev = []
                for meas_pair in meas_qbts_prev:
                    qbt, meas_outcome = meas_pair  # [copy.deepcopy(input_strats), [[output_qbt, "A"]], beginning_order]
                    if qbt == new_pauli_meas_qbt:
                        new_meas_qbts_prev.append([qbt, new_pauli_meas])
                    else:
                        new_meas_qbts_prev.append(meas_pair)
                new_tree[key].append(new_meas_qbts_prev)
                new_tree[key].append(third_item)
            # for key in tree.keys():
            #     print("Old tree: ", tree[key][1:])
            #     print()
            #     print("New tree: ", new_tree[key][1:])
            #    print()
            return new_tree
        else:
            return tree



    def update_identity_in_tree_meas_pattern(self, meas_qbts, new_strat, tree):
        new_pauli_measurement_qubits = new_strat[1]
        new_pauli_measurements = new_strat[3]
        new_pauli_meas = []
        new_pauli_meas_qbt = []
        flag = False
        new_tree = {}
        for meas_pair in meas_qbts:
            qbt, meas_outcome = meas_pair
            if meas_outcome == "I":
                if qbt in new_pauli_measurement_qubits:
                    new_pauli_meas.append(new_pauli_measurements[qbt])
                    new_pauli_meas_qbt.append(qbt)
                    flag = True
        if flag:
            for key in tree.keys():
                new_tree[key] = []
                first_item, meas_qbts_prev, third_item = tree[key]
                new_tree[key].append(first_item)
                new_meas_qbts_prev = []
                for meas_pair in meas_qbts_prev:
                    qbt, meas_outcome = meas_pair  # [copy.deepcopy(input_strats), [[output_qbt, "A"]], beginning_order]
                    if qbt in new_pauli_meas_qbt:
                        qbt_idx = new_pauli_meas_qbt.index(qbt)
                        new_meas_qbts_prev.append([qbt, new_pauli_meas[qbt_idx]])
                    else:
                        new_meas_qbts_prev.append(meas_pair)
                new_tree[key].append(new_meas_qbts_prev)
                new_tree[key].append(third_item)
            # for key in tree.keys():
            #     print("Old tree: ", tree[key][1:])
            #     print()
            #     print("New tree: ", new_tree[key][1:])
            #    print()
            return new_tree
        else:
            return tree

    def update_meas_qbts_in_loop(self, meas_qbts, stab_finished):
        new_meas_qbts = []
        new_pauli_measurement_qubits = stab_finished[1]
        new_pauli_measurements = stab_finished[3]
        for meas_pair in meas_qbts:
            qbt, meas_outcome = meas_pair
            if meas_outcome == "A":
                if qbt in new_pauli_measurement_qubits:
                    new_meas_qbts.append([qbt, new_pauli_measurements[qbt]])
                else:
                    new_meas_qbts.append(meas_pair)
            else:
                new_meas_qbts.append(meas_pair)
        return new_meas_qbts


    def update_identity_in_meas_qbts_in_loop(self, meas_qbts, stab_finished):
        new_meas_qbts = []
        new_pauli_measurement_qubits = stab_finished[1]
        new_pauli_measurements = stab_finished[3]
        for meas_pair in meas_qbts:
            qbt, meas_outcome = meas_pair
            if meas_outcome == "I":
                if qbt in new_pauli_measurement_qubits:
                    new_meas_qbts.append([qbt, new_pauli_measurements[qbt]])
                else:
                    new_meas_qbts.append(meas_pair)
            else:
                new_meas_qbts.append(meas_pair)
        return new_meas_qbts

    def update_valid_strategies(self, in_qubit, output_qbt, key, qbt, in_strats):
        new_strats = []
        for s in in_strats:
            new_strats.append(s)
        for qbt_out in range(len(self.gstate)):
            if qbt_out != in_qubit and qbt_out != output_qbt and qbt_out in list(self.output_qbt_counter.keys()):
                poss_further_strats = self.output_qbt_counter[qbt_out][:-1]
                lost_qbts = self.from_key_get_list_of_lost_qbt(key + "," + str(qbt))
                for l_qbt in lost_qbts:
                    poss_further_strats = self.filter_strat_lost_qubit(poss_further_strats, l_qbt)
                if len(poss_further_strats) > 0:
                    for strat in poss_further_strats:
                        new_strats.append(strat)
        return new_strats

    def get_numb_of_meas_qbts(self, pauli, qbts_meas, output_qbt, printing=False):
        numb = len(qbts_meas)
        if output_qbt not in qbts_meas:
            numb += 1
        for qbt, p in enumerate(pauli):
            if qbt != self.in_qubit and qbt != output_qbt and qbt not in qbts_meas:
            # if qbt != self.in_qubit and qbt not in qbts_meas:
                if p != "I":
                    numb += 1
                    if printing:
                        print(qbt, p)
        return numb


    def get_anticommuting_support(self, outcome_list, anti_support, list_of_anti_support, init_output_qbt):
        meas_qbts = outcome_list[1]
        this_traj_anti = []
        # print("outcome list: ", outcome_list[0])
        if len(outcome_list[0]) == 1:
            pauli = outcome_list[0][0][3]
            output_qbt = outcome_list[0][0][0][1]
        else:
            pauli = outcome_list[0][3]
            output_qbt = outcome_list[0][0][1]
        # print("Pauli: ", pauli, meas_qbts)
        this_traj_anti.append(init_output_qbt)
        if output_qbt not in this_traj_anti:
            this_traj_anti.append(output_qbt)
        if len(meas_qbts) != 0:
            if output_qbt not in anti_support:
                anti_support.append(output_qbt)
            for meas_pair in meas_qbts:
                qbt, meas_pauli = meas_pair
                if pauli[qbt] == "I" or pauli[qbt] == meas_pauli or qbt == output_qbt or meas_pauli == "I":
                    continue
                else:
                    this_traj_anti.append(qbt)
                    if qbt not in anti_support:
                        anti_support.append(qbt)

        list_of_anti_support.append(this_traj_anti)


    def from_key_get_lost_qbt(self, key):
        lost = 0
        for idx, char in enumerate(key):
            if idx != len(key) - 1:
                if char != "S" and char != "," and key[idx + 1] == ",":
                    lost += 1
            else:
                if char != "S" and char != ",":  # Double digit numbers is fixed here
                    lost += 1
        return lost



    def from_key_get_list_of_lost_qbt(self, key):
        lost = []
        for idx, char in enumerate(key):
            if idx != len(key) - 1:
                if char != "S" and char != ",": # and key[idx + 1] == ",":
                    if key[idx + 1] != ",":
                        lost.append(int(char + key[idx + 1]))
                    elif key[idx - 1] != ",":
                        continue
                    else:
                        lost.append(int(char))
            else:
                if key[idx - 1] != ",":
                    continue
                else:  # Double digit numbers is fixed here
                    lost.append(int(char))
        return lost


    def pick_qbt_to_measure(self, measuring_strat, input_all_strats, qbts_to_measure):
        largest_surv_strats = 0
        largest_surv_qbt = measuring_strat[1][0]
        flag_continue = True
        measure_order = []
        all_strats = copy.deepcopy(input_all_strats)
        while flag_continue:
            cnt = 0
            for qbt in qbts_to_measure:
                copied_all_strats = copy.deepcopy(all_strats)
                surv_strat = self.filter_strat_lost_qubit(copied_all_strats, qbt)
                if len(surv_strat) > largest_surv_strats:
                    largest_surv_strats = len(surv_strat)
                    largest_surv_qbt = qbt
                    cnt += 1
            if cnt > 0:
                measure_order.append(largest_surv_qbt)
                all_strats = self.filter_strat_lost_qubit(all_strats, largest_surv_qbt)
                largest_surv_qbt = 0
                if len(all_strats) == 0:
                    flag_continue = False

            else:
                flag_continue = False
        for qbt in qbts_to_measure:
            if qbt not in measure_order:
                measure_order.append(qbt)
        return measure_order



    def calculate_log_transmisison(self, transmission, analytic_exp):
        log_trans = []
        for trans in transmission:
            tot_trans = 0
            for trajectory in analytic_exp:
                loss_qbt, trans_qbt = trajectory
                tot_trans += (((1 - trans) ** loss_qbt) * (trans ** trans_qbt))
            log_trans.append(tot_trans)
        return log_trans


    def calculate_log_transmisison_from_min_loss_patt(self, transmission):
        numb_lost_patts = len(self.all_min_loss_patterns)
        min_qbts_lost = len(self.min_loss_pattern)
        n_total = len(self.gstate) - 1
        log_trans = []
        for trans in transmission:
            tot_trans = 0
            for n_qbt in range(min_qbts_lost + 1):
                pref = self.binom_coeff(n_total, n_qbt)
                if n_qbt == min_qbts_lost:
                    pref -= numb_lost_patts
                tot_trans += pref * (((1 - trans) ** n_qbt) * (trans ** (n_total - n_qbt)))
            log_trans.append(tot_trans)
        return log_trans

def single_qubit_commute(pauli1, pauli2, qbt):
    """
    Returns 0 if the operators on the qbt-th qubit of the two operators in the Pauli group commute,
    and 1 if they anticommute.
    """
    if pauli1[qbt] == 'I' or pauli2[qbt] == 'I' or pauli1[qbt] == pauli2[qbt]:
        return 0
    else:
        return 1


def same_qubit_pauli(pauli1, pauli2, qbt):
    """
    Returns 0 if the operators on the qbt-th qubit of the two operators are the same and not identity.
    """
    if (pauli1[qbt] != 'I' and pauli2[qbt] != 'I') and pauli1[qbt] == pauli2[qbt]:
        return 0

    elif (pauli1[qbt] == 'I' and pauli2[qbt] != 'I') or (pauli1[qbt] != 'I' and pauli2[qbt] == 'I'):
        return 1
    else:
        return 2



def get_commuting_patterns(possible_strats_dict, input_qbt, output_qbt, num_qbt, threshold_val=0):
    # TODO REMOVE ALL KEYS WITH ONLY ONE STARTEGY!
    commuting_qbts = []
    for qbt in range(num_qbt):
        if qbt != input_qbt and qbt != output_qbt:
            commuting_qbts.append(qbt)
    all_keys = list(possible_strats_dict.keys())
    lists_of_strats = []
    lists_of_strats.append(possible_strats_dict[all_keys[0]][0])
    cnt = 0
    all_anti_com_set = []
    while len(lists_of_strats) < len(all_keys):
        for key in all_keys[1:]:
            poss_strats_this_layer = possible_strats_dict[key]
            for strat in poss_strats_this_layer:
                pauli_strat = strat[3]
                flag_strat = True
                anti_com_set = []
                for app_strats in lists_of_strats:
                    pauli_app_strats = app_strats[3]
                    set_qbt = commute_loop_number_check(pauli_strat, pauli_app_strats, commuting_qbts)
                    for qbt in set_qbt:
                        if qbt not in anti_com_set:
                            anti_com_set.append(qbt)
                    if len(anti_com_set) > threshold_val:
                        flag_strat = False
                if flag_strat:
                    lists_of_strats.append(strat)
                    set_qbt = commute_loop_number_check(pauli_strat, pauli_app_strats, commuting_qbts)
                    for qbt in set_qbt:
                        if qbt not in all_anti_com_set:
                            all_anti_com_set.append(qbt)
                    break
        if len(lists_of_strats) < len(all_keys):
            cnt += 1
            if cnt == len(possible_strats_dict[all_keys[0]]):
                threshold_val += 1
                cnt = 0
                # return lists_of_strats
            lists_of_strats = []
            all_anti_com_set = []
            lists_of_strats.append(possible_strats_dict[all_keys[0]][cnt])
    return lists_of_strats, len(all_anti_com_set) # threshold_val



def commute_loop_number_check(pauli1, pauli2, qbts):
    anticommuting_qbts = []
    for qbt in qbts:
        commute_flag = single_qubit_commute(pauli1, pauli2, qbt)
        if commute_flag != 0:
            anticommuting_qbts.append(qbt)
    return anticommuting_qbts


def count_anti_commuting_qbt_sites(strats, input_qbt, output_qbt, num_qbt):
    anti_commuting_qbts = []
    pauli_support_qbts = []
    for qbt in range(num_qbt):
        if qbt != input_qbt and qbt != output_qbt:
            pauli_support_qbts.append(qbt)
    for qbt in pauli_support_qbts:
        compare_pauli_flag = False
        saved_idx = 0
        for idx, strat in enumerate(strats):
            pauli_string = strat[3]
            if not compare_pauli_flag and pauli_string[qbt] != "I":
                compare_pauli_flag = True
                compare_pauli = pauli_string[qbt]
                saved_idx = idx
            elif compare_pauli_flag:
                if pauli_string[qbt] == "I" or pauli_string[qbt] == compare_pauli:
                    continue
                else:
                    # print("Anticomm support!")
                    # print(strats[saved_idx])
                    # print(strat)
                    anti_commuting_qbts.append(qbt)
                    break
    return anti_commuting_qbts


if __name__ == '__main__':

    # TODO: Fix the manual setting of the output qubit and add this to the heuristic
    from CodeFunctions.graphs import *
    import matplotlib.pyplot as plt
    from itertools import permutations
    from ErasureDecoder import LT_Erasure_decoder
    # from DelayedMeasDecoder import LT_FullDecoder_QND

    #######################################################################
    ########################### EIGHT QBTS ################################
    #######################################################################

    # n_qbts = 7
    # distance = 2
    # in_qubit = 0
    # graph_nodes = list(range(8))
    # graph_edges = [(0, 4), (0, 5), (0, 7), (1, 3), (1, 4), (1, 5), (2, 3),
    #                (2, 5), (2, 7), (3, 6), (4, 6), (6, 7)]
    # patterns = list(permutations(range(1, n_qbts + 1), n_qbts))

    #######################################################################
    ########################### ELEVEN QBTS ###############################
    #######################################################################

    n_qbts = 10
    in_qubit = 0
    distance = 3
    graph_nodes = list(range(11))
    graph_edges = [(10, 1), (10, 2), (10, 5), (10, 4), (10, 3), (0, 10), (10, 6), (10, 8),
                   (1, 2), (1, 5), (1, 9), (1, 8), (2, 7), (2, 5), (3, 4), (3, 5),
                   (3, 9), (3, 6), (4, 5), (4, 7), (4, 0), (6, 8), (6, 9), (6, 7),
                   (7, 0), (8, 9), (9, 0)]
    patterns = list(permutations(range(1, n_qbts + 1), n_qbts))
    gstate = graph_from_nodes_and_edges(graph_nodes, graph_edges)# graphstate_from_nodes_and_edges(graph_nodes, graph_edges)

    # TESTING DIFFERENT DECODERS
    erasure_decoder = LT_Erasure_decoder(n_qbts, distance, gstate, in_qbt=in_qubit)
    input_strats = erasure_decoder.strategies_ordered

    erasure_decoder_no_matt = LT_Erasure_decoder(n_qbts, 2, gstate, in_qbt=in_qubit)
    input_strats_no_matt = erasure_decoder_no_matt.strategies_ordered

    # print(input_strats)
    # print("Number of strategies: ", len(input_strats))
    # print("Clash counter: ", erasure_decoder.cnt)
    # print("Clashed strategies (i.e. more than 1 anticomm. qbt): ", erasure_decoder.clashed_strats)
    no_anti_com_flag = False
    same = 0
    less = 0
    for m_idx, meas_pattern in enumerate(patterns[:1]):
        if m_idx % 500 == 0:
            print("At meas. pattern: ", m_idx)
            print(same)
        meas_pattern_list = [qbt for qbt in meas_pattern]
        adaptive_decoder = LT_FullHybridDecoder(copy.deepcopy(gstate), copy.deepcopy(input_strats), measurement_order=copy.deepcopy(meas_pattern_list), no_anti_com_flag=no_anti_com_flag, printing=False)
        adaptive_decoder_new = LT_FullHybridDecoderNew(copy.deepcopy(gstate), copy.deepcopy(input_strats), measurement_order=copy.deepcopy(meas_pattern_list), no_anti_com_flag=no_anti_com_flag, printing=False)
        adaptive_decoder_new_no_extra_matt = LT_FullHybridDecoderNew(copy.deepcopy(gstate), copy.deepcopy(input_strats_no_matt),
                                                       measurement_order=copy.deepcopy(meas_pattern_list),
                                                       no_anti_com_flag=True, printing=False)
        # print("Tree branches: ")
        # for branch in adaptive_decoder.tree_branches:
        #     print(branch)
        # print("All loss_patts with minimal support: ", len(adaptive_decoder.all_min_loss_patterns))
        old_dec_matt_qbts = adaptive_decoder.max_number_of_m_dec
        new_dec_matt_qbts = adaptive_decoder_new.max_number_of_m_dec
        # if old_dec_matt_qbts < 3 or new_dec_matt_qbts < 3:
        # if old_dec_matt_qbts < 5 or new_dec_matt_qbts < 5:
        # if old_dec_matt_qbts != new_dec_matt_qbts:
            # print(meas_pattern)
        print("Number of matter qbts: ", old_dec_matt_qbts )
        print("Number of matter new qbts: ", new_dec_matt_qbts)
        print()
        #     less += 1
        # elif new_dec_matt_qbts == 5:
        #     same += 1
        # else:
        #     same += 1

        # print(less, same)
    '''
    for patt in adaptive_decoder.all_min_loss_patterns:
        if len(patt) < 6:
            print(patt)
    '''
    t_list = np.linspace(0.99, 0.9999, 100)
    log_trans = adaptive_decoder_new.calculate_log_transmisison(t_list, adaptive_decoder_new.analytic_exp) # adaptive_decoder.calculate_log_transmisison_from_min_loss_patt(t_list)
    # log_trans = adaptive_decoder_new_no_extra_matt.calculate_log_transmisison(t_list, adaptive_decoder_new_no_extra_matt.analytic_exp)
    print(log_trans[:10])
    # print(adaptive_decoder.calculate_log_transmisison_from_min_loss_patt([0.9999]))
    loss_list = [1 - t for t in t_list]
    scaling_4 = [(1 - t) ** 4 for t in t_list]
    scaling_3 = [(1 - t) ** 3 for t in t_list]
    log_loss_hybrid = [1 - t for t in log_trans]
    plt.plot(loss_list, log_loss_hybrid, color="red", label="Code")
    plt.plot(loss_list, scaling_4, "k--", label="l=4")
    plt.plot(loss_list, scaling_3, "--", label="l=3", color="blue")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Physical loss")
    plt.ylabel("Logical loss")
    plt.legend()
    plt.savefig("10_qbt_graph_logical_loss.png")
    plt.show()
