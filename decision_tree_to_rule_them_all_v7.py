"""
Decision tree construction and plotting for quantum error correction strategies.

Replaces the `treelib` dependency with a lightweight built-in tree structure.
"""

import copy
from typing import Any


# ---------------------------------------------------------------------------
# Lightweight tree (replaces treelib)
# ---------------------------------------------------------------------------

class SimpleTree:
    """Minimal tree backed by dictionaries – no external dependencies."""

    def __init__(self):
        self._parent: dict[str, str | None] = {}
        self._children: dict[str, list[str]] = {}
        self._tags: dict[str, str] = {}

    def create_node(self, tag: str, key: str, parent: str | None = None):
        self._tags[key] = tag
        self._parent[key] = parent
        self._children.setdefault(key, [])
        if parent is not None:
            self._children.setdefault(parent, [])
            self._children[parent].append(key)

    def contains(self, key: str) -> bool:
        return key in self._tags

    def children(self, key: str) -> list[str]:
        return self._children.get(key, [])

    def parent_key(self, key: str) -> str | None:
        return self._parent.get(key)

    def all_node_keys(self) -> list[str]:
        return list(self._tags.keys())


# ---------------------------------------------------------------------------
# Key / string parsing helpers
# ---------------------------------------------------------------------------

def lost_qubits_from_key(key: str) -> list[int]:
    """Extract the list of lost qubit indices encoded in a comma-separated key."""
    lost: list[int] = []
    chars = list(key)
    n = len(chars)
    for i, ch in enumerate(chars):
        if ch in ("S", ","):
            continue
        # Try to form a (possibly multi-digit) number
        if i < n - 1 and chars[i + 1] not in (",", "S", "l"):
            # Two-digit number: current + next
            lost.append(int(ch + chars[i + 1]))
        elif i > 0 and chars[i - 1] not in (",", "S", "l"):
            # Second digit of a two-digit number already handled
            continue
        else:
            lost.append(int(ch))
    return lost


def _build_parent_key(init_parent: str, ix: int, total_lost: list[int], meas_order: list[int]) -> str:
    """Reconstruct the parent key string for node at position *ix* in *meas_order*."""
    parent = init_parent
    for qbt_idx in range(1, ix - 1):
        meas_qbt = meas_order[qbt_idx]
        suffix = f",{meas_qbt}l" if meas_qbt in total_lost else f",{meas_qbt}"
        parent += suffix
    return parent


# ---------------------------------------------------------------------------
# Strategy parsing
# ---------------------------------------------------------------------------

def parse_strategy(strat) -> tuple[Any, str]:
    """Return (output_qubit, human-readable parsed strategy string)."""
    if len(strat) < 2:
        output_qbt = strat[0][0][1]
        meas_qbts = strat[0][1]
        pauli_string = strat[0][3]
    else:
        output_qbt = strat[0][1]
        meas_qbts = strat[1]
        pauli_string = strat[3]

    parsed = "_".join(f"{pauli_string[q]}{q}" for q in meas_qbts)
    parsed += f"_A{output_qbt}"
    return output_qbt, parsed


def parse_measured_string(meas_qbts, ix: int, output_qbt, meas_patt, lost_qbts) -> str:
    """Build a string listing qubits measured so far (excluding lost ones)."""
    parts: list[str] = []
    for idx in range(ix):
        qbt = meas_qbts[idx]
        if qbt in lost_qbts:
            continue
        parts.append(str(qbt))
    return "_".join(parts) + ("_" if parts else "")


def parse_measured_string_key(key: str) -> str:
    """Parse a node-name key into a measured-qubit summary string."""
    parts: list[str] = []
    n = len(key)
    for i, ch in enumerate(key):
        if ch == "l":
            continue
        if i < n - 1 and key[i + 1] == ",":
            parts.append(ch)
        elif i == n - 1:
            parts.append(ch)
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Tree construction helpers (shared logic)
# ---------------------------------------------------------------------------

def _extract_trajectory_fields(traj, short_format: bool):
    """Unpack common fields from a trajectory entry."""
    strat = traj[0]
    out_q, strat_parsed = parse_strategy(strat)

    if len(strat) < 2:
        output_qbt = strat[0][0][1]
        to_be_meas_qbts = strat[0][1]
        meas_patt = strat[0][3]
    else:
        output_qbt = strat[0][1]
        to_be_meas_qbts = strat[1]
        meas_patt = strat[3]

    return out_q, strat_parsed, output_qbt, to_be_meas_qbts, meas_patt


def _ensure_meas_order_complete(meas_order, output_qbt, to_be_meas_qbts):
    """Append output / measurement qubits to *meas_order* if missing."""
    if output_qbt not in meas_order:
        meas_order.append(output_qbt)
    for qbt in to_be_meas_qbts:
        if qbt not in meas_order:
            meas_order.append(qbt)


# ---------------------------------------------------------------------------
# Core: build the first trajectory branch (idx == 0)
# ---------------------------------------------------------------------------

def _build_first_branch(
    tree: SimpleTree,
    traj,
    node_information: dict,
    node_name_idx: dict,
    leaf_nodes_idx: list,
    node_cnt: int,
    start_key: str,
    meas_order_offset: int,
) -> int:
    meas_order = traj[meas_order_offset]
    (out_q, strat_parsed, output_qbt,
     to_be_meas_qbts, meas_patt) = _extract_trajectory_fields(traj, short_format=False)

    lost_output_qbts = traj[meas_order_offset + 1]
    lost_qbts = lost_qubits_from_key(traj[-2])
    anti_commuting = traj[-1]
    actual_meas_pairs = traj[1]  # list of [qubit, pauli_basis] pairs

    _ensure_meas_order_complete(meas_order, output_qbt, to_be_meas_qbts)

    total_lost = list(lost_output_qbts) + lost_qbts

    # Root's first child
    qbt_to_measure = meas_order[1]
    node_information[node_cnt] = [
        out_q, strat_parsed, lost_qbts, [],
        qbt_to_measure, str(output_qbt), start_key, anti_commuting, [],
    ]
    node_name_idx[str(output_qbt)] = node_cnt
    tree.create_node(str(output_qbt), str(output_qbt), parent=start_key)
    node_cnt += 1

    for ix, qbt in enumerate(meas_order):
        if ix == 0:
            continue
        key = ",".join(str(meas_order[j]) for j in range(ix))
        full_key = f"{key},{qbt}"
        measured_info = parse_measured_string(meas_order, ix, output_qbt, meas_patt, lost_qbts)
        tree.create_node(full_key, full_key, parent=key)

        is_leaf = (ix == len(meas_order) - 1)
        next_qbt = "Success" if is_leaf else meas_order[ix + 1]

        if is_leaf:
            leaf_nodes_idx.append(node_cnt)

        node_information[node_cnt] = [
            out_q, strat_parsed, lost_qbts, measured_info,
            next_qbt, full_key, key, anti_commuting,
            actual_meas_pairs if is_leaf else [],
        ]
        node_name_idx[full_key] = node_cnt
        node_cnt += 1

    return node_cnt


# ---------------------------------------------------------------------------
# Core: build subsequent trajectory branches (idx > 0)
# ---------------------------------------------------------------------------

def _build_subsequent_branch(
    tree: SimpleTree,
    traj,
    node_information: dict,
    node_name_idx: dict,
    leaf_nodes_idx: list,
    node_cnt: int,
    start_key: str,
    meas_order_offset: int,
) -> int:
    meas_order = traj[meas_order_offset]
    (out_q, strat_parsed, output_qbt,
     to_be_meas_qbts, meas_patt) = _extract_trajectory_fields(traj, short_format=False)

    lost_output_qbts = traj[meas_order_offset + 1]
    lost_qbts = lost_qubits_from_key(traj[-2])
    anti_commuting = traj[-1]
    actual_meas_pairs = traj[1]  # list of [qubit, pauli_basis] pairs

    _ensure_meas_order_complete(meas_order, output_qbt, to_be_meas_qbts)

    total_lost = list(lost_output_qbts) + lost_qbts
    init_qbt = meas_order[0]
    init_parent = f"{init_qbt}l" if init_qbt in total_lost else str(init_qbt)

    for ix, qbt in enumerate(meas_order):
        if ix == 0:
            continue

        # Build key up to (but not including) the current qubit
        key = init_parent
        for qbt_idx in range(1, ix):
            meas_qbt = meas_order[qbt_idx]
            suffix = f",{meas_qbt}l" if meas_qbt in total_lost else f",{meas_qbt}"
            key += suffix

        measured_info = parse_measured_string(meas_order, ix, output_qbt, meas_patt, lost_qbts)

        if not tree.contains(key):
            parent = start_key if ix == 1 else _build_parent_key(init_parent, ix, total_lost, meas_order)
            node_information[node_cnt] = [
                out_q, strat_parsed, lost_qbts, measured_info,
                qbt, key, parent, anti_commuting, [],
            ]
            node_name_idx[key] = node_cnt
            node_cnt += 1
            tree.create_node(key, key, parent=parent)

        elif len(tree.children(key)) == 0:
            if qbt in total_lost:
                child_key = f"{key},{qbt}l"
                tree.create_node(child_key, child_key, parent=key)
                node_information[node_cnt] = [
                    out_q, strat_parsed, lost_qbts, measured_info,
                    qbt, child_key, key, anti_commuting, [],
                ]
                node_name_idx[child_key] = node_cnt
                node_cnt += 1
            else:
                child_key = f"{key},{qbt}"
                tree.create_node(child_key, child_key, parent=key)
                node_information[node_cnt] = [
                    out_q, strat_parsed, lost_qbts, measured_info,
                    qbt, child_key, key, anti_commuting, [],
                ]
                node_name_idx[child_key] = node_cnt
                node_cnt += 1

        # Leaf handling
        if ix == len(meas_order) - 1:
            parent_key = copy.deepcopy(key)
            meas_qbt = meas_order[ix]
            leaf_suffix = f",{meas_qbt}l" if meas_qbt in total_lost else f",{meas_qbt}"
            leaf_key = key + leaf_suffix
            tree.create_node(leaf_key, leaf_key, parent=parent_key)
            node_information[node_cnt] = [
                out_q, strat_parsed, lost_qbts, measured_info,
                "Success", leaf_key, parent_key, anti_commuting,
                actual_meas_pairs,
            ]
            node_name_idx[leaf_key] = node_cnt
            node_cnt += 1
            leaf_nodes_idx.append(node_cnt - 1)

    return node_cnt


# ---------------------------------------------------------------------------
# Public API: build trees
# ---------------------------------------------------------------------------

def get_tree(input_list):
    """Build decision tree from trajectory list (adaptive measurement order)."""
    tree = SimpleTree()
    start_key = "Start"
    tree.create_node(start_key, start_key)

    traj0 = input_list[0]
    out_q, strat_parsed = parse_strategy(traj0[0])
    output_qbt = traj0[0][0][0][1] if len(traj0[0]) < 2 else traj0[0][0][1]

    node_information = {0: [out_q, strat_parsed, [], [], output_qbt, [], "n", "n", []]}
    node_name_idx = {start_key: 0}
    leaf_nodes_idx: list[int] = []
    node_cnt = 1

    # meas_order lives at traj[-4] for adaptive variant
    MEAS_ORDER_OFFSET = -4

    for idx, traj in enumerate(input_list):
        if idx == 0:
            node_cnt = _build_first_branch(
                tree, traj, node_information, node_name_idx,
                leaf_nodes_idx, node_cnt, start_key, MEAS_ORDER_OFFSET,
            )
        else:
            node_cnt = _build_subsequent_branch(
                tree, traj, node_information, node_name_idx,
                leaf_nodes_idx, node_cnt, start_key, MEAS_ORDER_OFFSET,
            )

    return tree, node_information, node_name_idx, leaf_nodes_idx


def get_tree_fixed_meas_patt(input_list):
    """Build decision tree from trajectory list (fixed measurement pattern)."""
    tree = SimpleTree()
    start_key = "Start"
    tree.create_node(start_key, start_key)

    traj0 = input_list[0]
    out_q, strat_parsed = parse_strategy(traj0[0])
    output_qbt = traj0[0][0][0][1] if len(traj0[0]) < 2 else traj0[0][0][1]

    node_information = {0: [out_q, strat_parsed, [], [], output_qbt, [], "n", "n", []]}
    node_name_idx = {start_key: 0}
    leaf_nodes_idx: list[int] = []
    node_cnt = 1

    # meas_order lives at traj[-5] for fixed-pattern variant
    MEAS_ORDER_OFFSET = -5

    for idx, traj in enumerate(input_list):
        if idx == 0:
            node_cnt = _build_first_branch(
                tree, traj, node_information, node_name_idx,
                leaf_nodes_idx, node_cnt, start_key, MEAS_ORDER_OFFSET,
            )
        else:
            node_cnt = _build_subsequent_branch(
                tree, traj, node_information, node_name_idx,
                leaf_nodes_idx, node_cnt, start_key, MEAS_ORDER_OFFSET,
            )

    return tree, node_information, node_name_idx, leaf_nodes_idx


# ---------------------------------------------------------------------------
# Capture + Delay computation (post-tree-build)
# ---------------------------------------------------------------------------

def _parse_pauli_map_from_strat(strat_str: str) -> dict[str, str]:
    """Parse 'Z1_Z3_X4_Z5_A6' → {'1':'Z', '3':'Z', '4':'X', '5':'Z', '6':'A'}"""
    pauli_map = {}
    if not strat_str or strat_str == "—":
        return pauli_map
    for part in strat_str.split("_"):
        if not part:
            continue
        pauli = part[0]
        qbt = part[1:]
        if qbt:
            pauli_map[qbt] = pauli
    return pauli_map


def compute_max_capture_delay(
    tree: SimpleTree,
    node_information: dict,
    node_name_idx: dict,
    leaf_node_idxs: list,
    printing: bool = False,
) -> tuple[int, list]:
    """Derive measurement bases and capture+delay from tree structure + strategies.

    Uses 3 passes matching the viewer:
      1. Top-down: build meas_record per node from strategy, upgrade "I" where possible
      2. Bottom-up: resolve remaining "I" from children (only pre-branch qubits)
      3. Top-down push: enforce parent's resolved values on children (only pre-branch qubits)
    Then compute capture+delay at leaves.

    Returns (max_count, all_leaf_results).
    """

    def _get_pauli_map(node_id: int) -> dict:
        return _parse_pauli_map_from_strat(str(node_information[node_id][1]))

    def _get_output_qbt(node_id: int) -> str:
        return str(node_information[node_id][0])

    # Per-key storage
    records = {}      # key -> meas_record dict
    own_qubits = {}   # key -> set of qubits in record before bottom-up
    pauli_maps = {}   # key -> pauli_map
    output_qbts = {}  # key -> output qubit

    # ── Pass 1: top-down ──
    def pass1(key, parent_rec):
        node_id = node_name_idx[key]
        pm = _get_pauli_map(node_id)
        pauli_maps[key] = pm
        output_qbts[key] = _get_output_qbt(node_id)
        is_leaf = node_id in leaf_node_idxs

        rec = dict(parent_rec)
        # Add qubits from key, using this node's strategy
        if key and key != "Start":
            for part in key.split(","):
                q = part.rstrip("l")
                if q not in rec:
                    b = pm.get(q, "I")
                    rec[q] = b if b != "I" else "I"
                elif rec[q] == "I":
                    b = pm.get(q, "I")
                    if b != "I":
                        rec[q] = b

        # Leaves: supplement with actual_meas
        if is_leaf:
            raw_pairs = node_information[node_id][8] if len(node_information[node_id]) > 8 else []
            for pair in raw_pairs:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    q, b = str(pair[0]), str(pair[1])
                    if b != "I" and (q not in rec or rec[q] == "I"):
                        rec[q] = b

        records[key] = rec
        own_qubits[key] = set(rec.keys())

        for child_key in tree.children(key):
            pass1(child_key, rec)

    # ── Pass 2: bottom-up ──
    def pass2(key):
        for child_key in tree.children(key):
            pass2(child_key)
        rec = records[key]
        own = own_qubits[key]
        for child_key in tree.children(key):
            child_rec = records[child_key]
            for q, b in child_rec.items():
                if b and b != "I" and q in own and rec[q] == "I":
                    rec[q] = b

    # ── Pass 3: top-down enforce ──
    def pass3(key, parent_rec, pre_branch):
        rec = records[key]
        for q in pre_branch:
            b = parent_rec.get(q)
            if b and b != "I" and rec.get(q) != b:
                rec[q] = b
        for child_key in tree.children(key):
            pass3(child_key, rec, own_qubits[key])

    # Find root
    root_key = None
    for key in tree.all_node_keys():
        if tree.parent_key(key) is None:
            root_key = key
            break

    pass1(root_key, {})
    pass2(root_key)
    pass3(root_key, {}, set())

    # ── Compute capture+delay at leaves ──
    all_leaf_results = []
    for key, rec in records.items():
        node_id = node_name_idx[key]
        if node_id not in leaf_node_idxs:
            continue
        pm = pauli_maps[key]
        output_qbt = output_qbts[key]

        capture_delay = [output_qbt]
        conflicts = {}
        for q, meas_b in rec.items():
            strat_b = pm.get(q, "")
            if strat_b and meas_b != strat_b and meas_b != "I" and strat_b != "I":
                conflicts[q] = {"measured": meas_b, "needed": strat_b}
                if q not in capture_delay:
                    capture_delay.append(q)

        all_leaf_results.append({
            "key": key,
            "output_qbt": output_qbt,
            "capture_delay": capture_delay,
            "conflicts": conflicts,
            "meas_record": rec,
            "strategy": str(node_information[node_id][1]),
        })

    max_count = max((len(r["capture_delay"]) for r in all_leaf_results), default=0)

    if printing:
        for r in all_leaf_results:
            cd = r["capture_delay"]
            print(f"Leaf {r['key']}: capture+delay={cd} ({len(cd)} qubits), "
                  f"conflicts={r['conflicts']}")
        print(f"\nMax capture+delay: {max_count}")

    return max_count, all_leaf_results


# ---------------------------------------------------------------------------
# Plotting (requires networkx, pydot, matplotlib)
# ---------------------------------------------------------------------------

def _hierarchy_pos(G, root, width=1.0, vert_gap=0.2, xcenter=0.5):
    """Compute a top-down tree layout (Buchheim-style) using only networkx.

    Returns a dict  {node: (x, y)}  suitable for ``nx.draw``.
    """
    pos = {}

    def _subtree_width(node):
        children = list(G.successors(node))
        if not children:
            return 1
        return sum(_subtree_width(c) for c in children)

    def _assign(node, left, depth):
        children = list(G.successors(node))
        w = _subtree_width(node)
        pos[node] = (left + w / 2, -depth)
        offset = left
        for child in children:
            cw = _subtree_width(child)
            _assign(child, offset, depth + 1)
            offset += cw

    total_w = _subtree_width(root)
    _assign(root, 0, 0)
    # Normalise so the tree is centred and fits in a reasonable range
    for n in pos:
        x, y = pos[n]
        pos[n] = (x / total_w * width + xcenter - width / 2, y * vert_gap)
    return pos


def plot_decision_tree(
    tree: SimpleTree,
    node_information,
    node_name_idx,
    leaf_node_idxs,
    filename="Test",
):
    """Render the decision tree to a PNG using only networkx + matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    G = nx.DiGraph()
    root_node = None

    for key in tree.all_node_keys():
        node_idx = node_name_idx[key]
        G.add_node(node_idx)
        parent = tree.parent_key(key)
        if parent is not None:
            G.add_edge(node_name_idx[parent], node_idx)
        else:
            root_node = node_idx

    # Build labels, colours, edge colours
    labels = {}
    node_colors = []
    edge_colors = []

    for node_id in G.nodes():
        info = node_information[node_id]
        measured_str = parse_measured_string_key(info[5]) if isinstance(info[5], str) else ""

        if info[4] == "Success":
            labels[node_id] = (
                f"Strat {info[1]}\n"
                f"Lost {info[2]}\n"
                f"Meas {measured_str}\n"
                f"Out {info[0]}\n"
                f"Del.+Cap {info[-1]}\n"
                f"Success"
            )
        else:
            labels[node_id] = (
                f"Strat {info[1]}\n"
                f"Lost {info[2]}\n"
                f"Meas {measured_str}\n"
                f"Out {info[0]}\n"
                f"Try {info[4]}"
            )
        node_colors.append("lightgreen" if node_id in leaf_node_idxs else "lightskyblue")

    for u, v in G.edges():
        last_char = node_information[v][-3][-1]
        edge_colors.append("red" if last_char == "l" else "darkgreen")

    # Layout
    pos = _hierarchy_pos(G, root_node, width=10, vert_gap=1.0)

    fig, ax = plt.subplots(figsize=(max(14, len(G.nodes()) * 1.5), max(8, len(G.nodes()) * 0.6)))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, arrows=True, arrowsize=15, width=1.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=3000, edgecolors="black", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=5)

    ax.set_title(filename, fontsize=14)
    ax.axis("off")

    # Legend
    lost_patch = mpatches.Patch(color="red", label="Lost qubit edge")
    ok_patch = mpatches.Patch(color="darkgreen", label="OK qubit edge")
    leaf_patch = mpatches.Patch(color="lightgreen", label="Success (leaf)")
    ax.legend(handles=[lost_patch, ok_patch, leaf_patch], loc="upper left", fontsize=8)

    plt.tight_layout()
    output_file = f"{filename}.png"
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Interactive HTML viewer (self-contained, opens in browser / PyCharm)
# ---------------------------------------------------------------------------

def _build_tree_json(
    tree: SimpleTree,
    node_information: dict,
    node_name_idx: dict,
    leaf_node_idxs: list,
) -> dict:
    """Convert the tree into a nested dict ready for JSON serialisation."""

    def _parse_pauli_map(strat_str: str) -> dict:
        """Parse 'Z1_Z3_X4_Z5_A6' → {'1':'Z', '3':'Z', '4':'X', '5':'Z', '6':'A'}"""
        pauli_map = {}
        if not strat_str or strat_str == "—":
            return pauli_map
        for part in strat_str.split("_"):
            if not part:
                continue
            # First char(s) are the Pauli, rest is the qubit number
            # Handle single-letter Pauli (X/Y/Z/A) followed by qubit number
            pauli = part[0]
            qbt = part[1:]
            if qbt:
                pauli_map[qbt] = pauli
        return pauli_map

    def _build_subtree(key: str) -> dict:
        node_id = node_name_idx[key]
        info = node_information[node_id]
        # info layout (9 elements):
        #  [0] out_q, [1] strat_parsed, [2] lost_qbts, [3] measured_info,
        #  [4] next_qbt, [5] node_key, [6] parent_key, [7] anti_commuting,
        #  [8] actual_meas_pairs (list of [qubit, basis] — populated for leaves)
        measured_str = parse_measured_string_key(info[5]) if isinstance(info[5], str) else ""
        is_leaf = node_id in leaf_node_idxs
        is_lost = isinstance(info[5], str) and info[5].endswith("l")

        strat_str = str(info[1])
        pauli_map = _parse_pauli_map(strat_str)
        next_qbt = str(info[4])
        # Determine the Pauli basis for the next qubit to measure
        next_basis = pauli_map.get(next_qbt, "") if next_qbt != "Success" else ""

        # Build actual measurement bases dict: {"1": "Z", "3": "X", ...}
        # traj[1] only has qubits measured during the adaptive loop.
        # Merge with pauli_map to get the full picture (strategy fills in the rest).
        actual_meas = {}
        raw_pairs = info[8] if len(info) > 8 else []
        if raw_pairs:
            for pair in raw_pairs:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    actual_meas[str(pair[0])] = str(pair[1])
        if is_leaf:
            # Fill in remaining qubits from the strategy's Pauli map
            for q, b in pauli_map.items():
                if q not in actual_meas:
                    actual_meas[q] = b

        node_data = {
            "id": node_id,
            "key": key,
            "strategy": strat_str,
            "pauli_map": pauli_map,
            "actual_meas": actual_meas,
            "lost_qubits": str(info[2]),
            "measured_qubits": measured_str,
            "output_qubit": str(info[0]),
            "next_action": next_qbt,
            "next_basis": next_basis,
            "is_leaf": is_leaf,
            "is_lost_edge": is_lost,
            "anti_commuting": str(info[7]) if is_leaf else "",
            "children": [],
        }
        for child_key in tree.children(key):
            node_data["children"].append(_build_subtree(child_key))
        return node_data

    root_key = None
    for key in tree.all_node_keys():
        if tree.parent_key(key) is None:
            root_key = key
            break

    return _build_subtree(root_key)


def show_interactive_tree(
    tree: SimpleTree,
    node_information: dict,
    node_name_idx: dict,
    leaf_node_idxs: list,
    filename: str = "decision_tree_viewer.html",
    open_browser: bool = True,
):
    """Generate a self-contained HTML file with the tree data baked in and open it.

    Usage::

        tree, info, idx_map, leaves = get_tree(branches)
        show_interactive_tree(tree, info, idx_map, leaves)

    The HTML has no external dependencies (D3 is loaded from a CDN) and works
    when opened from PyCharm, VS Code, or any file manager.

    Parameters
    ----------
    filename : str
        Where to write the HTML file (default: ``decision_tree_viewer.html``).
    open_browser : bool
        If *True* (default), automatically open the file in the default browser.
    """
    import json
    import os
    import webbrowser

    tree_json = _build_tree_json(tree, node_information, node_name_idx, leaf_node_idxs)
    json_str = json.dumps(tree_json, indent=2)

    html = _HTML_TEMPLATE.replace("/* __TREE_DATA__ */", f"const TREE_DATA = {json_str};")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = os.path.abspath(filename)
    print(f"Viewer written to {abs_path}")

    if open_browser:
        webbrowser.open(f"file://{abs_path}")


# ---------------------------------------------------------------------------
# HTML template (everything in one string – no separate file needed)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Quantum Decision Tree</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=DM+Sans:wght@400;500;600;700&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --bg-deep: #0a0e1a;
    --bg-surface: #111827;
    --bg-card: #1a2235;
    --bg-card-hover: #1f2a40;
    --border: #2a3550;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-cyan: #22d3ee;
    --accent-emerald: #34d399;
    --accent-rose: #fb7185;
    --accent-amber: #fbbf24;
    --accent-violet: #a78bfa;
    --edge-ok: #34d399;
    --edge-lost: #fb7185;
  }

  body {
    background: var(--bg-deep);
    color: var(--text-primary);
    font-family: 'DM Sans', sans-serif;
    overflow: hidden;
    height: 100vh;
  }

  .header {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 100;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 24px;
    background: linear-gradient(180deg, rgba(10,14,26,0.97) 0%, rgba(10,14,26,0.85) 100%);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 14px;
  }

  .header-logo {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet));
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600; font-size: 14px;
    color: var(--bg-deep);
  }

  .header h1 {
    font-size: 16px; font-weight: 600;
    letter-spacing: -0.3px;
    color: var(--text-primary);
  }

  .header h1 span { color: var(--text-muted); font-weight: 400; }

  .header-actions { display: flex; gap: 8px; align-items: center; }

  .btn {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px; font-weight: 500;
    padding: 7px 14px; border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--bg-card); color: var(--text-secondary);
    cursor: pointer; transition: all 0.15s;
    display: flex; align-items: center; gap: 6px;
  }

  .btn:hover {
    background: var(--bg-card-hover);
    color: var(--text-primary);
    border-color: var(--accent-cyan);
  }

  .btn svg { width: 14px; height: 14px; }

  .stats-bar {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    z-index: 100;
    display: flex; align-items: center; gap: 24px;
    padding: 8px 24px;
    background: linear-gradient(0deg, rgba(10,14,26,0.97) 0%, rgba(10,14,26,0.85) 100%);
    backdrop-filter: blur(12px);
    border-top: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--text-muted);
  }

  .stat-item { display: flex; align-items: center; gap: 6px; }
  .stat-dot { width: 6px; height: 6px; border-radius: 50%; }
  .stat-value { color: var(--text-secondary); font-weight: 500; }

  .canvas-area { position: fixed; top: 52px; bottom: 28px; left: 0; right: 0; }
  .canvas-area svg { width: 100%; height: 100%; }
  .grid-bg { fill: url(#gridPattern); opacity: 0.3; }

  .node-group { cursor: pointer; }
  .node-group:hover .node-rect { filter: brightness(1.3); }

  .node-rect {
    rx: 10; ry: 10;
    stroke-width: 1.5;
    transition: filter 0.2s;
  }

  .node-label {
    font-family: 'JetBrains Mono', monospace;
    fill: var(--text-primary); font-size: 10px;
    text-anchor: middle; pointer-events: none;
  }

  .node-sublabel {
    font-family: 'DM Sans', sans-serif;
    fill: var(--text-muted); font-size: 9px;
    text-anchor: middle; pointer-events: none;
  }

  .link-path {
    fill: none; stroke-width: 2; opacity: 0.7;
    transition: opacity 0.2s, stroke-width 0.2s;
  }

  .link-path:hover { opacity: 1; stroke-width: 3; }
  .link-path.lost { stroke: var(--edge-lost); }
  .link-path.ok { stroke: var(--edge-ok); }

  .arrow-ok { fill: var(--edge-ok); }
  .arrow-lost { fill: var(--edge-lost); }

  .collapse-indicator {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; fill: var(--accent-amber);
    text-anchor: middle; pointer-events: none;
  }

  .tooltip {
    position: fixed; z-index: 200;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px; padding: 16px;
    min-width: 280px; max-width: 400px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
    pointer-events: none;
    opacity: 0; transform: translateY(6px);
    transition: opacity 0.2s, transform 0.2s;
    font-size: 12px;
  }

  .tooltip.visible { opacity: 1; transform: translateY(0); }

  .tooltip-title {
    font-weight: 600; font-size: 13px; margin-bottom: 10px;
    color: var(--accent-cyan);
    font-family: 'JetBrains Mono', monospace;
  }

  .tooltip-row {
    display: flex; justify-content: space-between;
    padding: 4px 0; border-bottom: 1px solid rgba(42,53,80,0.5);
  }

  .tooltip-row:last-child { border-bottom: none; }
  .tooltip-key { color: var(--text-muted); font-size: 11px; }

  .tooltip-val {
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; text-align: right;
    max-width: 180px; overflow: hidden; text-overflow: ellipsis;
  }

  .tooltip-val.success { color: var(--accent-emerald); }
  .tooltip-val.lost { color: var(--accent-rose); }

  .legend {
    position: fixed; bottom: 44px; right: 16px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 16px;
    font-size: 11px; z-index: 90;
  }

  .legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .legend-item:last-child { margin-bottom: 0; }
  .legend-swatch { width: 22px; height: 4px; border-radius: 2px; }
  .legend-circle { width: 10px; height: 10px; border-radius: 3px; }
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <div class="header-logo">QT</div>
    <h1>Decision Tree <span>Viewer</span></h1>
  </div>
  <div class="header-actions">
    <button class="btn" id="btn-fit">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>
      Fit
    </button>
    <button class="btn" id="btn-expand">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>
      Expand All
    </button>
    <button class="btn" id="btn-collapse">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14"/></svg>
      Collapse All
    </button>
  </div>
</div>

<div class="stats-bar">
  <div class="stat-item"><div class="stat-dot" style="background:var(--accent-cyan)"></div> Nodes: <span class="stat-value" id="stat-nodes">0</span></div>
  <div class="stat-item"><div class="stat-dot" style="background:var(--accent-emerald)"></div> Leaves: <span class="stat-value" id="stat-leaves">0</span></div>
  <div class="stat-item"><div class="stat-dot" style="background:var(--accent-rose)"></div> Max Depth: <span class="stat-value" id="stat-depth">0</span></div>
  <div class="stat-item"><div class="stat-dot" style="background:var(--accent-amber)"></div> Zoom: <span class="stat-value" id="stat-zoom">100%</span></div>
  <div style="flex:1"></div>
  <div class="stat-item">Scroll to zoom · Drag to pan · Click nodes to collapse</div>
</div>

<div class="tooltip" id="tooltip"></div>

<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:var(--edge-ok)"></div> Qubit OK</div>
  <div class="legend-item"><div class="legend-swatch" style="background:var(--edge-lost)"></div> Qubit Lost</div>
  <div class="legend-item"><div class="legend-circle" style="background:#1a4a6f"></div> Decision</div>
  <div class="legend-item"><div class="legend-circle" style="background:#065f46"></div> Success</div>
  <div class="legend-item"><div class="legend-circle" style="background:#312e81"></div> Root</div>
  <div style="border-top:1px solid #2a3550; margin:6px 0; padding-top:6px">
    <div class="legend-item" style="color:#94a3b8"><span style="font-family:'JetBrains Mono',monospace;font-size:10px">Z<sub>3</sub>&#x2192;X<sub>1</sub>&#x2192;Z<sub>4</sub>&#x2717;</span>&nbsp; = meas. path</div>
    <div class="legend-item" style="color:#94a3b8"><span style="color:#fb7185;font-family:'JetBrains Mono',monospace;font-size:10px">&#x2717;</span>&nbsp; = lost qubit</div>
    <div class="legend-item" style="color:#94a3b8"><span style="color:#fbbf24;font-family:'JetBrains Mono',monospace;font-size:10px">&#x26a1;</span>&nbsp; = capture + delay</div>
    <div class="legend-item" style="color:#94a3b8"><span style="color:#fb7185;font-family:'JetBrains Mono',monospace;font-size:10px">Z&#x2260;X</span>&nbsp; = basis conflict</div>
    <div style="border-top:1px solid #2a3550; margin:4px 0; padding-top:4px">
      <div class="legend-item"><span style="color:#60a5fa;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600">Z</span>&nbsp; <span style="color:#94a3b8">= Z basis</span></div>
      <div class="legend-item"><span style="color:#f472b6;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600">X</span>&nbsp; <span style="color:#94a3b8">= X basis</span></div>
      <div class="legend-item"><span style="color:#a78bfa;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600">Y</span>&nbsp; <span style="color:#94a3b8">= Y basis</span></div>
      <div class="legend-item"><span style="color:#fbbf24;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600">A</span>&nbsp; <span style="color:#94a3b8">= absorb (output)</span></div>
      <div class="legend-item"><span style="color:#64748b;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600">I</span>&nbsp; <span style="color:#94a3b8">= deferred basis</span></div>
    </div>
  </div>
</div>

<div class="canvas-area" id="canvas-area"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
/* __TREE_DATA__ */

(function() {
  const treeData = TREE_DATA;
  const NODE_W = 175, NODE_H = 66;
  const H_GAP = 24, V_GAP = 80;

  // Parse key like "6,1,3,4l,5" → [{qubit:"6",lost:false}, {qubit:"1",lost:false}, ...]
  function parseMeasOrder(key) {
    if (!key || key === "Start") return [];
    return key.split(",").map(part => {
      const lost = part.endsWith("l");
      const qubit = lost ? part.slice(0,-1) : part;
      return { qubit, lost };
    });
  }

  // Pauli basis color coding
  const basisColors = { Z: "#60a5fa", X: "#f472b6", Y: "#a78bfa", A: "#fbbf24", I: "#64748b" };
  function basisColor(b) { return basisColors[b] || "#94a3b8"; }

  // Build measurement record incrementally: parent's record + new qubit from this step
  function enrichNodes(node, depth, parentMeasRecord) {
    depth = depth || 0;
    parentMeasRecord = parentMeasRecord || {};
    node._measSteps = parseMeasOrder(node.key);
    node._stepNum = node._measSteps.length;
    node._pauliMap = node.pauli_map || {};
    node._actualMeas = node.actual_meas || {};

    // Inherit parent's measurement record, then add this node's new qubit
    // The basis for each qubit comes from the strategy that was active when it was measured
    const measRecord = Object.assign({}, parentMeasRecord);
    // The last step in _measSteps is the qubit just measured at THIS node
    // Its basis comes from the PARENT's strategy (which decided to measure it)
    // For intermediate nodes, that info is in the parent's pauli_map — but we
    // don't have the parent node here. Instead, each node's own pauli_map
    // reflects the strategy active AT this node. The qubit measured to GET here
    // was measured by the parent's strategy. We approximate: use own pauli_map
    // for qubits in it, inherit from parent for the rest.
    // Add new qubits to record, and upgrade "I" if current strategy resolves it
    node._measSteps.forEach(s => {
      if (!(s.qubit in measRecord)) {
        const b = node._pauliMap[s.qubit];
        measRecord[s.qubit] = (b && b !== "I") ? b : "I";
      } else if (measRecord[s.qubit] === "I") {
        const b = node._pauliMap[s.qubit];
        if (b && b !== "I") measRecord[s.qubit] = b;
      }
    });
    node._measRecord = measRecord;
    // Remember which qubits are in this node's record BEFORE bottom-up
    // These are qubits measured up to this node (pre-branch)
    const ownQubits = new Set(Object.keys(measRecord));

    // Annotate each step with its basis from the accumulated record
    node._measSteps.forEach(s => {
      s.basis = measRecord[s.qubit] || "I";
    });

    // For leaves: supplement with actual_meas for qubits not yet in record
    // Do NOT overwrite inherited bases — those were set when the qubit was actually measured
    node._conflicts = {};
    node._captureDelay = [];
    if (node.is_leaf && Object.keys(node._actualMeas).length) {
      // Add only qubits not already in the record
      Object.entries(node._actualMeas).forEach(([q, b]) => {
        if (!(q in measRecord) && b && b !== "I") {
          measRecord[q] = b;
        }
        // Also upgrade "I" entries if actual_meas has a real basis
        if (measRecord[q] === "I" && b && b !== "I") {
          measRecord[q] = b;
        }
      });
      // Re-annotate steps with the final record
      node._measSteps.forEach(s => {
        s.basis = measRecord[s.qubit] || "I";
      });
    } else if (node.is_leaf) {
      // Leaf without actual_meas — still mark for later
    }

    (node.children || []).forEach(c => enrichNodes(c, depth + 1, measRecord));

    // Bottom-up: resolve "I" entries from children — only for qubits
    // already in this node's record (measured before the branch)
    const childNodes = node.children || [];
    childNodes.forEach(c => {
      const childRec = c._measRecord || {};
      Object.entries(childRec).forEach(([q, b]) => {
        if (b && b !== "I" && ownQubits.has(q) && measRecord[q] === "I") {
          measRecord[q] = b;
        }
      });
    });
    // Re-annotate steps after bottom-up update
    node._measSteps.forEach(s => {
      s.basis = measRecord[s.qubit] || "I";
    });

    // Push resolved values to ALL children — but only for qubits
    // that were in this node's record before bottom-up (pre-branch qubits)
    childNodes.forEach(c => _pushDown(c, measRecord, ownQubits));
  }

  // Enforce parent → child consistency for pre-branch qubits
  function _pushDown(node, parentRec, preBranchQubits) {
    const rec = node._measRecord;
    let changed = false;
    preBranchQubits.forEach(q => {
      const b = parentRec[q];
      if (b && b !== "I" && rec[q] !== b) {
        rec[q] = b;
        changed = true;
      }
    });
    if (changed) {
      node._measSteps.forEach(s => {
        s.basis = rec[s.qubit] || "I";
      });
    }
    (node.children || []).forEach(c => _pushDown(c, rec, preBranchQubits));
  }

  enrichNodes(treeData);

  // Final pass: compute capture+delay at leaves using fully resolved measRecords
  function computeCaptureDelay(node) {
    if (node.is_leaf) {
      const rec = node._measRecord || {};
      const pm = node._pauliMap || {};
      node._conflicts = {};
      node._captureDelay = [String(node.output_qubit)];
      Object.entries(rec).forEach(([q, measB]) => {
        const stratB = pm[q];
        if (stratB && measB !== stratB && measB !== "I" && stratB !== "I") {
          node._conflicts[q] = { measured: measB, needed: stratB };
          if (node._captureDelay.indexOf(q) === -1) {
            node._captureDelay.push(q);
          }
        }
      });
    }
    (node.children || []).forEach(c => computeCaptureDelay(c));
  }
  computeCaptureDelay(treeData);

  const tooltip = document.getElementById("tooltip");

  const area = document.getElementById("canvas-area");
  const svg = d3.select(area).append("svg").attr("width","100%").attr("height","100%");
  const defs = svg.append("defs");

  const gridPat = defs.append("pattern")
    .attr("id","gridPattern").attr("width",40).attr("height",40)
    .attr("patternUnits","userSpaceOnUse");
  gridPat.append("circle").attr("cx",20).attr("cy",20).attr("r",0.6).attr("fill","#2a3550");

  ["ok","lost"].forEach(type => {
    defs.append("marker")
      .attr("id","arrow-"+type)
      .attr("viewBox","0 0 10 6").attr("refX",10).attr("refY",3)
      .attr("markerWidth",8).attr("markerHeight",6)
      .attr("orient","auto")
      .append("path").attr("d","M0,0 L10,3 L0,6 Z")
      .attr("class","arrow-"+type);
  });

  const glowOk = defs.append("filter").attr("id","glow-ok");
  glowOk.append("feGaussianBlur").attr("stdDeviation","3").attr("result","blur");
  glowOk.append("feFlood").attr("flood-color","#34d399").attr("flood-opacity","0.3");
  glowOk.append("feComposite").attr("in2","blur").attr("operator","in");
  const m = glowOk.append("feMerge");
  m.append("feMergeNode");
  m.append("feMergeNode").attr("in","SourceGraphic");

  svg.append("rect").attr("class","grid-bg")
    .attr("width","200%").attr("height","200%")
    .attr("x","-50%").attr("y","-50%");

  const g = svg.append("g");

  const zoom = d3.zoom().scaleExtent([0.05,4]).on("zoom", e => {
    g.attr("transform", e.transform);
    document.getElementById("stat-zoom").textContent = Math.round(e.transform.k*100)+"%";
  });
  svg.call(zoom);

  function computeLayout(n, depth) {
    n._depth = depth || 0;
    const ch = n._collapsed ? [] : (n.children||[]);
    if (!ch.length) { n._width = NODE_W + H_GAP; return; }
    ch.forEach(c => computeLayout(c, n._depth+1));
    n._width = Math.max(ch.reduce((s,c) => s+c._width, 0), NODE_W+H_GAP);
  }

  function assignPos(n, left) {
    left = left || 0;
    const ch = n._collapsed ? [] : (n.children||[]);
    if (!ch.length) { n._x = left + n._width/2; }
    else {
      let off = left;
      ch.forEach(c => { assignPos(c,off); off += c._width; });
      n._x = (ch[0]._x + ch[ch.length-1]._x)/2;
    }
    n._y = n._depth * (NODE_H + V_GAP);
  }

  function treeStats(n) {
    let nodes=1, leaves=n.is_leaf?1:0, maxD=n._depth||0;
    (n.children||[]).forEach(c => {
      const s=treeStats(c); nodes+=s.nodes; leaves+=s.leaves;
      if(s.maxD>maxD) maxD=s.maxD;
    });
    return {nodes,leaves,maxD};
  }

  function render() {
    computeLayout(treeData, 0);
    assignPos(treeData, 0);
    const nodes=[], links=[];
    (function walk(n){
      nodes.push(n);
      (n._collapsed?[]:(n.children||[])).forEach(c => { links.push({source:n,target:c}); walk(c); });
    })(treeData);

    const stats = treeStats(treeData);
    document.getElementById("stat-nodes").textContent = stats.nodes;
    document.getElementById("stat-leaves").textContent = stats.leaves;
    document.getElementById("stat-depth").textContent = stats.maxD;

    g.selectAll("*").remove();

    g.selectAll(".link-path").data(links).join("path")
      .attr("class", d => "link-path "+(d.target.is_lost_edge?"lost":"ok"))
      .attr("marker-end", d => "url(#arrow-"+(d.target.is_lost_edge?"lost":"ok")+")")
      .attr("d", d => {
        const sH = d.source.is_leaf ? NODE_H+14 : NODE_H;
        const tH = d.target.is_leaf ? NODE_H+14 : NODE_H;
        const sx=d.source._x, sy=d.source._y+sH/2;
        const tx=d.target._x, ty=d.target._y-tH/2;
        const my=(sy+ty)/2;
        return "M"+sx+","+sy+" C"+sx+","+my+" "+tx+","+my+" "+tx+","+ty;
      });

    const nodeG = g.selectAll(".node-group").data(nodes).join("g")
      .attr("class","node-group")
      .attr("transform", d => { const h=d.is_leaf?NODE_H+14:NODE_H; return "translate("+(d._x-NODE_W/2)+","+(d._y-h/2)+")"; });
    nodeG.append("rect").attr("class","node-rect")
      .attr("width",NODE_W).attr("height", d => d.is_leaf ? NODE_H + 14 : NODE_H)
      .attr("fill", d => d.id===0?"#312e81":d.is_leaf?"#065f46":"#1a4a6f")
      .attr("stroke", d => d.id===0?"#6366f1":d.is_leaf?"#34d399":"#38bdf8")
      .attr("filter", d => d.is_leaf?"url(#glow-ok)":"none");

    nodeG.append("text").attr("class","node-label")
      .attr("x",NODE_W/2).attr("y",17)
      .text(d => {
        if (d.id===0) return "START";
        if (d.is_leaf) return "\u2713 SUCCESS \u2014 Out: q"+d.output_qubit;
        const mr = (d._measRecord||{})[d.next_action];
        // next_action qubit may not be in this node's record — check children
        let childBasis = "";
        if (!mr || mr === "I") {
          (d.children || []).some(c => {
            const cb = (c._measRecord||{})[d.next_action];
            if (cb && cb !== "I") { childBasis = cb; return true; }
          });
        }
        const basis = (mr && mr !== "I" ? mr : "") || childBasis || d.next_basis || d._pauliMap[d.next_action] || "";
        const bLabel = basis === "A" ? " (absorb)" : (basis && basis !== "I") ? " in "+basis : "";
        return "Step "+d._stepNum+": q"+d.next_action+bLabel;
      });

    // Line 2: measurement path with Pauli bases
    nodeG.filter(d => d.id !== 0).append("text")
      .attr("class","node-sublabel")
      .attr("x",NODE_W/2).attr("y",32)
      .text(d => {
        const steps = d._measSteps;
        if (!steps.length) return "";
        const path = steps.map(s => {
          const b = s.basis && s.basis !== "?" ? s.basis : "";
          const lost = s.lost ? "\u2717" : "";
          return b + s.qubit + lost;
        }).join("\u2192");
        return path.length > 28 ? path.slice(0,26)+"\u2026" : path;
      });

    // Line 3: strategy (truncated)
    nodeG.filter(d => d.id !== 0).append("text")
      .attr("class","node-sublabel")
      .attr("x",NODE_W/2).attr("y",47)
      .text(d => {
        const s = d.strategy;
        return s.length > 24 ? s.slice(0,22)+"\u2026" : s;
      });

    // Root: output qubit line
    nodeG.filter(d => d.id === 0).append("text")
      .attr("class","node-sublabel")
      .attr("x",NODE_W/2).attr("y",34)
      .text(d => "Output: q"+d.output_qubit);

    // Capture + Delay line — leaf nodes only (computed from resolved meas bases)
    nodeG.filter(d => d.is_leaf).append("text")
      .attr("x",NODE_W/2).attr("y",64)
      .attr("text-anchor","middle")
      .attr("pointer-events","none")
      .attr("font-family","'JetBrains Mono', monospace")
      .attr("font-size","9px")
      .attr("fill","#fbbf24")
      .text(d => {
        const cd = d._captureDelay || [];
        if (!cd.length) return "no capture+delay";
        return "\u26a1 capture+delay: q" + cd.join(", q");
      });

    nodeG.filter(d => (d.children||[]).length>0).append("text")
      .attr("class","collapse-indicator")
      .attr("x",NODE_W/2).attr("y", d => { const h=d.is_leaf?NODE_H+14:NODE_H; return h+14; })
      .text(d => d._collapsed?"\u25b8 "+d.children.length+" hidden":"");

    nodeG.on("click", (ev,d) => {
      if(!(d.children||[]).length) return;
      d._collapsed = !d._collapsed;
      render(); fitToScreen();
    });

    nodeG.on("mouseenter", (ev,d) => {
      let h = '<div class="tooltip-title">Node #'+d.id+(d._stepNum?' \u2014 Step '+d._stepNum:'')+'</div>';

      // Measurement path with Pauli bases, color coded
      const steps = d._measSteps;
      if (steps.length) {
        const orderHtml = steps.map(s => {
          const b = s.basis || "?";
          const color = s.lost ? '#fb7185' : (basisColors[b] || '#94a3b8');
          const symbol = s.lost ? '\u2717' : '';
          return '<span style="color:'+color+'" title="'+b+' basis">'+b+'<sub>'+s.qubit+'</sub>'+symbol+'</span>';
        }).join(' <span style="color:#64748b">\u2192</span> ');
        h += '<div class="tooltip-row"><span class="tooltip-key">Meas. Path</span><span class="tooltip-val" style="color:inherit">'+orderHtml+'</span></div>';
      }

      // Strategy's Pauli map (what the current strategy needs)
      const pm = d._pauliMap || {};
      if (Object.keys(pm).length) {
        const mapHtml = Object.entries(pm).sort((a,b) => parseInt(a[0])-parseInt(b[0])).map(([q,b]) => {
          const color = basisColors[b] || '#94a3b8';
          return '<span style="color:'+color+'">'+b+'<sub>'+q+'</sub></span>';
        }).join('  ');
        h += '<div class="tooltip-row"><span class="tooltip-key">Strategy</span><span class="tooltip-val" style="color:inherit">'+mapHtml+'</span></div>';
      }

      // Accumulated measurement record (consistent along paths)
      const mr = d._measRecord || {};
      if (Object.keys(mr).length) {
        const mrHtml = Object.entries(mr).sort((a,b) => parseInt(a[0])-parseInt(b[0])).map(([q,b]) => {
          const conflict = d._conflicts && d._conflicts[q];
          const color = conflict ? '#fb7185' : (basisColors[b] || '#94a3b8');
          const suffix = conflict ? '<span style="color:#fb7185"> \u2260'+conflict.needed+'</span>' : '';
          return '<span style="color:'+color+'">'+b+'<sub>'+q+'</sub>'+suffix+'</span>';
        }).join('  ');
        h += '<div class="tooltip-row"><span class="tooltip-key">Meas. Bases</span><span class="tooltip-val" style="color:inherit">'+mrHtml+'</span></div>';
      }

      const rows = [["Output Qubit","q"+d.output_qubit+" (A measurement)"],
        ["Lost Qubits",d.lost_qubits||"none"]];

      if (!d.is_leaf) {
        let nb = d.next_basis || pm[d.next_action] || "";
        if (!nb || nb === "I") {
          (d.children || []).some(c => {
            const cb = (c._measRecord||{})[d.next_action];
            if (cb && cb !== "I") { nb = cb; return true; }
          });
        }
        nb = nb || "?";
        const basisLabel = nb === "A" ? "absorb (A)" : nb+" basis";
        rows.push(["Next Measurement","q"+d.next_action+" in "+basisLabel]);
      } else {
        rows.push(["Result", "Success"]);

        // Show conflicts explicitly
        const conflicts = d._conflicts || {};
        const cKeys = Object.keys(conflicts);
        if (cKeys.length) {
          const confHtml = cKeys.map(q => {
            return '<span style="color:#fb7185">q'+q+': measured '+conflicts[q].measured+', need '+conflicts[q].needed+'</span>';
          }).join(', ');
          h += '<div class="tooltip-row"><span class="tooltip-key">Basis Conflicts</span><span class="tooltip-val" style="color:#fb7185">'+confHtml+'</span></div>';
        }

        const cd = d._captureDelay || [];
        if (cd.length) rows.push(["Capture + Delay", "q" + cd.join(", q")]);
        else rows.push(["Capture + Delay", "none"]);
      }

      rows.forEach(([k,v]) => {
        let c = "";
        if(v==="Success") c=" success";
        h += '<div class="tooltip-row"><span class="tooltip-key">'+k+'</span><span class="tooltip-val'+c+'">'+v+'</span></div>';
      });
      tooltip.innerHTML = h;
      tooltip.style.left = (ev.clientX+16)+"px";
      tooltip.style.top = (ev.clientY-10)+"px";
      tooltip.classList.add("visible");
    }).on("mousemove", ev => {
      tooltip.style.left = (ev.clientX+16)+"px";
      tooltip.style.top = (ev.clientY-10)+"px";
    }).on("mouseleave", () => tooltip.classList.remove("visible"));
  }

  function fitToScreen() {
    const w=area.clientWidth, h=area.clientHeight;
    let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
    (function walk(n){
      if(n._x!==undefined){
        const h = n.is_leaf ? NODE_H+14 : NODE_H;
        minX=Math.min(minX,n._x-NODE_W/2); maxX=Math.max(maxX,n._x+NODE_W/2);
        minY=Math.min(minY,n._y-h/2); maxY=Math.max(maxY,n._y+h/2+20);
      }
      (n._collapsed?[]:(n.children||[])).forEach(walk);
    })(treeData);

    const tw=maxX-minX+80, th=maxY-minY+80;
    const scale=Math.min(w/tw, h/th, 1.5);
    const cx=(minX+maxX)/2, cy=(minY+maxY)/2;
    svg.transition().duration(600).call(zoom.transform,
      d3.zoomIdentity.translate(w/2,h/2).scale(scale).translate(-cx,-cy));
  }

  document.getElementById("btn-fit").addEventListener("click", fitToScreen);
  document.getElementById("btn-expand").addEventListener("click", () => {
    (function ex(n){ n._collapsed=false; (n.children||[]).forEach(ex); })(treeData);
    render(); fitToScreen();
  });
  document.getElementById("btn-collapse").addEventListener("click", () => {
    (function co(n){ if((n.children||[]).length && n.id!==0) n._collapsed=true; (n.children||[]).forEach(co); })(treeData);
    render(); fitToScreen();
  });

  render();
  setTimeout(fitToScreen, 100);
  window.addEventListener("resize", fitToScreen);
})();
</script>
</body>
</html>"""
if __name__ == '__main__':
    from CodeFunctions.graphs import *
    import matplotlib.pyplot as plt
    from ErasureDecoder import LT_Erasure_decoder, AllPossStrats
    from DecoderFixedMeasUpdated import *

    n_qbts = 7
    distance = 2
    in_qubit = 0
    graph_nodes = list(range(8))
    graph_edges = [(0, 4), (0, 5), (0, 7), (1, 3), (1, 4), (1, 5), (2, 3),
                   (2, 5), (2, 7), (3, 6), (4, 6), (6, 7)]
    gstate = graph_from_nodes_and_edges(graph_nodes, graph_edges)
    erasure_decoder = LT_Erasure_decoder(n_qbts, distance, gstate, in_qbt=in_qubit)
    input_strats = erasure_decoder.strategies_ordered

    # strats = AllPossStrats(graph_nodes, gstate)
    # input_strats = strats.get_possible_decoding_strats()


    measurement_order = [1, 2, 3, 6, 4, 7, 5]
    no_anti_com_flag = False
    # adaptive_decoder = LT_FullHybridDecoder(gstate, input_strats, measurement_order=measurement_order,
    #                                         no_anti_com_flag=no_anti_com_flag, printing=False)

    adaptive_decoder = LT_FullHybridDecoderNew(gstate, input_strats, measurement_order=measurement_order,
                                            no_anti_com_flag=no_anti_com_flag, printing=False)
    tree_branches = adaptive_decoder.tree_branches


    filename = "cube_fixed_order_test"
    tree, node_info, name_idx, leaf_idxs = get_tree_fixed_meas_patt(adaptive_decoder.tree_branches)
    max_cd, all_leaves = compute_max_capture_delay(tree, node_info, name_idx, leaf_idxs)
    print("Number of decoding qubits: ", max_cd)

    tree, node_information, node_name_idx, leaf_node_idxs = get_tree_fixed_meas_patt(tree_branches)

    # plot_decision_tree(tree, node_information, node_name_idx, leaf_node_idxs, filename)
    show_interactive_tree(tree, node_information, node_name_idx, leaf_node_idxs)
