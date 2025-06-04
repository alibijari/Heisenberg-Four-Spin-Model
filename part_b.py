import numpy as np

# ======================================
# 1) Basis Generation and Ordering
# ======================================

def generate_basis_4_spins():
    """
    Generate all possible 4-spin states, with each spin in {+1, -1}.
    'Up' is +1, 'Down' is -1.

    Returns:
        A list of tuples, each tuple is (s1, s2, s3, s4).
    """
    basis = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            for s3 in (+1, -1):
                for s4 in (+1, -1):
                    basis.append((s1, s2, s3, s4))
    return basis


def total_sz(state):
    """
    Compute the total S_z of a 4-spin state, where each spin is +1 or -1.
    In quantum-mechanical terms, each +1 corresponds to spin up (+1/2)
    and each -1 corresponds to spin down (-1/2).

    total_sz = (s1 + s2 + s3 + s4) / 2

    Returns:
        float: The total Sz value.
    """
    return sum(state) / 2.0


def group_and_order_basis():
    """
    Group all 16 states by their total S_z, in descending order:
      +2, +1, 0, -1, -2.
    Then create a single list with this order.

    Returns:
        ordered_basis (list): The 16 states in the desired S_z-sorted order.
        state_to_index (dict): Maps each state -> index in the ordered list.
    """
    all_states = generate_basis_4_spins()
    # Possible total S_z values: +2, +1, 0, -1, -2
    grouping = {+2: [], +1: [], 0: [], -1: [], -2: []}

    # Distribute states into groups based on their total S_z
    for st in all_states:
        grouping[total_sz(st)].append(st)

    # Concatenate groups to form a single ordered list
    ordered_basis = []
    for sz_val in (+2, +1, 0, -1, -2):
        ordered_basis.extend(grouping[sz_val])

    # Build a dictionary mapping each state to its index in the sorted list
    state_to_index = {}
    for i, st in enumerate(ordered_basis):
        state_to_index[st] = i

    return ordered_basis, state_to_index


# ======================================
# 2) Spin Operators and Matrix Filling
# ======================================

def spin_z(state, i):
    """
    Compute the spin_z for the i-th spin in the given state.
    state[i] is either +1 (up) or -1 (down).
    For spin-1/2, Sz(i) = state[i] / 2.0.

    Args:
        state (tuple): A 4-spin state.
        i (int): Index of the spin site (0 through 3).

    Returns:
        float: The S_z value of spin i in the given state.
    """
    return state[i] / 2.0


def flip_spin(state, i):
    """
    Flip the i-th spin: +1 -> -1 or -1 -> +1.

    For spin-1/2 in a simple formalism, we assume the overall phase factor is +1.
    If we had higher spins or a more complete quantum approach,
    the flip could carry sqrt factors or phases.

    Returns:
        (new_state, phase) where new_state is a tuple identical to 'state'
        but with the i-th spin flipped, and 'phase' is 1.0 in this simplified model.
    """
    st_list = list(state)
    st_list[i] *= -1
    return tuple(st_list), 1.0


def add_si_sj_terms(H, state_to_index, i, j):
    """
    Add contributions from (Si · Sj) to the Hamiltonian matrix H
    for spin sites i and j.

    We use:
      Si·Sj = (Si_z * Sj_z) + 1/2 (Si+ Sj- + Si- Sj+).

    - Diagonal part: Si_z * Sj_z
      This is straightforward: up·up or down·down => +1/4,
      up·down or down·up => -1/4, etc.
    - Flip-flop part: (Si+ Sj- + Si- Sj+)
      This acts on pairs of spins that are up/down => down/up.
      The matrix element is 1/2 for such transitions.

    Args:
        H (numpy.ndarray): The Hamiltonian matrix to be updated (16×16).
        state_to_index (dict): Maps each state -> index in H.
        i, j (int): Spin site indices (0..3).
    """
    dim = H.shape[0]

    # Create a reverse mapping: index -> state
    index_to_state = [None] * dim
    for st, idx in state_to_index.items():
        index_to_state[idx] = st

    # Loop over all basis states
    for idx, st in enumerate(index_to_state):

        # Diagonal contribution: Si_z * Sj_z
        val_diag = spin_z(st, i) * spin_z(st, j)
        H[idx, idx] += val_diag

        # Flip-flop contributions:
        #   1) Si+ Sj- : requires st[i] = -1 and st[j] = +1
        if st[i] == -1 and st[j] == +1:
            new_st, phase1 = flip_spin(st, i)  # flip i
            new_st, phase2 = flip_spin(new_st, j)  # flip j
            new_idx = state_to_index[new_st]
            H[new_idx, idx] += 0.5 * (phase1 * phase2)

        #   2) Si- Sj+ : requires st[i] = +1 and st[j] = -1
        if st[i] == +1 and st[j] == -1:
            new_st, phase1 = flip_spin(st, i)
            new_st, phase2 = flip_spin(new_st, j)
            new_idx = state_to_index[new_st]
            H[new_idx, idx] += 0.5 * (phase1 * phase2)


# ======================================
# 3) Build the 4-spin Hamiltonian
# ======================================

def build_hamiltonian_4_spins():
    """
    Build the 16×16 Hamiltonian matrix for 4 spin-1/2 sites in the basis
    sorted by total Sz:
       H = S1·S2 + S2·S3 + S3·S4 + S4·S1.

    Returns:
        H (numpy.ndarray): The 16×16 Hamiltonian matrix.
        ordered_basis (list): The 16 states in the chosen ordering.
    """
    ordered_basis, state_to_index = group_and_order_basis()
    dim = len(ordered_basis)  # should be 16
    H = np.zeros((dim, dim), dtype=float)

    # We add 4 terms for neighboring pairs: (0,1), (1,2), (2,3), (3,0).
    neighbor_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for (i, j) in neighbor_pairs:
        add_si_sj_terms(H, state_to_index, i, j)

    return H, ordered_basis


# ======================================
# 4) Printing Functions
# ======================================

def print_matrix(M, label=None):
    """
    Print the matrix M row by row, with extra spacing for readability.
    We use a format specifier such as +8.3f, ensuring:
      - sign (+/-)
      - 8 total width
      - 3 decimals

    Args:
        M (numpy.ndarray): The matrix to be printed.
        label (str): An optional label or title to print before the matrix.
    """
    if label is not None:
        print(label)
    rows, cols = M.shape
    for r in range(rows):
        # Two spaces between columns for clarity
        row_str = "  ".join(f"{M[r, c]:+8.3f}" for c in range(cols))
        print(f"row {r:2d}   {row_str}")
    print()  # blank line after the matrix


# ======================================
# 5) Main Execution
# ======================================

def main():
    # 1) Build Hamiltonian and get the ordered basis
    H, ordered_basis = build_hamiltonian_4_spins()

    # 2) Print the ordered basis (16 states) with total Sz
    print(">>> Ordered Basis (state) and their total Sz:\n")
    print(f"{'Index':>5}   {'State':>20}   {'Total Sz':>8}")
    print("-" * 40)
    for i, st in enumerate(ordered_basis):
        print(f"{i:5d}   {str(st):>20}   {total_sz(st):8.1f}")
    print()

    # 3) Print the full 16x16 Hamiltonian
    print(">>> Full 16×16 Hamiltonian matrix in the chosen (S_z-sorted) basis:")
    print_matrix(H)

    # 4) Show each S_z block individually
    #    Distribution of states in ordered_basis:
    #    - S_z=+2 => 1 state  -> indices [0,1)
    #    - S_z=+1 => 4 states -> indices [1,5)
    #    - S_z= 0 => 6 states -> indices [5,11)
    #    - S_z=-1 => 4 states -> indices [11,15)
    #    - S_z=-2 => 1 state  -> indices [15,16)
    block_ranges = {
        "+2": (0, 1),
        "+1": (1, 5),
        "0": (5, 11),
        "-1": (11, 15),
        "-2": (15, 16),
    }

    for sz_label, (start, end) in block_ranges.items():
        submatrix = H[start:end, start:end]
        block_dim = end - start
        block_label = f">>> Block for S_z = {sz_label}  (dimension: {block_dim}×{block_dim})"
        print_matrix(submatrix, label=block_label)

    # 5) Eigenvalues
    eigenvalues = np.linalg.eigvalsh(H)
    print(">>> Eigenvalues of the Hamiltonian:")
    for i, eig in enumerate(sorted(eigenvalues)):
        print(f"Eigenvalue {i:2d}: {eig:+.6f}")


if __name__ == "__main__":
    main()


