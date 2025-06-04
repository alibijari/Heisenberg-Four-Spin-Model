import numpy as np

# ======================================
# 1) Basis Generation and Ordering
# ======================================

def generate_basis_4_spins():
    """
    Generate all possible 4-spin states, with each spin in {+1, -1}.
    '+1' represents spin-up (|+1/2⟩), '-1' represents spin-down (|-1/2⟩).

    Returns:
        list: A list of tuples, each tuple representing a state (s1, s2, s3, s4).
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
    Compute the total S_z of a 4-spin state.
    Each spin is +1 (up, S_z = +1/2) or -1 (down, S_z = -1/2).
    Total S_z = (s1 + s2 + s3 + s4) / 2.
    """
    return sum(state) / 2.0

def group_and_order_basis():
    """
    Group all 16 spin states by their total S_z (+2, +1, 0, -1, -2) and order them
    in descending S_z order to form the basis for the Hamiltonian.

    Returns:
        ordered_basis (list): List of 16 states sorted by S_z.
        state_to_index (dict): Maps each state to its index in ordered_basis.
    """
    all_states = generate_basis_4_spins()
    # Initialize groups for each possible S_z value
    grouping = {+2: [], +1: [], 0: [], -1: [], -2: []}

    # Assign states to their S_z group
    for st in all_states:
        grouping[total_sz(st)].append(st)

    # Concatenate groups in descending S_z order
    ordered_basis = []
    for sz_val in (+2, +1, 0, -1, -2):
        ordered_basis.extend(grouping[sz_val])

    # Map each state to its index in the ordered basis
    state_to_index = {st: i for i, st in enumerate(ordered_basis)}
    return ordered_basis, state_to_index

# ======================================
# 2) Spin Operators and Matrix Filling
# ======================================

def spin_z(state, i):
    """
    Compute the S_z value for the i-th spin in the given state.
    For spin-1/2, S_z = state[i] / 2, where state[i] is +1 (up) or -1 (down).
    """
    return state[i] / 2.0

def flip_spin(state, i):
    """
    Flip the i-th spin: +1 -> -1 or -1 -> +1.
    In this model, the phase factor for the flip is assumed to be +1.
    """
    st_list = list(state)
    st_list[i] *= -1
    return tuple(st_list), 1.0

def add_si_sj_terms(H, state_to_index, i, j):
    """
    Add contributions from the Si·Sj term to the Hamiltonian matrix.
    Si·Sj = Si_z Sj_z + (1/2)(Si+ Sj- + Si- Sj+).
    - Diagonal: Si_z Sj_z contributes ±1/4 based on spin alignment.
    - Off-diagonal: Si+ Sj- and Si- Sj+ cause spin flips with amplitude 1/2.
    """
    dim = H.shape[0]
    # Create reverse mapping: index -> state
    index_to_state = [None] * dim
    for st, idx in state_to_index.items():
        index_to_state[idx] = st

    for idx, st in enumerate(index_to_state):
        # Diagonal term: Si_z Sj_z
        val_diag = spin_z(st, i) * spin_z(st, j)
        H[idx, idx] += val_diag

        # Off-diagonal terms
        # Si+ Sj-: Requires st[i] = -1, st[j] = +1
        if st[i] == -1 and st[j] == +1:
            new_st, phase1 = flip_spin(st, i)
            new_st, phase2 = flip_spin(new_st, j)
            new_idx = state_to_index[new_st]
            H[new_idx, idx] += 0.5 * (phase1 * phase2)

        # Si- Sj+: Requires st[i] = +1, st[j] = -1
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
    Construct the 16x16 Hamiltonian for a 4-spin system:
    H = S1·S2 + S2·S3 + S3·S4 + S4·S1.
    The basis is sorted by total S_z.

    Returns:
        H (numpy.ndarray): 16x16 Hamiltonian matrix.
        ordered_basis (list): List of 16 basis states in S_z order.
    """
    ordered_basis, state_to_index = group_and_order_basis()
    dim = len(ordered_basis)
    H = np.zeros((dim, dim), dtype=float)

    # Define nearest-neighbor pairs for the ring topology
    neighbor_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for (i, j) in neighbor_pairs:
        add_si_sj_terms(H, state_to_index, i, j)

    return H, ordered_basis

# ==============================================
# 4) Build Transition Matrix for Each S_z Block
# ==============================================

def build_transition_matrix_block(states, state_to_index):
    """
    Build the transition matrix T for a given S_z block.
    T represents a cyclic shift: |s1, s2, s3, s4⟩ -> |s4, s1, s2, s3⟩.
    The matrix is constructed within the subspace of states with the same S_z.

    Args:
        states (list): List of states in the S_z block.
        state_to_index (dict): Maps each state to its index in the full basis.

    Returns:
        T (numpy.ndarray): Transition matrix for the block.
    """
    dim = len(states)
    T = np.zeros((dim, dim), dtype=float)
    for idx, state in enumerate(states):
        # Perform cyclic shift: (s1, s2, s3, s4) -> (s4, s1, s2, s3)
        shifted = (state[3], state[0], state[1], state[2])
        # Compute relative index within the block
        new_idx = state_to_index[shifted] - state_to_index[states[0]]
        T[new_idx, idx] = 1.0
    return T

# ======================================
# 5) Printing Functions
# ======================================

def print_matrix(M, label=None):
    """
    Print a matrix row by row with formatted output.
    Each element is displayed with sign, 8-character width, and 3 decimal places.

    Args:
        M (numpy.ndarray): Matrix to print.
        label (str, optional): Title to print before the matrix.
    """
    if label is not None:
        print(label)
    rows, cols = M.shape
    for r in range(rows):
        row_str = "  ".join(f"{M[r, c]:+8.3f}" for c in range(cols))
        print(f"row {r:2d}   {row_str}")
    print()

def print_eigen_info(T, H_block, states, final_matrix, start):
    """
    Print eigenvalues and eigenvectors of the transition matrix T,
    apply the Hamiltonian block to each eigenvector, and update the final matrix.
    The final matrix stores H in the T-eigenvector basis.

    Args:
        T (numpy.ndarray): Transition matrix for the S_z block.
        H_block (numpy.ndarray): Hamiltonian submatrix for the S_z block.
        states (list): Basis states for the S_z block.
        final_matrix (numpy.ndarray): 16x16 matrix to store H in T-eigenvector basis.
        start (int): Starting index of the block in the full basis.
    """
    eigvals, eigvecs = np.linalg.eig(T)
    print(f">>> Eigenvalues and Eigenvectors of T:\n")

    for i, val in enumerate(eigvals):
        print(f"Eigenvalue |ψ{i}⟩ : {val.real:+.1f} {val.imag:+.1f}i")
        vec = eigvecs[:, i]
        # Express eigenvector as a linear combination of S_z basis states
        ket_str = " + ".join(
            f"({vec[j].real:+.1f}{vec[j].imag:+.1f}i) |{states[j]}⟩"
            for j in range(len(states)) if abs(vec[j]) > 1e-10
        )
        print(f"Eigenket |ψ{i}⟩ = {ket_str}")
        print()

        # Apply H_block to the eigenvector
        H_ket = H_block @ vec
        if np.allclose(H_ket, 0):
            H_ket_str = "0"
        else:
            H_ket_str = " + ".join(
                f"({H_ket[j].real:+.1f}{H_ket[j].imag:+.1f}i) |{states[j]}⟩"
                for j in range(len(states)) if abs(H_ket[j]) > 1e-10
            )
        print(f"H |ψ{i}⟩ = {H_ket_str}")
        print()

    # Transform H_block to T-eigenvector basis: V^dagger H V
    V = eigvecs
    submatrix = np.dot(V.conj().T, np.dot(H_block, V))
    block_dim = len(states)
    final_matrix[start:start + block_dim, start:start + block_dim] = submatrix

# ======================================
# 6) Main Execution
# ======================================

def main():
    """
    Main function to compute and analyze the 4-spin Hamiltonian.
    - Builds the Hamiltonian in the S_z-sorted basis.
    - Prints the basis, full Hamiltonian, and S_z blocks.
    - Constructs transition matrices and transforms H to T-eigenvector basis.
    - Computes and prints eigenvalues of the final matrix.
    """
    # Build Hamiltonian and basis
    H, ordered_basis = build_hamiltonian_4_spins()
    state_to_index = {st: i for i, st in enumerate(ordered_basis)}

    # Print ordered basis with total S_z
    print(">>> Ordered Basis (state) and their total Sz:\n")
    print(f"{'Index':>5}   {'State':>20}   {'Total Sz':>8}")
    print("-" * 40)
    for i, st in enumerate(ordered_basis):
        print(f"{i:5d}   {str(st):>20}   {total_sz(st):8.1f}")
    print()

    # Print full 16x16 Hamiltonian
    print(">>> Full 16×16 Hamiltonian matrix in the chosen (S_z-sorted) basis:\n")
    print_matrix(H)

    # Initialize final matrix for H in T-eigenvector basis
    final_matrix = np.zeros((16, 16), dtype=complex)

    # Define S_z block ranges
    block_ranges = {
        "+2": (0, 1),   # 1 state
        "+1": (1, 5),   # 4 states
        "0": (5, 11),   # 6 states
        "-1": (11, 15), # 4 states
        "-2": (15, 16), # 1 state
    }

    # Process each S_z block
    print(">>> Eigenkets of T and Action of H:\n")
    for sz_label, (start, end) in block_ranges.items():
        block_dim = end - start
        states = ordered_basis[start:end]
        submatrix = H[start:end, start:end]
        block_label = f">>> Block for S_z = {sz_label}  states: {states}  (dimension: {block_dim}×{block_dim})"
        print_matrix(submatrix, label=block_label)
        T = build_transition_matrix_block(states, state_to_index)
        H_block = H[start:end, start:end]
        print_eigen_info(T, H_block, states, final_matrix, start)
        print(f">>> Transition Matrix for S_z = {sz_label}:")
        print_matrix(T)

    # Print final 16x16 matrix in T-eigenvector basis
    print(">>> Full 16x16 Matrix after Transition in the chosen (S_z-sorted) basis:\n")
    print_matrix(final_matrix.real)  # Print real part; imaginary parts are zero

    # Compute and print eigenvalues of the final matrix
    eigenvalues = np.linalg.eigvalsh(final_matrix)
    print(">>> Eigenvalues of the Matrix after Transition:")
    for i, eig in enumerate(sorted(eigenvalues)):
        print(f"Eigenvalue {i:2d}: {eig:+.3f}")

if __name__ == "__main__":
    main()