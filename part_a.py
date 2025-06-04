import numpy as np

# ======================================
# 1) Define single-spin (2x2) operators
# ======================================
Sx = (1 / 2) * np.array([[0, 1],
                         [1, 0]], dtype=complex)

Sy = (1 / 2) * np.array([[0, -1j],
                         [1j, 0]], dtype=complex)

Sz = (1 / 2) * np.array([[1, 0],
                         [0, -1]], dtype=complex)

I2 = np.eye(2, dtype=complex)


# ======================================
# 2) Helper functions
# ======================================
def kron4(op_list):
    """
    Returns the Kronecker product of exactly 4 operators in op_list.
    Each operator must be 2x2 (for a total dimension of 2^4 = 16).
    """
    # op_list is expected to be something like [Sx, Sy, I2, Sz].
    return np.kron(
        np.kron(
            np.kron(op_list[0], op_list[1]),
            op_list[2]
        ),
        op_list[3]
    )


def build_two_spin_term(opA, opB, idxA, idxB):
    """
    Places operator opA on spin idxA and opB on spin idxB,
    with the identity on the other spins.

    idxA, idxB ∈ {0,1,2,3}, idxA != idxB.
    """
    operators = []
    for spin_site in range(4):
        if spin_site == idxA:
            operators.append(opA)
        elif spin_site == idxB:
            operators.append(opB)
        else:
            operators.append(I2)
    return kron4(operators)


def S_dot_S(i, j):
    """
    Builds the matrix for the dot product:
        S_i · S_j = S^x_i S^x_j + S^y_i S^y_j + S^z_i S^z_j
    in a 4-spin system, with i, j ∈ {1,2,3,4}.

    Internally converts i,j (1-based) to Python 0-based indices.
    """
    i_index = i - 1
    j_index = j - 1

    sx_term = build_two_spin_term(Sx, Sx, i_index, j_index)
    sy_term = build_two_spin_term(Sy, Sy, i_index, j_index)
    sz_term = build_two_spin_term(Sz, Sz, i_index, j_index)

    return sx_term + sy_term + sz_term


# ======================================
# 3) Build the 4-spin Hamiltonian
# ======================================
def build_hamiltonian():
    """
    Constructs the Hamiltonian for 4 spin-1/2 particles in a ring:
        H = S1·S2 + S2·S3 + S3·S4 + S4·S1
    where S·S = S^x S^x + S^y S^y + S^z S^z.
    """
    H = np.zeros((16, 16), dtype=complex)

    H += S_dot_S(1, 2)
    H += S_dot_S(2, 3)
    H += S_dot_S(3, 4)
    H += S_dot_S(4, 1)

    return H


# ======================================
# 4) Utility to print a matrix nicely
# ======================================
def print_nice_matrix(M):
    """
    Prints a 2D matrix of complex numbers with consistent spacing,
    so columns align neatly. Each entry is shown as:
        +X.XXX+Y.YYYj
    with sign and 3 decimals for both real & imag parts.
    """
    rows, cols = M.shape
    for r in range(rows):
        row_parts = []
        for c in range(cols):
            val = M[r, c]
            formatted = f"{val.real:+7.3f}{val.imag:+7.3f}j"
            row_parts.append(formatted)
        # Join columns with two spaces for clarity
        print("  ".join(row_parts))


# ======================================
# 5) Main script
# ======================================
def main():
    # 1) Build the Hamiltonian
    H = build_hamiltonian()

    # 2) Print H (16x16)
    print("Hamiltonian (16x16):")
    print_nice_matrix(H)

    # 3) Diagonalize H to get eigenvalues & eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(H)

    # 4) Sort eigenvalues in ascending order, reorder eigenvectors accordingly
    idx_sorted = np.argsort(eigenvals)
    eigenvals_sorted = eigenvals[idx_sorted]
    eigenvecs_sorted = eigenvecs[:, idx_sorted]

    # 5) Print sorted eigenvalues (to many decimals if desired)
    print("\nEigenvalues (sorted):")
    for i, val in enumerate(eigenvals_sorted):
        print(f"Eigenvalue {i + 1} = {val.real:.3f}")

    # 6) Print all eigenkets (in ascending order), each to 3 decimals
    print("\nAll eigenkets in ascending order of eigenvalue:")
    for i, val in enumerate(eigenvals_sorted):
        print(f"Eigenket {i + 1} (Eigenvalue = {val.real:.3f}):")
        ket_str = [f"{comp.real:.3f}+{comp.imag:.3f}j" for comp in eigenvecs_sorted[:, i]]
        print(ket_str)
        print()

    # 7) Build matrix U (16x16) from the sorted eigenvectors
    U = np.zeros((16, 16), dtype=complex)
    for i in range(16):
        U[:, i] = eigenvecs_sorted[:, i]

    print("\nMatrix U (16x16), columns = sorted eigenkets:")
    print_nice_matrix(U)

    # 8) Compute U^dagger = (U*)^T
    U_dagger = U.conjugate().T

    print("\nMatrix U^\u2020 (U dagger, 16x16):")
    print_nice_matrix(U_dagger)

    # 9) Build the diagonalized Hamiltonian, H_diag = U^dagger * H * U
    H_diag = U_dagger @ H @ U

    print("\nDiagonalized Hamiltonian H_diag = U^\u2020 * H * U:")
    print_nice_matrix(H_diag)


# ======================================
# 6) Execution
# ======================================
if __name__ == "__main__":
    main()
