import numpy as np

"""
───────────────────────────────────────────────────────────────────────────────
Section 1 — Single-spin computational basis
Defines the column-vector representations of the two basis states for a spin-½
particle: |↑⟩ ≡ (1,0)ᵀ and |↓⟩ ≡ (0,1)ᵀ.
───────────────────────────────────────────────────────────────────────────────
"""
ket_up = np.array([[1], [0]])
ket_down = np.array([[0], [1]])

"""
───────────────────────────────────────────────────────────────────────────────
Section 2 — Two-spin triplet and singlet states
* `triplet_states()` returns the symmetric S = 1 manifold
    – |T₊⟩ = |↑↑⟩
    – |T₀⟩ = (|↑↓⟩ + |↓↑⟩)/√2
    – |T₋⟩ = |↓↓⟩
* `singlet_state()` returns the antisymmetric S = 0 singlet
    – |S₀⟩ = (|↑↓⟩ − |↓↑⟩)/√2
───────────────────────────────────────────────────────────────────────────────
"""


def triplet_states():
    Tp = np.kron(ket_up, ket_up)
    T0 = (1 / np.sqrt(2)) * (np.kron(ket_up, ket_down) + np.kron(ket_down, ket_up))
    Tm = np.kron(ket_down, ket_down)
    return Tp, T0, Tm


def singlet_state():
    return (1 / np.sqrt(2)) * (np.kron(ket_up, ket_down) - np.kron(ket_down, ket_up))


Tp, T0, Tm = triplet_states()
S0_two = singlet_state()

"""
───────────────────────────────────────────────────────────────────────────────
Section 3 — Four-spin basis organised by (S, mₛ)
* `groups` lists every symmetry sector together with the actual kets and a tag
  that tells how it was constructed (T⊗T, T⊗S, S⊗T, S⊗S).
* `order` fixes a global ordering so that every state has an unambiguous index.
* `sm_basis_labels` holds printable names used later when pretty-printing.
───────────────────────────────────────────────────────────────────────────────
"""
groups = [
    ('2', '+2', [np.kron(Tp, Tp)], ['T⊗T']),
    ('2', '+1', [(1 / np.sqrt(2)) * (np.kron(Tp, T0) + np.kron(T0, Tp))], ['T⊗T']),
    ('2', '0', [(1 / np.sqrt(6)) * (np.kron(Tp, Tm) + 2 * np.kron(T0, T0) + np.kron(Tm, Tp))], ['T⊗T']),
    ('2', '-1', [(1 / np.sqrt(2)) * (np.kron(Tm, T0) + np.kron(T0, Tm))], ['T⊗T']),
    ('2', '-2', [np.kron(Tm, Tm)], ['T⊗T']),
    ('1', '+1', [-(1 / np.sqrt(2)) * (np.kron(Tp, T0) - np.kron(T0, Tp)),
                 np.kron(Tp, S0_two),
                 np.kron(S0_two, Tp)], ['T⊗T', 'T⊗S', 'S⊗T']),
    ('1', '0', [-(1 / np.sqrt(2)) * (np.kron(Tp, Tm) - np.kron(Tm, Tp)),
                np.kron(T0, S0_two),
                np.kron(S0_two, T0)], ['T⊗T', 'T⊗S', 'S⊗T']),
    ('1', '-1', [-(1 / np.sqrt(2)) * (np.kron(T0, Tm) - np.kron(Tm, T0)),
                 np.kron(Tm, S0_two),
                 np.kron(S0_two, Tm)], ['T⊗T', 'T⊗S', 'S⊗T']),
    ('0', '0', [(1 / np.sqrt(3)) * (np.kron(Tp, Tm) - np.kron(T0, T0) + np.kron(Tm, Tp)),
                np.kron(S0_two, S0_two)], ['T⊗T', 'S⊗S'])
]

order = [
    ('2', '+2', 0), ('2', '+1', 0), ('2', '0', 0), ('2', '-1', 0), ('2', '-2', 0),
    ('1', '+1', 0), ('1', '+1', 1), ('1', '+1', 2),
    ('1', '0', 0), ('1', '0', 1), ('1', '0', 2),
    ('1', '-1', 0), ('1', '-1', 1), ('1', '-1', 2),
    ('0', '0', 0), ('0', '0', 1)
]

sm_basis_labels = [
    '|2,+2⟩', '|2,+1⟩', '|2,0⟩', '|2,-1⟩', '|2,-2⟩',
    '|1,+1⟩', '|1,+1⟩', '|1,+1⟩',
    '|1,0⟩', '|1,0⟩', '|1,0⟩',
    '|1,-1⟩', '|1,-1⟩', '|1,-1⟩',
    '|0,0⟩', '|0,0⟩'
]

"""
───────────────────────────────────────────────────────────────────────────────
Section 4 — Heisenberg Hamiltonian on a 4-site ring
* Defines the single-spin matrices I, S_z, S_±.
* `op_site(op, site)` inserts the operator on the chosen site and identities
  elsewhere.
* Builds H = Σ Sᵢ·Sᵢ₊₁ with periodic boundary conditions.
───────────────────────────────────────────────────────────────────────────────
"""
I = np.eye(2)
Sz = 0.5 * np.array([[1, 0], [0, -1]])
Sp = np.array([[0, 1], [0, 0]])
Sm = np.array([[0, 0], [1, 0]])


def op_site(op, site, N=4):
    ops = [I] * N
    ops[site] = op
    res = ops[0]
    for o in ops[1:]:
        res = np.kron(res, o)
    return res


H = sum(op_site(Sz, i) @ op_site(Sz, (i + 1) % 4) for i in range(4))
H += 0.5 * sum(op_site(Sp, i) @ op_site(Sm, (i + 1) % 4) for i in range(4))
H += 0.5 * sum(op_site(Sm, i) @ op_site(Sp, (i + 1) % 4) for i in range(4))

"""
───────────────────────────────────────────────────────────────────────────────
Section 5 — Utility routines for formatted output
  • Column-width constants and a helper to render basis states as |↑↓…⟩ strings
  • `print_block` prints any matrix with row/column labels in a neat table
  • `show_state` and `show_H_state` give sparse listings of kets and H|ket⟩
───────────────────────────────────────────────────────────────────────────────
"""
COL_W = 8
FMT_VAL = f"{{:+{COL_W}.3f}}"
LABEL_W_SUB = 16
LABEL_W_GR = 18

_arrow = {0: '↑', 1: '↓'}


def idx_arrow(i, N=4):
    bits = [(i >> (N - 1 - j)) & 1 for j in range(N)]
    return '|' + ''.join(_arrow[b] for b in bits) + '⟩'


col_titles = [idx_arrow(i) for i in range(16)]


def print_block(mat, row_labels, title, lbl_w, col_labels=None):
    if col_labels is None:
        col_labels = col_titles
    print(f"\n>>> {title}   {mat.shape[0]} × {mat.shape[1]}")
    header = ' ' * lbl_w + ''.join(f"{h:>{COL_W}}" for h in col_labels)
    print(header)
    with np.printoptions(precision=3, suppress=True, sign='+', floatmode='fixed'):
        for lbl, row in zip(row_labels, mat):
            print(f"{lbl.ljust(lbl_w)}" + ''.join(FMT_VAL.format(v) for v in row))


def show_state(state, label, tag):
    print(f"\n{label.ljust(LABEL_W_SUB - 2)}[{tag}]")
    for i, a in enumerate(state.flatten()):
        if abs(a) > 1e-6:
            print(f"  {FMT_VAL.format(a)}  {idx_arrow(i)}")


def show_H_state(state, label):
    out = H @ state
    print(f"H{label}")
    for i, a in enumerate(out.flatten()):
        if abs(a) > 1e-6:
            print(f"  {FMT_VAL.format(a)}  {idx_arrow(i)}")


"""
───────────────────────────────────────────────────────────────────────────────
Section 6 — Build the (S,mₛ) basis and block-diagonal H
Creates `ket_mat` whose rows are the ordered basis kets; then constructs
H_basis = ket_mat · H · ket_matᵀ, which is block-diagonal in (S,mₛ).
───────────────────────────────────────────────────────────────────────────────
"""
states = []
for S, ms, idx in order:
    for Sg, Msg, kets, _ in groups:
        if Sg == S and Msg == ms:
            states.append(kets[idx])
            break

ket_mat = np.vstack([st.flatten() for st in states])
H_basis = ket_mat @ H @ ket_mat.T

"""
───────────────────────────────────────────────────────────────────────────────
Section 7 — Diagnostic print-outs per symmetry sector
Shows each ket, the action of H on it, and the corresponding block of H in the
global basis. Comment these calls out if you only need final spectra.
───────────────────────────────────────────────────────────────────────────────
"""
for S, ms, kets, tags in groups:
    for ket, tag in zip(kets, tags):
        show_state(ket, f"|{S},{ms}⟩", tag)
        show_H_state(ket, f"|{S},{ms}⟩")
    H_block = np.vstack([(H @ k).flatten() for k in kets])
    H_block_in_basis = ket_mat @ H_block.T
    print_block(H_block_in_basis.T,
                [f"H|{S},{ms}⟩ [{t}]" for t in tags],
                f"H·Block (S={S}, m_s={ms})",
                LABEL_W_SUB,
                col_labels=sm_basis_labels)

"""
───────────────────────────────────────────────────────────────────────────────
Section 8 — Translation operator, symmetry blocks, and H in the T-eigenbasis
* Builds the cyclic-shift operator T in the computational basis and rotates it
  into the (S,mₛ) basis (T_sm).
* Prints every (S,mₛ) block of T.
* Diagonalises T_block sector by sector, prints the eigenkets, and transforms
  the Hamiltonian into that eigenbasis, giving `final_matrix`.
───────────────────────────────────────────────────────────────────────────────
"""


def shift_index(i, N=4):
    return (i >> 1) | ((i & 1) << (N - 1))


T_comp = np.zeros((16, 16))
for i in range(16):
    T_comp[shift_index(i), i] = 1.0

T_sm = ket_mat @ T_comp @ ket_mat.T

block_ranges = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 8), (8, 11), (11, 14),
    (14, 16)
]

print("\n>>> Transition matrix in (S, m_s) blocks:")
for start, end in block_ranges:
    labels = sm_basis_labels[start:end]
    title = f"T matrix   (S={order[start][0]}, m_s={order[start][1]})"
    print_block(T_sm[start:end, start:end].real, labels, title, LABEL_W_SUB, col_labels=labels)

final_matrix = np.zeros((16, 16), dtype=complex)

print("\n>>> Transition Matrix Eigenkets and Hamiltonian Action:\n")
for start, end in block_ranges:
    dim = end - start
    S, ms, _ = order[start]
    print(f">>> Block S={S}, m_s={ms} (dim {dim}×{dim})\n")

    if dim == 1:
        T_val = T_sm[start, start]
        H_val = H_basis[start, start]
        final_matrix[start, start] = H_basis[start, start]
        print(f"Eigenvalue |ψ0⟩: {T_sm[start, start].real:+.3f}{T_sm[start, start].imag:+.3f}i")
        print(f"|ψ0⟩ = {T_val.real:+.3f} |{S},{ms}⟩")
        print(f"H |ψ0⟩ = {H_val.real:+.3f} |{S},{ms}⟩\n")
        continue

    T_block = T_sm[start:end, start:end]
    H_block = H_basis[start:end, start:end]
    eigvals, eigvecs = np.linalg.eig(T_block)

    for k, val in enumerate(eigvals):
        vec = eigvecs[:, k]
        ket_str = " + ".join(
            f"({vec[j].real:+.3f}{vec[j].imag:+.3f}i) {sm_basis_labels[start + j]}"
            for j in range(dim) if abs(vec[j]) > 1e-6
        )
        print(f"Eigenvalue |ψ{k}⟩: {val.real:+.3f}{val.imag:+.3f}i")
        print(f"|ψ{k}⟩ = {ket_str}")

        H_vec = H_block @ vec
        if np.allclose(H_vec, 0):
            H_ket_str = "0"
        else:
            H_ket_str = " + ".join(
                f"({H_vec[j].real:+.3f}{H_vec[j].imag:+.3f}i) {sm_basis_labels[start + j]}"
                for j in range(dim) if abs(H_vec[j]) > 1e-6
            )
        print(f"H |ψ{k}⟩ = {H_ket_str}\n")

    V = eigvecs
    final_matrix[start:end, start:end] = V.conj().T @ H_block @ V

"""
───────────────────────────────────────────────────────────────────────────────
Section 9 — Grand matrices in the full basis
Prints H expressed in the original (S,mₛ) basis and in the final T-eigenbasis.
───────────────────────────────────────────────────────────────────────────────
"""
labels = sm_basis_labels
H_mat = np.vstack([(H @ st).flatten() for st in states])
print_block(ket_mat @ H_mat.T,
            [f"H{l}" for l in labels],
            "Grand H·matrix",
            LABEL_W_GR,
            col_labels=sm_basis_labels)

print_block(final_matrix.real,
            sm_basis_labels,
            "Final H in T-eigenvector basis",
            LABEL_W_GR,
            col_labels=sm_basis_labels)

"""
───────────────────────────────────────────────────────────────────────────────
Section 10 — Energy eigenvalues
Computes the sorted spectrum of the Hamiltonian in the momentum (T-eigen) basis.
───────────────────────────────────────────────────────────────────────────────
"""
eigenvalues = np.linalg.eigvalsh(final_matrix)
print("\n>>> Eigenvalues of the Matrix after Transition:")
for i, eig in enumerate(sorted(eigenvalues)):
    print(f"Eigenvalue {i:2d}: {eig:+.3f}")
