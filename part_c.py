import numpy as np

# ======================================
# 1) single-spin basis |↑⟩ , |↓⟩
# ======================================

ket_up = np.array([[1], [0]])
ket_down = np.array([[0], [1]])


# ======================================
# 2) two-spin triplet & singlet states
# ======================================

def triplet_states():
    Tp = np.kron(ket_up, ket_up)
    T0 = (1 / np.sqrt(2)) * (np.kron(ket_up, ket_down) + np.kron(ket_down, ket_up))
    Tm = np.kron(ket_down, ket_down)
    return Tp, T0, Tm


def singlet_state():
    return (1 / np.sqrt(2)) * (np.kron(ket_up, ket_down) - np.kron(ket_down, ket_up))


Tp, T0, Tm = triplet_states()
S0_two = singlet_state()

# ======================================
# 3) four-spin kets grouped by (S, m_s)  + coupling tags
# ======================================

groups = [
    ('2', '+2', [np.kron(Tp, Tp)], ['T⊗T']),
    ('2', '+1', [(1 / np.sqrt(2)) * (np.kron(Tp, T0) + np.kron(T0, Tp))], ['T⊗T']),
    ('2', '0', [(1 / np.sqrt(6)) * (np.kron(Tp, Tm) + 2 * np.kron(T0, T0) + np.kron(Tm, Tp))], ['T⊗T']),
    ('2', '-1', [(1 / np.sqrt(2)) * (np.kron(Tm, T0) + np.kron(T0, Tm))], ['T⊗T']),
    ('2', '-2', [np.kron(Tm, Tm)], ['T⊗T']),
    ('1', '+1', [
        (1 / np.sqrt(2)) * (np.kron(Tp, T0) - np.kron(T0, Tp)),
        np.kron(Tp, S0_two),
        np.kron(S0_two, Tp)
    ], ['T⊗T', 'T⊗S', 'S⊗T']),
    ('1', '0', [
        (1 / np.sqrt(2)) * (np.kron(Tp, Tm) - np.kron(Tm, Tp)),
        np.kron(T0, S0_two),
        np.kron(S0_two, T0)
    ], ['T⊗T', 'T⊗S', 'S⊗T']),
    ('1', '-1', [
        (1 / np.sqrt(2)) * (np.kron(T0, Tm) - np.kron(Tm, T0)),
        np.kron(Tm, S0_two),
        np.kron(S0_two, Tm)
    ], ['T⊗T', 'T⊗S', 'S⊗T']),
    ('0', '0', [
        (1 / np.sqrt(3)) * (np.kron(Tp, Tm) - np.kron(T0, T0) + np.kron(Tm, Tp)),
        np.kron(S0_two, S0_two)
    ], ['T⊗T', 'S⊗S'])
]

order = [
    ('2', '+2', 0),
    ('2', '+1', 0),
    ('2', '0', 0),
    ('2', '-1', 0),
    ('2', '-2', 0),
    ('1', '+1', 0), ('1', '+1', 1), ('1', '+1', 2),
    ('1', '0', 0), ('1', '0', 1), ('1', '0', 2),
    ('1', '-1', 0), ('1', '-1', 1), ('1', '-1', 2),
    ('0', '0', 0), ('0', '0', 1)
]

# Define the (S, m_s) basis labels
sm_basis_labels = [
    '|2,+2⟩', '|2,+1⟩', '|2,0⟩', '|2,-1⟩', '|2,-2⟩',
    '|1,+1⟩', '|1,+1⟩', '|1,+1⟩',
    '|1,0⟩', '|1,0⟩', '|1,0⟩',
    '|1,-1⟩', '|1,-1⟩', '|1,-1⟩',
    '|0,0⟩', '|0,0⟩'
]

# ======================================
# 4) Heisenberg Hamiltonian
# ======================================

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

# ======================================
# 5) pretty-printing helpers
# ======================================

COL_W = 8
FMT_VAL = f"{{:+{COL_W}.3f}}"
LABEL_W_SUB = 16  # for per-subspace rows
LABEL_W_GR = 18  # for grand matrix rows

_arrow = {0: '↑', 1: '↓'}


def idx_arrow(i, N=4):
    bits = [(i >> (N - 1 - j)) & 1 for j in range(N)]
    return '|' + ''.join(_arrow[b] for b in bits) + '⟩'


col_titles = [idx_arrow(i) for i in range(16)]


def print_block(mat, row_labels, title, lbl_w, col_labels=None):
    if col_labels is None:
        col_labels = col_titles
    print(f"\n>>> {title}   {mat.shape[0]} × {mat.shape[1]}")
    header_str = ' ' * lbl_w + ''.join(f"{h:>{COL_W}}" for h in col_labels)
    print(header_str)
    with np.printoptions(precision=3, suppress=True, sign='+', floatmode='fixed'):
        for lbl, row in zip(row_labels, mat):
            print(f"{lbl.ljust(lbl_w)}" + ''.join(FMT_VAL.format(v) for v in row))


# detailed per-state prints
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


# ======================================
# 6) per-subspace output
# ======================================

# Build the ket_mat for basis transformation early
states = []
for S, ms, idx in order:
    for Sg, Msg, kets, tags in groups:
        if Sg == S and Msg == ms:
            states.append(kets[idx])
            break
ket_mat = np.vstack([st.flatten() for st in states])  # 16 × 16

for S, ms, kets, tags in groups:
    for ket, tag in zip(kets, tags):
        lbl = f"|{S},{ms}⟩"
        show_state(ket, lbl, tag)
        show_H_state(ket, lbl)

    # Transform the H·block into the (S, m_s) basis
    H_block = np.vstack([(H @ k).flatten() for k in kets])
    H_block_in_basis = ket_mat @ H_block.T  # Transform to (S, m_s) basis
    print_block(H_block_in_basis.T, [f"H|{S},{ms}⟩ [{t}]" for t in tags],
                f"H·Block (S={S}, m_s={ms})", LABEL_W_SUB, col_labels=sm_basis_labels)

# ======================================
# 7) build the grand 16×16 matrices
# ======================================

labels = [sm_basis_labels[i] for i in range(len(order))]
H_mat = np.vstack([(H @ st).flatten() for st in states])
H_basis = ket_mat @ H @ ket_mat.T  # Hamiltonian in (S, m_s) basis

# Print all grand matrices in the (S, m_s) basis
print_block(ket_mat @ H_mat.T, [f"H{l}" for l in labels], "Grand H·matrix (ordered)", LABEL_W_GR,
            col_labels=sm_basis_labels)

# ======================================
# 8) Eigenvalues
# ======================================

eigenvalues = np.linalg.eigvalsh(H_basis)
print("\n>>> Eigenvalues of the Hamiltonian in (S, m_s) basis:")
for i, eig in enumerate(sorted(eigenvalues)):
    print(f"Eigenvalue {i:2d}: {eig:+.3f}")
