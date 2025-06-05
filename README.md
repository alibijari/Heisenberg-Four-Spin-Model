# 🌀 Heisenberg Spin-1/2 Four-Site Chain  
## Symmetry, Block Diagonalization & Quantum Spectrum

A modular Python package for **exploring symmetries, bases, and spectra** of the 1D Heisenberg model with four spin-½ sites (ring geometry).  
Includes code, detailed analysis, and theory for each symmetry sector—**perfect for advanced coursework, research, or quantum education.**

---

## ⚡️ Project Structure

| File          | Description                                                                          |
|---------------|--------------------------------------------------------------------------------------|
| `part_a.py`   | 🔹 Diagonalization in the **computational (product) basis** (16D).                   |
| `part_b.py`   | 🔹 **Block-diagonalization by total $S_z$** (magnetization sectors).                  |
| `part_c.py`   | 🔹 Diagonalization in the **coupled-spin $(S, m_s)$ basis** (SU(2) symmetry blocks). |
| `part_d.py`   | 🔹 Construction of the **translation/momentum operator** and its eigenbasis.          |
| `part_e.py`   | 🔹 **Full symmetry reduction**: $(S, m_s)$ & momentum blocks (most compact form).     |
| `part b.pdf`, `part c.pdf`, `part d & e.pdf` | Theory notes and step-by-step derivations for each part. |

---

##  📚  Project Structure

- **part_a.py:**  
  *Diagonalization in the computational (product) basis*  
  Constructs the Heisenberg Hamiltonian in the standard 16-dimensional computational basis (product of single-spin states), directly diagonalizes it, and outputs eigenvalues and eigenvectors. This serves as a baseline for comparison with more symmetry-adapted bases.

- **part_b.py:**  
  *Block-diagonalization by total S<sub>z</sub>*  
  Groups basis states according to the total spin-z projection (S<sub>z</sub>), reconstructs the Hamiltonian with this ordering, and extracts blocks for each S<sub>z</sub> sector. The block structure is analyzed, and eigenvalues for each sector are calculated.

- **part_c.py:**  
  *Coupled (S, m<sub>s</sub>) symmetry basis*  
  Constructs the coupled spin basis, grouping states by total spin quantum numbers (S, m<sub>s</sub>). This basis exploits the full SU(2) symmetry of the model, providing deeper physical insight. The Hamiltonian is expressed and block-diagonalized in this basis, with detailed output for each symmetry sector.

- **part_d.py:**  
  *Translation (momentum) operator and eigenbasis*  
  Introduces the cyclic translation (shift) operator on the ring and constructs its eigenstates (momentum basis) within each S<sub>z</sub> sector. The Hamiltonian is transformed into this momentum basis, and the spectral properties under translation symmetry are explored.

- **part_e.py:**  
  *Full symmetry analysis: combined (S, m<sub>s</sub>) and momentum blocks*  
  Combines all symmetry considerations. The Hamiltonian is block-diagonalized first by total spin (S, m<sub>s</sub>) and then by translation eigenstates (momentum), yielding the most compact and physical representation. This script provides the final spectra and comprehensive structural diagnostics.

---

## 🧩 Physical Model

The Heisenberg Hamiltonian for a 4-site ring (periodic boundary) is:

$$
\[
H = \vec{S}_1 \cdot \vec{S}_2 + \vec{S}_2 \cdot \vec{S}_3 + \vec{S}_3 \cdot \vec{S}_4 + \vec{S}_4 \cdot \vec{S}_1
\]
$$

- $\vec{S}_i$: Spin-½ operators on site $i$
- **Symmetries:**  
  - SU(2) (total spin conservation)  
  - Translation (momentum/eigenstate blocks)

---

## 🚀 Quick Start

**Requirements:**  
- Python 3.x  
- [NumPy](https://numpy.org/)

**How to use:**  
1. Clone/download the repo  
2. Run any part you want:  
   ```bash
   python part_a.py
   python part_b.py
   # ...and so on
