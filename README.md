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
