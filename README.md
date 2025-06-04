# Heisenberg Spin-1/2 Four-Site Chain — Symmetry, Block Diagonalization, and Spectral Analysis

## Overview

This repository provides a comprehensive Python-based analysis of the quantum Heisenberg spin-1/2 model on a ring of four sites (spins), with a particular focus on symmetry, basis construction, block-diagonalization, and spectrum computation. The code is designed for advanced studies in quantum many-body physics, including undergraduate, graduate, and research-level projects.

Each script explores a distinct basis or aspect of the problem, highlighting how different symmetries and representations affect the structure of the Hamiltonian and its eigenstates.

---

## Project Structure

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

## Physical Background

The **Heisenberg model** for four spin-1/2 particles on a ring is defined as:

H = S₁·S₂ + S₂·S₃ + S₃·S₄ + S₄·S₁

where Sᵢ are spin-1/2 operators and periodic boundary conditions are applied (S₅ ≡ S₁).
The model exhibits both SU(2) symmetry (total spin conservation) and translation symmetry, making it a textbook example for quantum symmetry and block-diagonalization.

---

## How to Use

1. **Requirements:**
   - Python 3.x
   - NumPy

2. **Run each script independently:**  
   Each script can be executed from the command line:
   ```bash
   python part_a.py
