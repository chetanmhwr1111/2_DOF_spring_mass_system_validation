# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 10:26:06 2025

@author: cheta
"""

import numpy as np
from numpy.linalg import eig

# ---------------------------------------------------------
# 1. Define system parameters (You can modify these)
# ---------------------------------------------------------

# Masses
m1 = 2.0      # kg
m2 = 1.5      # kg

# Stiffnesses
k1 = 2000     # N/m (spring connected to ground)
k2 = 1500     # N/m (spring between masses)

# Damping coefficients
c1 = 15        # Ns/m (damper to ground)
c2 = 10        # Ns/m (damper between masses)

# Mass matrix
M = np.array([[m1, 0],
              [0,  m2]])

# Stiffness matrix
K = np.array([[k1 + k2, -k2],
              [-k2,      k2]])

# Damping matrix
C = np.array([[c1 + c2, -c2],
              [-c2,       c2]])

print("\nMass Matrix M:\n", M)
print("\nStiffness Matrix K:\n", K)
print("\nDamping Matrix C:\n", C)

# ---------------------------------------------------------
# 2. Compute undamped natural frequencies
#    Solve (K - ω²M)ϕ = 0
# ---------------------------------------------------------
eigvals_undamped, eigvecs = eig(np.linalg.inv(M) @ K)
wn = np.sqrt(np.real(eigvals_undamped))
fn = wn / (2*np.pi)

print("\nUndamped Natural Frequencies (rad/s):\n", wn)
print("\nUndamped Natural Frequencies (Hz):\n", fn)

# ---------------------------------------------------------
# 3. Form state-space representation for damped system
#    [0   I] [x] = [x']
#    [-M⁻¹K  -M⁻¹C] [x'] = [x'']
# ---------------------------------------------------------

Z = np.zeros((2,2))
I = np.eye(2)

A_top = np.hstack((Z, I))
A_bottom = np.hstack((-np.linalg.inv(M) @ K, -np.linalg.inv(M) @ C))
A = np.vstack((A_top, A_bottom))

# ---------------------------------------------------------
# 4. Solve state-space eigenproblem for damped system
# ---------------------------------------------------------
lambda_damped, modes_damped = eig(A)

# Natural frequency & damping ratio from complex eigenvalues
wn_damped = np.abs(lambda_damped)      # damped natural frequency (rad/s)
sigma = -np.real(lambda_damped)        # decay constant
omega_d = np.imag(lambda_damped)       # damped oscillatory term
zeta = -sigma / wn_damped              # damping ratio

print("\nDamped System Eigenvalues:\n", lambda_damped)
print("\nDamped Natural Frequencies (rad/s):\n", wn_damped)
print("\nDamping Ratios:\n", zeta)
