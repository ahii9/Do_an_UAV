# File: network_utility.py

import math
import random
def compute_dmax_uav_uav(G_tU, G_rU, lambda_S, P_tU, P_rU_min):
    numerator = G_tU * G_rU * (lambda_S ** 2) * P_tU
    denominator = (4 * math.pi) ** 2 * P_rU_min
    return math.sqrt(numerator / denominator)

def connectivity_index(distance, dmax):
    return 1 if distance <= dmax else 0

def distance_2d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_connectivity_matrix(uav_positions, dmax):
    n = len(uav_positions)
    matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = 1 if i==j else connectivity_index(distance_2d(uav_positions[i], uav_positions[j]), dmax)
    return matrix

def network_connectivity_utility(path, conn_matrix):
    util = 1
    for v,w in path:
        util *= conn_matrix[v][w]
        if util == 0:
            break
    return util


if __name__ == "__main__":
    # Simulation Parameters (Table 1)
    G_tU = 1.0              # Antenna transmit gain
    G_rU = 1.0              # Antenna receive gain
    freq = 2.4e9            # 2.4 GHz
    lambda_S = 3e8 / freq   # ≈0.125 m
    P_tU = 0.25           # 250 mW (within 150–300 mW)
    P_rU_min = 10**((-76 - 30)/10)  # -76 dBm → W

    # Compute max UAV‑UAV range
    dmax = compute_dmax_uav_uav(G_tU, G_rU, lambda_S, P_tU, P_rU_min)
    print(f"dmax_U-U = {dmax:.1f} m\n")

    # Generate 5 UAV positions uniformly over 2×2 km (converted to meters)
    random.seed(42)
    uav_positions = [(random.uniform(0,2000), random.uniform(0,2000), 100) for _ in range(5)]

    # Print UAV coordinates
    print("UAV Positions (x, y, z=100m):")
    for idx, pos in enumerate(uav_positions):
        print(f"  UAV{idx}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) m")
    print()

    # Build and display connectivity matrix
    conn_matrix = compute_connectivity_matrix(uav_positions, dmax)
    print("Connectivity Matrix (1=connected, 0=not connected):")
    for row in conn_matrix:
        print("  " + " ".join(str(v) for v in row))
