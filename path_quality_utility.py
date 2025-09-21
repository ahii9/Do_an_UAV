import math
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from sensing_utility import compute_sensing_radius, sensors_in_uav_area

# Chuyển DEBUG = True để bật in thông tin chi tiết debug
DEBUG = 0


# -----------------------------------------
# Hàm hỗ trợ: Tính khoảng cách 3D
# -----------------------------------------
def distance_3d(pos1, pos2):
    """
    Tính khoảng cách 3D giữa hai điểm pos1 và pos2.
    pos1, pos2: tuple (x, y, z)
    """
    return math.sqrt((pos1[0] - pos2[0]) ** 2 +
                     (pos1[1] - pos2[1]) ** 2 +
                     (pos1[2] - pos2[2]) ** 2)


# -----------------------------------------
# Hàm tính SINR theo mô hình Friis
# -----------------------------------------
def compute_link_sinr(pos1, pos2, P_tx, G_tx, G_rx, lambda_val, N0, interference=0):
    """
    Tính SINR của liên kết giữa hai nút theo mô hình Friis.

    Công thức:
      PL = (G_tx * G_rx * lambda^2) / ((4 * pi * d)^2)
      SINR = (P_tx * PL) / (N0 + interference)

    Parameters:
        pos1: Tọa độ nút phát (x, y, z). Nếu có 2 phần tử sẽ mặc định z=0.
        pos2: Tọa độ nút nhận (x, y, z).
        P_tx: Công suất phát của nút truyền.
        G_tx: Gain của nút truyền.
        G_rx: Gain của nút nhận.
        lambda_val: Bước sóng.
        N0: Noise floor.
        interference: Tổng can nhiễu (mặc định = 0).

    Returns:
        float: Giá trị SINR.
    """
    # Nếu pos1 hoặc pos2 chỉ có 2 giá trị, thêm z=0.
    if len(pos1) == 2:
        pos1 = (pos1[0], pos1[1], 0)
    if len(pos2) == 2:
        pos2 = (pos2[0], pos2[1], 0)

    d = distance_3d(pos1, pos2)
    if d == 0:
        return float('inf')
    PL = (G_tx * G_rx * (lambda_val ** 2)) / ((4 * math.pi * d) ** 2)
    received_power = P_tx * PL
    sinr_linear= received_power / (N0 + interference)
    sinr=10 * math.log10(sinr_linear)
    if DEBUG:
        print(f"DEBUG: pos1={pos1}, pos2={pos2}, d={d:.3f}, PL={PL:.3e}, "
              f"received_power={received_power:.3e}, interference={interference:.3e}, SINR={sinr:.3f}")
    return sinr


# -----------------------------------------
# Tính SINR giữa sensor và UAV (bao gồm interference từ các sensor khác cùng vùng cảm biến)
# -----------------------------------------
def compute_sensor_SINR_dict_using_sensing(sensor_positions, uav_positions,
                                           G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                                           N0):
    """
    Tính SINR giữa sensor và UAV dựa trên vùng cảm biến và khoảng cách 3D,
    đồng thời tính can nhiễu từ các sensor khác cùng nằm trong vùng cảm biến của UAV.

    Parameters:
        sensor_positions: Danh sách vị trí các sensor (dạng (x,y))
        uav_positions: Danh sách vị trí các UAV (dạng (x,y,z))
        G_St: Gain của sensor (truyền)
        G_Ur: Gain của UAV (nhận)
        lambda_S: Bước sóng.
        P_st: Công suất phát của sensor.
        P_U_min_r: Công suất thu tối thiểu của UAV (dùng để tính sensing radius).
        h: Độ cao UAV (dùng để tính sensing radius).
        N0: Noise floor.

    Returns:
        tuple: (sensor_SINR_dict, uav_sensor_map, sensing_radius)
          - sensor_SINR_dict: dict {uav_id: [SINR của từng sensor liên kết với UAV]}
          - uav_sensor_map: dict ánh xạ UAV -> các sensor nằm trong vùng cảm biến.
          - sensing_radius: bán kính vùng cảm biến.
    """
    # 1. Tính bán kính vùng cảm biến từ sensing_utility.py
    sensing_radius = compute_sensing_radius(G_St, G_Ur, lambda_S, P_st, P_U_min_r, h)
    if DEBUG:
        print(f"DEBUG: Sensing radius = {sensing_radius:.3f} m")

    # 2. Xác định sensor nằm trong vùng cảm biến của từng UAV
    uav_sensor_map, _ = sensors_in_uav_area(sensor_positions, uav_positions, sensing_radius)
    if DEBUG:
        print("DEBUG: uav_sensor_map =", uav_sensor_map)

    # 3. Tính SINR cho mỗi sensor → UAV, có tính interference từ các sensor khác
    sensor_SINR_dict = {}
    for u in range(len(uav_positions)):
        sensor_SINR_dict[u] = []
        uav_pos = uav_positions[u]
        sensors_in_area = uav_sensor_map[u]
        if DEBUG:
            print(f"\nDEBUG: UAV {u} tại vị trí {uav_pos} có sensors trong vùng: {sensors_in_area}")
        for sid in sensors_in_area:
            sp = sensor_positions[sid]
            sensor_pos_3d = (sp[0], sp[1], 0)

            # use compute_link_sinr without interference
            sinr = compute_link_sinr(
                sensor_pos_3d, uav_pos,
                P_st, G_St, G_Ur, lambda_S, N0
            )

            if DEBUG:
                print(f"DEBUG: UAV {u} - Sensor {sid}: SINR={sinr:.3f} dB")
            sensor_SINR_dict[u].append(sinr)
    return sensor_SINR_dict, uav_sensor_map, sensing_radius


# -----------------------------------------
# Tính Packet Error Rate (dựa trên SINR)
# -----------------------------------------
def average_packet_error_rate(SINR, N, cm, km):
    """
    Tính PER trung bình trên kênh Rayleigh fading dựa trên SINR.

    Parameters:
        SINR: Tỉ số tín hiệu trên nhiễu.
        N: Số bit trong packet.
        cm: Hằng số cm.
        km: Hằng số km.

    Returns:
        float: Tỉ lệ lỗi gói tin trung bình.
    """
    if SINR <= 0:
        return 1.0
    a_N = math.log10(N * cm) / km
    b_N = 1.0 / km
    per = 1.0 - math.exp(-a_N / SINR) * gamma(1.0 + b_N / SINR)
    return per


# -----------------------------------------
# Tính Packet Success Rate (dựa trên SINR)
# -----------------------------------------
def packet_success_rate(SINR, N, cm, km, Q):
    """
    Tính xác suất truyền gói tin thành công trên một liên kết dựa trên SINR.

    Parameters:
        SINR: Tỉ số tín hiệu trên nhiễu.
        N: Số bit trong packet.
        cm: Hằng số cm.
        km: Hằng số km.
        Q: Số lần truyền lại tối đa.

    Returns:
        float: Xác suất truyền thành công.
    """
    PER = average_packet_error_rate(SINR, N, cm, km)
    return 1 - (PER ** Q)


# -----------------------------------------
# Sinh candidate paths (UAV→UAV) dựa trên SINR
# -----------------------------------------
def generate_candidate_paths(UAV_positions, source, target, SINR_threshold,
                             P_tx_uav, G_tx_uav, G_rx_uav, lambda_S, N0):
    """
    Sinh tất cả các đường đi đơn giản từ source đến target dựa trên đồ thị kết nối.

    Liên kết giữa các UAV được đánh giá dựa trên SINR (tính theo khoảng cách 3D):
      - Nếu SINR giữa hai UAV >= SINR_threshold thì coi là có liên kết.

    Parameters:
        UAV_positions: dict {node_id: (x,y,z)}
        source: ID của nút nguồn.
        target: ID của nút đích (ví dụ, GBS có id = 0).
        SINR_threshold: Ngưỡng SINR tối thiểu để xác định liên kết.
        P_tx_uav: Công suất phát của UAV (cho liên kết UAV→UAV).
        G_tx_uav: Gain của UAV (truyền).
        G_rx_uav: Gain của UAV (nhận).
        lambda_S: Bước sóng.
        N0: Noise floor.

    Returns:
        list: Danh sách candidate paths.
    """
    nodes = list(UAV_positions.keys())
    graph = {u: [] for u in nodes}
    # Xây dựng đồ thị liên kết dựa trên SINR
    for u in nodes:
        for v in nodes:
            if u != v:
                pos_u = UAV_positions[u]
                pos_v = UAV_positions[v]
                sinr = compute_link_sinr(pos_u, pos_v, P_tx_uav, G_tx_uav, G_rx_uav, lambda_S, N0)
                if sinr >= SINR_threshold:
                    graph[u].append(v)
    if DEBUG:
        print("\nDEBUG: UAV→UAV Graph (danh sách các nút kề):")
        for key, value in graph.items():
            print(f"  UAV {key}: {value}")
    # Sinh các đường đi đơn giản bằng DFS
    candidate_paths = []

    def dfs(current, path, visited):
        if current == target:
            candidate_paths.append(path.copy())
            return
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)

    dfs(source, [source], set([source]))
    if DEBUG:
        print("DEBUG: Candidate paths từ UAV", source, "đến UAV", target, ":", candidate_paths)
    return candidate_paths


# -----------------------------------------
# Chọn đường truyền tối ưu (theo tích xác suất thành công của UAV→UAV links)
# -----------------------------------------
def select_optimal_path(candidate_paths, UAV_positions, gbs_pos, Q, N, cm, km,
                        P_tx_uav, G_tx_uav, G_rx_uav, lambda_S, N0):
    """
    Chọn đường truyền tối ưu trong candidate_paths dựa trên tích xác suất truyền thành công.

    Parameters:
        candidate_paths: Danh sách các đường đi khả thi.
        UAV_positions: dict {node_id: (x,y,z)}.
        gbs_pos: Vị trí của GBS (dạng (x,y,z)).
        Q: Số lần truyền lại tối đa.
        N: Số bit trong packet.
        cm: Hằng số cm.
        km: Hằng số km.
        P_tx_uav: Công suất phát của UAV (UAV→UAV).
        G_tx_uav: Gain truyền của UAV.
        G_rx_uav: Gain nhận của UAV.
        lambda_S: Bước sóng.
        N0: Noise floor.

    Returns:
        tuple: (best_path, best_product)
    """
    best_path = None
    best_product = -np.inf
    if DEBUG:
        print("\nDEBUG: Tính tích xác suất trên từng candidate path:")
    for path in candidate_paths:
        product = 1.0
        if DEBUG:
            print(f"  Path: {path}")
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            pos_current = UAV_positions[current]
            pos_next = UAV_positions[next_node]
            sinr = compute_link_sinr(pos_current, pos_next, P_tx_uav, G_tx_uav, G_rx_uav, lambda_S, N0)
            ps = packet_success_rate(sinr, N, cm, km, Q)
            product *= ps
            if DEBUG:
                print(f"    Link {current}→{next_node}: SINR={sinr:.3f}, ps={ps:.3f}")
        if DEBUG:
            print(f"    Product cho path này = {product:.3e}")
        if product > best_product:
            best_product = product
            best_path = path
    if DEBUG:
        print("DEBUG: Best candidate path =", best_path, "với product =", best_product)
    return best_path, best_product


# -----------------------------------------
# Tính Communication Path Quality Utility cho UAV
# -----------------------------------------
def communication_path_quality_utility(uav_id, UAV_positions, sensor_SINR_dict,
                                       SINR_threshold, N, cm, km, Q,
                                       P_tx_uav, G_tx_uav, G_rx_uav, lambda_S, N0,
                                       candidate_paths=None):
    """
    Tính Communication Path Quality Utility của UAV dựa trên:
      - Xác suất truyền thành công từ sensor đến UAV.
      - Xác suất truyền thành công dọc đường từ UAV đến GBS (id = 0).

    Parameters:
        uav_id: ID của UAV cần tính utility.
        UAV_positions: dict {node_id: (x,y,z)}.
        sensor_SINR_dict: dict SINR từ sensor đến UAV.
        SINR_threshold: Ngưỡng SINR tối thiểu (cho UAV→UAV).
        N: Số bit trong packet.
        cm: Hằng số cm.
        km: Hằng số km.
        Q: Số lần truyền lại tối đa.
        P_tx_uav: Công suất phát của UAV (UAV→UAV).
        G_tx_uav: Gain truyền của UAV.
        G_rx_uav: Gain nhận của UAV.
        lambda_S: Bước sóng.
        N0: Noise floor.
        candidate_paths: (Tùy chọn) danh sách candidate paths.

    Returns:
        float: Giá trị utility của UAV.
    """
    # a) Tính PS_sensor: Tích xác suất truyền từ sensor đến UAV
    sensor_list = sensor_SINR_dict.get(uav_id, [])
    if not sensor_list:
        PS_sensor = 0.0
    else:
        # 1) SINR trung bình từ tất cả sensor về UAV u
        avg_sinr = sum(sensor_list) / len(sensor_list)

        # 2) PER trung bình dựa trên avg_sinr
        per_avg = average_packet_error_rate(avg_sinr, N, cm, km)

        # 3) Xác suất thành công sau Q lần retransmission
        PS_sensor = 1.0 - (per_avg ** Q)

        if DEBUG:
            print(f"DEBUG: UAV {uav_id} - SINR trung bình = {avg_sinr:.3f}")
            print(f"DEBUG: PER trung bình = {per_avg:.3e} → PS_sensor = {PS_sensor:.3f}")

    # b) Sinh candidate paths nếu chưa có
    if candidate_paths is None:
        candidate_paths = generate_candidate_paths(
            UAV_positions, source=uav_id, target=0,
            SINR_threshold=SINR_threshold,
            P_tx_uav=P_tx_uav, G_tx_uav=G_tx_uav, G_rx_uav=G_rx_uav,
            lambda_S=lambda_S, N0=N0
        )
    else:
        candidate_paths = [path for path in candidate_paths if path[0] == uav_id and path[-1] == 0]
    if len(candidate_paths) == 0:
        return 0
    # c) Tìm đường truyền tối ưu từ UAV đến GBS
    gbs_pos = UAV_positions[0]
    best_path, best_product = select_optimal_path(candidate_paths, UAV_positions, gbs_pos,
                                                  Q, N, cm, km,
                                                  P_tx_uav, G_tx_uav, G_rx_uav, lambda_S, N0)
    return PS_sensor * best_product


# -----------------------------------------
# ------------------- MAIN -------------------
# -----------------------------------------
if __name__ == "__main__":
    # --- Tham số cho liên kết sensor → UAV ---
    G_St = 1.0  # gain sensor (tx)
    G_Ur = 1.0  # gain UAV (rx)
    lambda_S = 3e8 / 2.4e9  # bước sóng ≈ 0.125 m
    P_st = 0.006  # công suất sensor
    P_U_min_r = 10 ** ((-76 - 30) / 10)  # ngưỡng thu UAV
    h_sensor = 100  # độ cao UAV (m)



    N0 = 1e-11  # W

    # --- Tham số PER và đường truyền ---
    N = 20  # bit/packet
    cm = 0.5
    km = 0.5
    Q = 3
    SINR_threshold = 2  # dB

    # --- Tham số cho liên kết UAV → UAV ---
    P_tx_uav = 0.25  # 250 mW
    G_tx_uav = 1.0
    G_rx_uav = 1.0

    # --- Vị trí của UAV ---
    # Dictionary với id UAV và tọa độ (x, y, z). GBS có id = 0.
    UAV_positions = {
        0: (0, 15, 0),  # GBS, id 0
        1: (200, 400, 100),
        2: (350, 800, 100),
        3: (800, 600, 100),
        4: (600, 900, 100),
        5: (0, 1790, 100)
    }

    # --- Vị trí của sensor ---
    # Danh sách 15 sensor với tọa độ (x,y) (giả sử z=0)
    sensor_positions = [
        (200, 200), (500, 500), (800, 800),
        (200, 202), (500, 530), (800, 900),
        (805, 900),(810, 900),(860, 900),
        (650, 1000),(620, 1000),(800, 900),
        (1200, 1200), (1300, 1300), (1000, 1400),
        (1000, 1405), (1000, 1401), (1000, 1410),
        (1700, 1500), (1600, 1700), (1900, 1800),
        (700, 400), (900, 600), (300, 800),
        (1400, 900), (1800, 700), (1500, 400),
        (0,1755),(0,1759),(0,1660)
    ]


    sensor_SINR_dict, uav_sensor_map, sensing_radius = compute_sensor_SINR_dict_using_sensing(
        sensor_positions, list(UAV_positions.values()),
        G_St, G_Ur, lambda_S, P_st, P_U_min_r, h_sensor,
        N0
    )

    print("\n---------- Kết quả tính SINR từ sensor → UAV ----------")
    print("Sensing Radius (m):", sensing_radius)
    for u in sensor_SINR_dict:
        print(f"UAV {u} SINR values from sensors: {sensor_SINR_dict[u]}")

    # --- Sinh candidate paths từ UAV 5 đến GBS (id 0) dựa trên liên kết UAV→UAV ---
    candidate_paths = generate_candidate_paths(
        UAV_positions, source=5, target=0,
        SINR_threshold=SINR_threshold,
        P_tx_uav=P_tx_uav, G_tx_uav=G_tx_uav, G_rx_uav=G_rx_uav,
        lambda_S=lambda_S, N0=N0
    )
    print("\n---------- Candidate Paths từ UAV 5 đến GBS ----------")
    print(candidate_paths)

    # --- Tính Communication Path Quality Utility cho UAV 5 ---
    comm_util = communication_path_quality_utility(
        5, UAV_positions, sensor_SINR_dict,
        SINR_threshold, N, cm, km, Q,
        P_tx_uav, G_tx_uav, G_rx_uav, lambda_S, N0,
        candidate_paths
    )
    print("\nCommunication Path Quality Utility cho UAV 5:", comm_util)

    # --- Chọn đường truyền tối ưu ---
    gbs_pos = UAV_positions[0]
    best_path, best_product = select_optimal_path(
        candidate_paths, UAV_positions, gbs_pos,
        Q, N, cm, km,
        P_tx_uav, G_tx_uav, G_rx_uav, lambda_S, N0
    )
    print("\n---------- Kết quả chọn đường truyền tối ưu ----------")
    print("Đường truyền tối ưu được chọn:", best_path)
    print("Tích xác suất truyền thành công của đường:", best_product)

    # -----------------------------------------
    # Vẽ sơ đồ: Vẽ UAV, Sensor và đường truyền tối ưu
    # -----------------------------------------
    figsize = (10, 8)
    plt.figure(figsize=figsize)

    # Vẽ UAV
    for node_id, pos in UAV_positions.items():
        x, y, z = pos
        if node_id == 0:
            plt.scatter(x, y, s=200, marker='s', color='red', edgecolors='black', label='GBS')
            plt.text(x + 20, y + 20, 'GBS', fontsize=9)
        else:
            plt.scatter(x, y, s=100, marker='^', color='blue', edgecolors='black')
            plt.text(x + 20, y + 20, f'UAV {node_id}', fontsize=9)

    # Vẽ Sensor (marker khác nhau theo sensor type)
    markers = {1: 'o', 2: '^', 3: 'x'}
    for idx, pos in enumerate(sensor_positions):
        sensor_type = (idx % 3) + 1
        plt.scatter(pos[0], pos[1], s=60, marker=markers[sensor_type], color='black', alpha=0.6)

    # Vẽ đường truyền tối ưu nếu có
    if best_path:
        path_coords = [UAV_positions[node][:2] for node in best_path]
        xs, ys = zip(*path_coords)
        plt.plot(xs, ys, 'g--', marker='o', markersize=8, linewidth=2, label='Optimal Path')
        # Vẽ mũi tên và in ra giá trị SINR của từng link trên đường đi
        for i in range(len(xs) - 1):
            # Tính SINR trên liên kết
            pos_current = UAV_positions[best_path[i]]
            pos_next = UAV_positions[best_path[i + 1]]
            sinr_value = compute_link_sinr(pos_current, pos_next, P_tx_uav, G_tx_uav, G_rx_uav, lambda_S, N0)
            plt.arrow(xs[i], ys[i],
                      (xs[i + 1] - xs[i]) * 0.9,
                      (ys[i + 1] - ys[i]) * 0.9,
                      shape='full', color='green', length_includes_head=True,
                      head_width=20, head_length=20)
            mid_x = (xs[i] + xs[i + 1]) / 2
            mid_y = (ys[i] + ys[i + 1]) / 2
            plt.text(mid_x, mid_y, f"{sinr_value:.2f}", fontsize=9, color='purple')

    plt.title('UAV Positions and Optimal Communication Path to GBS (SINR-based)')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.axis('equal')
    plt.show()
