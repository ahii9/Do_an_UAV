from sensing_utility import compute_total_sensing_utility, sensors_in_uav_area, compute_sensing_radius
from path_quality_utility import communication_path_quality_utility, generate_candidate_paths, select_optimal_path, \
    compute_sensor_SINR_dict_using_sensing
from network_utility import compute_dmax_uav_uav, compute_connectivity_matrix, network_connectivity_utility
from vsi import compute_vsi
import math
import matplotlib.pyplot as plt
import numpy as np


def compute_total_utility(uav_positions, sensor_positions, sensor_types, current_time, t_prev, R,
                          Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                          P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N,
                          cm, km, Q, N0):
    """
    Hàm tính tổng tiện ích của hệ thống UAV dựa trên vị trí hiện tại.

    Parameters:
        - uav_positions (list): Danh sách tọa độ UAV [(x, y, z), ...].
        - sensor_positions (list): Danh sách tọa độ cảm biến [(x, y), ...].
        - sensor_types (list): Danh sách loại của từng cảm biến (e.g., ["type1", "type2", ...]).
        - current_time (float): Thời gian hiện tại để tính VSI.
        - t_prev (list): Danh sách thời gian trước đó của VSI cho từng cảm biến.
        - R (list): Danh sách hệ số VSI cho từng loại cảm biến [R_type1, R_type2, R_type3].
        - Vmax, Vmin (list): Danh sách giá trị VSI tối đa và tối thiểu cho từng loại cảm biến.
        - G_St, G_Ur, lambda_S, P_st, P_U_min_r, h: Tham số cho bán kính cảm biến.
        - P_tx_uav: Công suất phát của UAV (cho liên kết UAV→UAV).
        - G_tx_uav: Gain truyền của UAV.
        - G_rx_uav: Gain nhận của UAV.
        - SINR_threshold: Ngưỡng SINR tối thiểu cho liên kết.
        - N, cm, km, Q: Tham số cho tiện ích đường truyền.
        - N0: Noise floor.

    Returns:
        - total_utility (float): Tổng tiện ích của hệ thống.
        - t_prev (list): Danh sách thời gian thu thập được cập nhật.
        - collected_sensors (set): Tập hợp các cảm biến được thu thập.
    """
    # Tính bán kính vùng cảm biến
    sensing_radius = compute_sensing_radius(G_St, G_Ur, lambda_S, P_st, P_U_min_r, h)
    print(f"DEBUG: Sensing radius = {sensing_radius:.2f} m")

    # Xác định cảm biến trong vùng của từng UAV
    uav_sensor_map, sensor_uav_map = sensors_in_uav_area(sensor_positions, uav_positions, sensing_radius)
    print("DEBUG: uav_sensor_map =", uav_sensor_map)
    print("DEBUG: sensor_uav_map =", {k: v for k, v in sensor_uav_map.items() if len(v) > 0})

    # Tính kết nối UAV-UAV
    uav_dmax = compute_dmax_uav_uav(G_tx_uav, G_rx_uav, lambda_S, P_tx_uav, P_U_min_r)
    conn_matrix = compute_connectivity_matrix(uav_positions, uav_dmax)
    print(f"DEBUG: UAV-UAV max distance = {uav_dmax:.2f} m")
    print("DEBUG: Connectivity matrix =\n", np.array(conn_matrix))

    # Khởi tạo tổng tiện ích
    total_utility = 0
    num_uav = len(uav_positions)
    UAV_positions_dict = {i: pos for i, pos in enumerate(uav_positions)}

    # Tập hợp các cảm biến được thu thập
    collected_sensors = set()

    # Tính SINR thực tế giữa cảm biến và UAV (bao gồm can nhiễu từ các cảm biến khác)
    sensor_SINR_dict, _, _ = compute_sensor_SINR_dict_using_sensing(
        sensor_positions, uav_positions,
        G_St, G_Ur, lambda_S, P_st, P_U_min_r, h, N0
    )
    print("DEBUG: sensor_SINR_dict =", {k: [f"{v:.2f}" for v in val] for k, val in sensor_SINR_dict.items()})

    # Tính tiện ích cho từng UAV
    for u in range(num_uav):
        # Tính tiện ích cảm biến
        uav_sensor_data = {}
        for sensor_id in uav_sensor_map[u]:
            sensor_type = sensor_types[sensor_id]
            type_idx = int(sensor_type.replace("type", "")) - 1
            t_prev_sensor = t_prev[sensor_id]
            vsi = compute_vsi(current_time, t_prev_sensor, Vmax[type_idx], Vmin[type_idx], R[type_idx])
            N_ujk = len(sensor_uav_map[sensor_id])
            uav_sensor_data.setdefault(sensor_type, []).append((vsi, N_ujk))
            collected_sensors.add(sensor_id)
            print(f"DEBUG: UAV {u}, Sensor {sensor_id}, Type {sensor_type}, VSI = {vsi:.4f}, N_ujk = {N_ujk}")

        # Tham số c, d cho compute_total_sensing_utility
        c = 30  # Giá trị mặc định
        d = 40  # Giá trị mặc định
        sensing_util = compute_total_sensing_utility(uav_sensor_data, c, d)
        print(f"DEBUG: UAV {u} sensing utility = {sensing_util:.4f}")

        # Tính tiện ích đường truyền
        candidate_paths = generate_candidate_paths(
            UAV_positions_dict, source=u, target=0,
            SINR_threshold=SINR_threshold,
            P_tx_uav=P_tx_uav, G_tx_uav=G_tx_uav, G_rx_uav=G_rx_uav,
            lambda_S=lambda_S, N0=N0
        )
        print(f"DEBUG: UAV {u} candidate paths = {candidate_paths}")
        path_util = communication_path_quality_utility(
            u, UAV_positions_dict, sensor_SINR_dict,
            SINR_threshold, N, cm, km, Q,
            P_tx_uav, G_tx_uav, G_rx_uav, lambda_S, N0,
            candidate_paths
        )
        print(f"DEBUG: UAV {u} path quality utility = {path_util:.4f}")

        # Tính tiện ích kết nối mạng
        if candidate_paths:
            gbs_pos = UAV_positions_dict[0]
            best_path, best_product = select_optimal_path(
                candidate_paths, UAV_positions_dict, gbs_pos,
                Q, N, cm, km,
                P_tx_uav, G_tx_uav, G_rx_uav, lambda_S, N0
            )
            path_pairs = [(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)] if best_path else []
            conn_util = network_connectivity_utility(path_pairs, conn_matrix) if path_pairs else 0
            print(f"DEBUG: UAV {u} best path = {best_path}, connectivity utility = {conn_util:.4f}")
        else:
            conn_util = 0
            print(f"DEBUG: UAV {u} no valid paths, connectivity utility = {conn_util:.4f}")

        # Tổng hợp tiện ích cho UAV
        utility_u = sensing_util * path_util * conn_util
        total_utility += utility_u
        print(f"DEBUG: UAV {u} total utility contribution = {utility_u:.4f}")

    print(f"DEBUG: Tổng tiện ích tính trong compute_total_utility = {total_utility:.4f}")
    print(f"DEBUG: Số cảm biến được thu thập = {len(collected_sensors)}, IDs = {sorted(collected_sensors)}")

    # Cập nhật t_prev cho các cảm biến được thu thập
    for sensor_id in collected_sensors:
        t_prev[sensor_id] = current_time

    return total_utility, t_prev, collected_sensors


# ----------------------------------------------------
# MAIN FUNCTION FOR TESTING
# ----------------------------------------------------
if __name__ == "__main__":
    # Tham số cho tính VSI
    R = [3, 5, 7]  # Hệ số VSI cho 3 loại cảm biến
    Vmax = [10, 15, 20]  # VSI tối đa cho 3 loại cảm biến
    Vmin = [0.1, 0.15, 0.20]  # VSI tối thiểu cho 3 loại cảm biến

    # Tham số cho tính bán kính vùng cảm biến
    G_St = 1.0  # Hệ số khuếch đại ăng-ten cảm biến
    G_Ur = 1.0  # Hệ số khuếch đại ăng-ten UAV
    lambda_S = 3e8 / 2.4e9  # Bước sóng (tần số 2.4 GHz)
    P_st = 0.006  # Công suất phát của cảm biến (5 mW)
    P_U_min_r = 10 ** ((-76 - 30) / 10)  # Công suất thu tối thiểu (-76 dBm)
    h = 100  # Độ cao của UAV (m)

    # Tham số cho kết nối UAV-UAV
    G_tx_uav = 1.0  # Hệ số khuếch đại ăng-ten phát UAV
    G_rx_uav = 1.0  # Hệ số khuếch đại ăng-ten thu UAV
    P_tx_uav = 0.25  # Công suất phát của UAV (250 mW)

    # Tham số tính SNR và truyền dữ liệu
    SINR_threshold = 4  # Ngưỡng SINR tối thiểu cho liên kết
    N = 20  # Số bit trong packet
    cm = 0.5  # Hằng số cm trong tính PER
    km = 0.5  # Hằng số km trong tính PER
    Q = 3  # Số lần truyền lại tối đa

    # Noise floor
    N0 = 1e-11  # Giá trị nhiễu nền

    # Tham số đồ thị
    figsize_time_plot = (8, 6)  # Kích thước biểu đồ theo thời gian
    figsize_map = (10, 8)  # Kích thước bản đồ hệ thống

    # Đơn vị: mét
    uav_positions = [
        (0, 15, 0),  # GBS, id 0
        (200, 400, 100),
        (350, 800, 100),
        (250, 800, 100),
        (600, 900, 100),
        (0, 1691, 100)
    ]

    sensor_positions = [
        (200, 200), (500, 500), (800, 800),
        (200, 202), (500, 530), (800, 900),
        (805, 900), (810, 900), (860, 900),
        (650, 1000), (620, 1000), (800, 900),
        (1200, 1200), (1300, 1300), (1000, 1400),
        (1000, 1405), (1000, 1401), (1000, 1410),
        (1700, 1500), (1600, 1700), (1900, 1800),
        (700, 400), (900, 600), (300, 800),
        (1400, 900), (1800, 700), (1500, 400),
        (0, 1755), (0, 1759), (0, 1660)
    ]

    sensor_types = [f"type{(i % 3) + 1}" for i in range(len(sensor_positions))]
    t_prev = [0.0 for _ in range(len(sensor_positions))]

    # Tính toán và kiểm tra
    sensing_radius = compute_sensing_radius(G_St, G_Ur, lambda_S, P_st, P_U_min_r, h)
    uav_sensor_map, _ = sensors_in_uav_area(sensor_positions, uav_positions, sensing_radius)

    # Tính SINR từ sensor → UAV cho tất cả UAV (dùng cho visualization)
    all_sensor_SINR_dict, all_uav_sensor_map, _ = compute_sensor_SINR_dict_using_sensing(
        sensor_positions, uav_positions,
        G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
        N0
    )

    time_steps = range(0, 11, 2)  # t = 0, 2, 4, 6, 8, 10
    total_utilities = []
    for current_time in time_steps:
        total_util, t_prev, collected_sensors = compute_total_utility(
            uav_positions, sensor_positions, sensor_types, current_time, t_prev,
            R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
            P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
        )
        total_utilities.append(total_util)

    # Vẽ biểu đồ tổng tiện ích theo thời gian
    plt.figure(figsize=figsize_time_plot)
    plt.plot(time_steps, total_utilities, marker='o', linestyle='-', color='b')
    plt.title("Tổng tiện ích của hệ thống theo thời gian")
    plt.xlabel("Thời gian (t)")
    plt.ylabel("Tổng tiện ích")
    plt.grid(True)
    plt.savefig('total_utility_time.png')
    plt.show()

    # Vẽ bản đồ hệ thống
    fig, ax = plt.subplots(figsize=figsize_map)
    ax.set_title(f"Vị trí UAV, Vùng cảm biến và Đường truyền tối ưu đến GBS (t = {current_time})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle='--', alpha=0.5)

    # Vẽ Sensor (marker khác nhau theo sensor type)
    markers = {1: 'o', 2: '^', 3: 'x'}
    for idx, pos in enumerate(sensor_positions):
        sensor_type = (idx % 3) + 1
        plt.scatter(pos[0], pos[1], s=60, marker=markers[sensor_type], color='black', alpha=0.6)

    # Vẽ UAV và vùng cảm biến
    uav_colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, pos in enumerate(uav_positions):
        if i == 0:  # GBS
            ax.plot(pos[0], pos[1], marker='*', markersize=14, color='black', label="GBS (UAV0)")
        else:  # UAV
            color = uav_colors[(i - 1) % len(uav_colors)]
            ax.plot(pos[0], pos[1], marker='s', markersize=10, color=color, label=f"UAV{i}")
            # Vẽ vùng cảm biến
            circle = plt.Circle((pos[0], pos[1]), sensing_radius, color=color, alpha=0.1)
            ax.add_patch(circle)

    # Thêm một circle duy nhất vào legend để đại diện cho "Vùng cảm biến"
    circle_legend = plt.Circle((0, 0), 1, color='blue', alpha=0.1)
    ax.add_patch(circle_legend)
    circle_legend.set_visible(False)

    # Vẽ đường truyền tối ưu
    UAV_positions_dict = {i: pos for i, pos in enumerate(uav_positions)}
    path_drawn = False  # Để chỉ thêm label một lần cho đường tối ưu

    for i in range(1, len(uav_positions)):
        candidate_paths = generate_candidate_paths(
            UAV_positions_dict, source=i, target=0,
            SINR_threshold=SINR_threshold,
            P_tx_uav=P_tx_uav, G_tx_uav=G_tx_uav, G_rx_uav=G_rx_uav,
            lambda_S=lambda_S, N0=N0
        )

        if candidate_paths:
            gbs_pos = UAV_positions_dict[0]
            best_path, _ = select_optimal_path(
                candidate_paths, UAV_positions_dict, gbs_pos,
                Q, N, cm, km,
                P_tx_uav, G_tx_uav, G_rx_uav, lambda_S, N0
            )

            if best_path:
                path_coords = [UAV_positions_dict[node][:2] for node in best_path]
                xs, ys = zip(*path_coords)

                if not path_drawn:
                    ax.plot(xs, ys, 'g--', marker='o', markersize=8, linewidth=2, label="Optimal Path to GBS")
                    path_drawn = True
                else:
                    ax.plot(xs, ys, 'g--', marker='o', markersize=8, linewidth=2)

                # Vẽ mũi tên và hiển thị giá trị SINR
                for j in range(len(xs) - 1):
                    dx = xs[j + 1] - xs[j]
                    dy = ys[j + 1] - ys[j]

                    # Tính SINR trên liên kết UAV-UAV
                    from path_quality_utility import compute_link_sinr

                    pos_current = UAV_positions_dict[best_path[j]]
                    pos_next = UAV_positions_dict[best_path[j + 1]]
                    sinr_value = compute_link_sinr(
                        pos_current, pos_next,
                        P_tx_uav, G_tx_uav, G_rx_uav,
                        lambda_S, N0
                    )

                    ax.arrow(xs[j], ys[j], dx * 0.9, dy * 0.9, shape='full', color='green',
                             length_includes_head=True, head_width=20, head_length=20)

                    # Hiển thị giá trị SINR trên link
                    mid_x = (xs[j] + xs[j + 1]) / 2
                    mid_y = (ys[j] + ys[j + 1]) / 2
                    ax.text(mid_x, mid_y, f"{sinr_value:.2f} dB", fontsize=9, color='purple')

    # Thêm vùng cảm biến mẫu vào legend
    ax.plot([], [], 'o', color='white', markerfacecolor='blue', alpha=0.1, markersize=15, label="Vùng cảm biến")

    # Điều chỉnh hiển thị
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Loại bỏ nhãn trùng lặp
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))

    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('system_map.png')
    plt.show()