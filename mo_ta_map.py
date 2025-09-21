import random
import matplotlib.pyplot as plt
from copy import deepcopy
from path_quality_utility import generate_candidate_paths, select_optimal_path
from sensing_utility import compute_sensing_radius

# Hàm vẽ biểu đồ tại timestep 1 với khung chú thích thống nhất
def plot_map_timestep_1(uav_positions, sensor_positions, sensor_types, t_prev, current_time,
                        vsi_recovery_intervals, x_min, x_max, y_min, y_max,
                        G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                        P_tx_uav, G_tx_uav, G_rx_uav, lambda_S_plot, N0, SINR_threshold, Q, N, cm, km):
    """
    Vẽ bản đồ tại timestep 1 với tất cả chú thích trong một khung legend.

    Parameters:
        uav_positions: Danh sách vị trí các UAV [(x, y, z), ...]
        sensor_positions: Danh sách vị trí cảm biến [(x, y), ...]
        sensor_types: Danh sách loại cảm biến ['type1', 'type2', 'type3']
        t_prev: Thời gian thu thập gần nhất của mỗi cảm biến
        current_time: Thời gian hiện tại
        vsi_recovery_intervals: Thời gian phục hồi VSI cho từng loại cảm biến
        x_min, x_max, y_min, y_max: Giới hạn vùng vẽ
        G_St, G_Ur, lambda_S, P_st, P_U_min_r, h: Tham số tính bán kính cảm biến
        P_tx_uav, G_tx_uav, G_rx_uav, lambda_S_plot, N0, SINR_threshold, Q, N, cm, km: Tham số cho path_quality_utility
    """
    # Thiết lập kích thước và tiêu đề
    plt.figure(figsize=(10, 8))

    plt.xlabel("Tọa độ X (m)", fontsize=12)
    plt.ylabel("Tọa độ Y (m)", fontsize=12)

    # Vẽ cảm biến với màu sắc thể hiện trạng thái VSI
    max_delta = max(vsi_recovery_intervals.values())
    sensor_colors = {'type1': [], 'type2': [], 'type3': []}
    sensor_positions_by_type = {'type1': [], 'type2': [], 'type3': []}

    for i, ((sx, sy), stype) in enumerate(zip(sensor_positions, sensor_types)):
        delta_t = current_time - t_prev[i]
        recovery_interval = vsi_recovery_intervals[stype]
        normalized_vsi = min(delta_t / recovery_interval, 1.0)
        color = (1.0 - normalized_vsi, 0.0, normalized_vsi)  # Từ đỏ (VSI thấp) đến xanh (VSI cao)
        sensor_colors[stype].append(color)
        sensor_positions_by_type[stype].append((sx, sy))

    # Vẽ cảm biến với ký hiệu riêng cho từng loại
    plt.scatter([p[0] for p in sensor_positions_by_type['type1']],
                [p[1] for p in sensor_positions_by_type['type1']],
                c=sensor_colors['type1'], marker='o', s=50, alpha=0.8, label='Cảm biến Type 1 ')
    plt.scatter([p[0] for p in sensor_positions_by_type['type2']],
                [p[1] for p in sensor_positions_by_type['type2']],
                c=sensor_colors['type2'], marker='^', s=60, alpha=0.8, label='Cảm biến Type 2 ')
    plt.scatter([p[0] for p in sensor_positions_by_type['type3']],
                [p[1] for p in sensor_positions_by_type['type3']],
                c=sensor_colors['type3'], marker='x', s=70, alpha=0.8, label='Cảm biến Type 3 ')

    # Vẽ UAV và GBS
    for i, (ux, uy, _) in enumerate(uav_positions):
        if i == 0:
            plt.plot(ux, uy, 's', color='red', markersize=12, label='GBS (Trạm mặt đất)')
        else:
            plt.plot(ux, uy, 'D', color='blue', markersize=8, label=f'UAV' if i == 1 else None)

    # Vẽ vùng cảm biến của UAV
    sensing_r = compute_sensing_radius(G_St, G_Ur, lambda_S, P_st, P_U_min_r, h)
    for i, (ux, uy, _) in enumerate(uav_positions):
        if i > 0:  # Bỏ qua GBS
            circle = plt.Circle((ux, uy), sensing_r, color='blue', alpha=0.2, fill=True, linestyle='--',
                                label='Vùng thu thập dữ liệu cảm biến của UAV' if i == 1 else None)
            plt.gca().add_patch(circle)

    # Vẽ đường truyền tối ưu
    uav_dict = {idx: (pos[0], pos[1], pos[2]) for idx, pos in enumerate(uav_positions)}
    first_path_labeled = False
    for i in range(1, len(uav_positions)):
        candidate_paths = generate_candidate_paths(
            uav_dict, source=i, target=0,
            SINR_threshold=SINR_threshold,
            P_tx_uav=P_tx_uav, G_tx_uav=G_tx_uav, G_rx_uav=G_rx_uav,
            lambda_S=lambda_S_plot, N0=N0
        )
        if candidate_paths:
            best_path, _ = select_optimal_path(
                candidate_paths, uav_dict, gbs_pos=uav_dict[0],
                Q=Q, N=N, cm=cm, km=km,
                P_tx_uav=P_tx_uav, G_tx_uav=G_tx_uav, G_rx_uav=G_rx_uav,
                lambda_S=lambda_S_plot, N0=N0
            )
            if best_path:
                for idx in range(len(best_path) - 1):
                    s_id = best_path[idx]
                    e_id = best_path[idx + 1]
                    (sx, sy, _) = uav_dict[s_id]
                    (ex, ey, _) = uav_dict[e_id]
                    plt.plot([sx, ex], [sy, ey], 'g--', linewidth=1.5, label='Đường truyền tối ưu' if not first_path_labeled else None)
                    first_path_labeled = True

    # Thiết lập giới hạn trục
    plt.xlim(x_min - 100, x_max + 100)
    plt.ylim(y_min - 100, y_max + 100)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Thêm khung chú thích
    plt.legend(loc='upper right', fontsize=10, title='Chú thích', title_fontsize=12, bbox_to_anchor=(1.0, 1.0))
    # Thêm ghi chú về màu sắc VSI dưới khung chú thích


    # Lưu và hiển thị biểu đồ
    plt.savefig('timestep_1_map_with_legend.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Thiết lập tham số
    random.seed(123)
    x_min, x_max = 0.0, 2000.0
    y_min, y_max = 0.0, 2000.0
    h_fixed = 100.0
    current_time = 1.0

    # Tham số cho VSI
    vsi_recovery_intervals = {"type1": 3, "type2": 5, "type3": 7}

    # Tham số cho tính bán kính cảm biến
    G_St = 1.0
    G_Ur = 1.0
    lambda_S = 3e8 / 2.4e9
    P_st = 0.006
    P_U_min_r = 10 ** ((-76 - 30) / 10)
    h = h_fixed

    # Tham số cho kết nối UAV
    P_tx_uav = 0.25
    G_tx_uav = 1.0
    G_rx_uav = 1.0
    N0 = 1e-11
    SINR_threshold = 4
    Q = 3
    N = 20
    cm = 0.5
    km = 0.5

    # Tạo dữ liệu cảm biến
    cluster_centers = [(400, 1600), (1200, 1700), (500, 400), (1500, 600)]
    std_dev = 150.0

    def generate_sensor_clusters(seed, cluster_centers, std_dev):
        random.seed(seed)
        sensor_positions = []
        sensor_types = []
        for cx, cy in cluster_centers:
            for _ in range(25):
                x = random.gauss(cx, std_dev)
                y = random.gauss(cy, std_dev)
                sensor_positions.append((x, y))
                sensor_types.append("type1")
            for _ in range(25):
                x = random.gauss(cx, std_dev)
                y = random.gauss(cy, std_dev)
                sensor_positions.append((x, y))
                sensor_types.append("type2")
            for _ in range(25):
                x = random.gauss(cx, std_dev)
                y = random.gauss(cy, std_dev)
                sensor_positions.append((x, y))
                sensor_types.append("type3")
        return sensor_positions, sensor_types

    sensor_positions, sensor_types = generate_sensor_clusters(42, cluster_centers, std_dev)
    t_prev = [0.0] * len(sensor_positions)

    # Vị trí UAV ban đầu
    init_uav_positions = [
        (100.0, 100.0, 0.0),  # GBS
        (500.0, 200.0, h_fixed),
        (600.0, 1000.0, h_fixed),
        (1500.0, 500.0, h_fixed),
        (1200.0, 1200.0, h_fixed)
    ]

    # Vẽ biểu đồ
    plot_map_timestep_1(
        init_uav_positions, sensor_positions, sensor_types, t_prev, current_time,
        vsi_recovery_intervals, x_min, x_max, y_min, y_max,
        G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
        P_tx_uav, G_tx_uav, G_rx_uav, lambda_S, N0, SINR_threshold, Q, N, cm, km
    )

if __name__ == "__main__":
    main()