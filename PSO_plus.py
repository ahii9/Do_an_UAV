import random
import math
import matplotlib.pyplot as plt
from copy import deepcopy
from path_quality_utility import compute_sensor_SINR_dict_using_sensing, compute_link_sinr

# --------- IMPORT từ các module sẵn có ---------
from total_utility import compute_total_utility
from path_quality_utility import generate_candidate_paths, select_optimal_path
from sensing_utility import compute_sensing_radius, sensors_in_uav_area

# -----------------------------
# 3) Sinh dữ liệu cảm biến
# -----------------------------
def generate_sensor_clusters(seed, cluster_centers, std_dev):
    """
    Sinh 4 cụm, mỗi cụm 75 sensor => tổng 300 sensor
    (3 loại: type1, type2, type3, mỗi loại 100 sensor)

    Parameters:
        seed: Giá trị seed cho random
        cluster_centers: Danh sách các tọa độ trung tâm của cụm
        std_dev: Độ lệch chuẩn của phân phối cảm biến trong cụm

    Returns:
        tuple: (sensor_positions, sensor_types)
    """
    random.seed(seed)
    sensor_positions = []
    sensor_types = []

    for cx, cy in cluster_centers:
        # Mỗi cụm: 75 sensor => 25 sensor/loại
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

# -----------------------------
# 4) Lớp Particle (cho PSO)
# -----------------------------
class Particle:
    def __init__(self, num_uavs, init_positions, x_min, x_max, y_min, y_max, v_min, v_max, h_fixed):
        """
        Nếu init_positions != None, ta khởi tạo quanh vị trí đó (hoặc đúng vị trí).

        Parameters:
            num_uavs: Số lượng UAV
            init_positions: Vị trí ban đầu của các UAV
            x_min, x_max, y_min, y_max: Giới hạn vùng di chuyển
            v_min, v_max: Giới hạn vận tốc
            h_fixed: Độ cao cố định của UAV
        """
        self.num_uavs = num_uavs
        self.position = []
        self.velocity = []
        if init_positions is not None:
            # Khởi tạo quanh vị trí cũ (hoặc đúng vị trí)
            for i in range(num_uavs):
                if i == 0:
                    # UAV0 = GBS
                    self.position.append(init_positions[i])
                    self.velocity.append((0.0, 0.0))
                else:
                    x0, y0, z0 = init_positions[i]
                    # Giữ nguyên vị trí ban đầu - không thêm nhiễu
                    self.position.append((x0, y0, z0))
                    vx = random.uniform(v_min, v_max)
                    vy = random.uniform(v_min, v_max)
                    self.velocity.append((vx, vy))
        else:
            # Khởi tạo hoàn toàn ngẫu nhiên
            for i in range(num_uavs):
                if i == 0:
                    self.position.append((0.0, 0.0, 0.0))
                    self.velocity.append((0.0, 0.0))
                else:
                    x = random.uniform(x_min, x_max)
                    y = random.uniform(y_min, y_max)
                    self.position.append((x, y, h_fixed))
                    vx = random.uniform(v_min, v_max)
                    vy = random.uniform(v_min, v_max)
                    self.velocity.append((vx, vy))

        self.best_position = deepcopy(self.position)
        self.best_fitness = -math.inf
        self.fitness = -math.inf

def evaluate_fitness(particle, sensor_positions, sensor_types, t_prev, current_time,
                     R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                     P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0):
    """
    Tính giá trị hàm mục tiêu (total utility) cho một hạt.

    Parameters:
        particle: Đối tượng hạt cần tính fitness
        sensor_positions, sensor_types: Vị trí và loại cảm biến
        t_prev: Thời gian thu thập gần nhất của mỗi cảm biến
        current_time: Thời gian hiện tại
        Các tham số còn lại: Các tham số tính toán utility

    Returns:
        tuple: (total_util, new_t_prev)
    """
    t_prev_copy = deepcopy(t_prev)


    total_util, _, _, _ = compute_total_utility(
        particle.position, sensor_positions, sensor_types,
        current_time, t_prev_copy,
        R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
        P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
    )
    return total_util

def update_velocity_position(particle, global_best, omega, c1, c2, speed_factor, v_min, v_max, x_min, x_max, y_min, y_max):
    """
    Cập nhật velocity, position cho particle.
    Tăng tốc di chuyển bằng cách nhân SPEED_FACTOR.

    Parameters:
        particle: Đối tượng hạt cần cập nhật
        global_best: Vị trí tốt nhất toàn cục
        omega: Hệ số quán tính
        c1, c2: Hệ số nhận thức và xã hội
        speed_factor: Hệ số tăng tốc
        v_min, v_max: Giới hạn vận tốc
        x_min, x_max, y_min, y_max: Giới hạn vùng di chuyển
    """
    for i in range(particle.num_uavs):
        if i == 0:
            # GBS cố định
            particle.position[i] = particle.position[0]
            particle.velocity[i] = (0.0, 0.0)
            continue

        (x, y, z) = particle.position[i]
        (vx, vy) = particle.velocity[i]
        (lbx, lby, _) = particle.best_position[i]
        (gbx, gby, _) = global_best[i]

        r1 = random.random()
        r2 = random.random()

        new_vx = omega * vx + c1 * r1 * (lbx - x) + c2 * r2 * (gbx - x)
        new_vy = omega * vy + c1 * r1 * (lby - y) + c2 * r2 * (gby - y)

        # Nhân SPEED_FACTOR để UAV di chuyển nhanh hơn
        new_vx *= speed_factor
        new_vy *= speed_factor

        # Giới hạn vận tốc
        new_vx = max(v_min, min(new_vx, v_max))
        new_vy = max(v_min, min(new_vy, v_max))

        # Cập nhật vị trí
        nx = x + new_vx
        ny = y + new_vy
        nx = max(x_min, min(nx, x_max))
        ny = max(y_min, min(ny, y_max))

        particle.position[i] = (nx, ny, z)
        particle.velocity[i] = (new_vx, new_vy)

# -----------------------------
# 5) Hàm chạy PSO cho tối ưu hóa vị trí
# -----------------------------
def run_pso(uav_init_positions, sensor_positions, sensor_types, t_prev, current_time, current_utility,
            num_particles, max_iterations, convergence_threshold,
            omega, c1, c2, speed_factor,
            x_min, x_max, y_min, y_max, v_min, v_max, h_fixed,
            R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
            P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0):
    """
    Chạy PSO theo đúng logic mô tả trong thuật toán:
    - Lặp cho đến khi số vòng lặp đạt max hoặc độ chênh lệch tiện ích < epsilon
    - Duy trì local best và global best
    - Cập nhật vận tốc và vị trí

    Parameters:
        uav_init_positions: Vị trí ban đầu của các UAV
        sensor_positions, sensor_types: Vị trí và loại cảm biến
        t_prev: Thời gian thu thập gần nhất của mỗi cảm biến
        current_time: Thời gian hiện tại
        current_utility: Tiện ích hiện tại
        num_particles: Số lượng hạt trong quần thể
        max_iterations: Số vòng lặp tối đa
        convergence_threshold: Ngưỡng hội tụ
        omega, c1, c2: Các hệ số cập nhật vận tốc
        speed_factor: Hệ số tăng tốc
        x_min, x_max, y_min, y_max: Giới hạn vùng di chuyển
        v_min, v_max: Giới hạn vận tốc
        h_fixed: Độ cao cố định của UAV
        Các tham số còn lại: Các tham số tính toán utility

    Returns:
        tuple: (global_best_fitness, global_best_position, best_t_prev, better_than_current)
    """
    swarm = [Particle(num_uavs=5, init_positions=uav_init_positions,
                      x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                      v_min=v_min, v_max=v_max, h_fixed=h_fixed)
             for _ in range(num_particles)]
    global_best_fitness = -math.inf
    global_best_position = None
    best_t_prev = t_prev

    # Theo dõi tiện ích vòng lặp trước đó cho điều kiện hội tụ
    prev_utility_iteration = -math.inf

    # Vòng lặp PSO
    for iteration in range(max_iterations):
        current_utility_iteration = -math.inf

        # Lặp qua từng hạt
        for p in swarm:
            # Tính tiện ích
            fitness = evaluate_fitness(p, sensor_positions, sensor_types, t_prev, current_time,
                                                   R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                                                   P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0)
            p.fitness = fitness

            # Cập nhật local best
            if fitness > p.best_fitness:
                p.best_fitness = fitness
                p.best_position = deepcopy(p.position)

            # Cập nhật global best
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = deepcopy(p.position)


            # Theo dõi tiện ích hiện tại tốt nhất
            current_utility_iteration = max(current_utility_iteration, fitness)

        # Cập nhật vận tốc và vị trí cho tất cả hạt
        if global_best_position is not None:
            for p in swarm:
                update_velocity_position(p, global_best_position,
                                         omega, c1, c2, speed_factor,
                                         v_min, v_max, x_min, x_max, y_min, y_max)

        # Kiểm tra điều kiện hội tụ
        if abs(current_utility_iteration - prev_utility_iteration) < convergence_threshold:
            print(f"PSO hội tụ sau {iteration+1} vòng lặp")
            break

        prev_utility_iteration = current_utility_iteration

    # Kiểm tra xem tiện ích tìm được có tốt hơn tiện ích hiện tại không
    better_than_current = global_best_fitness > current_utility

    if better_than_current:
        print(f"Đã tìm được vị trí tốt hơn: {global_best_fitness:.4f} > {current_utility:.4f}")
    else:
        print(f"Không tìm được vị trí tốt hơn: {global_best_fitness:.4f} <= {current_utility:.4f}")

    return global_best_fitness, global_best_position, better_than_current

# -----------------------------
# 6) Hàm reset VSI cho cảm biến vừa được thu thập
# -----------------------------
def update_sensors_after_collection(uav_positions, sensor_positions, t_prev, current_time,
                                    G_St, G_Ur, lambda_S, P_st, P_U_min_r, h):
    """
    Mọi sensor nằm trong vùng phủ sóng của UAV => t_prev[sensor_id] = current_time
    => VSI reset về Vmin cho sensor đó.

    Parameters:
        uav_positions: Vị trí các UAV
        sensor_positions: Vị trí các cảm biến
        t_prev: Thời gian thu thập gần nhất của mỗi cảm biến
        current_time: Thời gian hiện tại
        G_St, G_Ur, lambda_S, P_st, P_U_min_r, h: Tham số tính bán kính cảm biến

    Returns:
        tuple: (t_prev, collected_sensors)
    """
    sensing_r = compute_sensing_radius(G_St, G_Ur, lambda_S, P_st, P_U_min_r, h)
    uav_sensor_map, _ = sensors_in_uav_area(sensor_positions, uav_positions, sensing_r)

    collected_sensors = []
    for _, sensor_list in uav_sensor_map.items():
        for s_id in sensor_list:
            t_prev[s_id] = current_time
            collected_sensors.append(s_id)

    return t_prev, collected_sensors

# -----------------------------
# 7) Hàm vẽ
# -----------------------------
def plot_map(uav_positions, sensor_positions, sensor_types, t_prev, iteration, current_time,
             vsi_recovery_intervals, x_min, x_max, y_min, y_max,
             G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
             P_tx_uav, G_tx_uav, G_rx_uav, lambda_S_plot, N0, SINR_threshold, Q, N, cm, km):
    """
    Vẽ bản đồ hiển thị vị trí UAV, cảm biến và đường truyền

    Parameters:
        uav_positions: Vị trí các UAV
        sensor_positions: Vị trí các cảm biến
        sensor_types: Loại của các cảm biến
        t_prev: Thời gian thu thập gần nhất của mỗi cảm biến
        iteration: Số thứ tự vòng lặp
        current_time: Thời gian hiện tại
        vsi_recovery_intervals: Thời gian phục hồi VSI cho mỗi loại cảm biến
        x_min, x_max, y_min, y_max: Giới hạn vùng hiển thị
        G_St, G_Ur, lambda_S, P_st, P_U_min_r, h: Tham số tính bán kính cảm biến
        P_tx_uav, G_tx_uav, G_rx_uav, lambda_S_plot, N0, SINR_threshold, Q, N, cm, km: Tham số cho path_quality_utility
    """
    plt.clf()
    plt.title(f"Time Step = {iteration}, time={current_time}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    # Vẽ sensor với màu khác nhau tùy theo trạng thái VSI
    max_delta = max(vsi_recovery_intervals.values())
    for i, ((sx, sy), stype) in enumerate(zip(sensor_positions, sensor_types)):
        # Tính delta_t
        delta_t = current_time - t_prev[i]
        recovery_interval = vsi_recovery_intervals[stype]

        # Chuẩn hóa về [0, 1]
        normalized_vsi = min(delta_t / recovery_interval, 1.0)

        # Màu từ đỏ (mới thu) đến xanh (đã phục hồi VSI)
        color = (1.0 - normalized_vsi, 0.0, normalized_vsi)  # (R,G,B)

        if stype == "type1":
            plt.plot(sx, sy, 'o', color=color, markersize=3, alpha=0.8)
        elif stype == "type2":
            plt.plot(sx, sy, '^', color=color, markersize=4, alpha=0.8)
        else:
            plt.plot(sx, sy, 'x', color=color, markersize=4, alpha=0.8)

    # Vẽ UAV
    for i, (ux, uy, _) in enumerate(uav_positions):
        if i == 0:
            plt.plot(ux, uy, 's', color='red', markersize=10, label="GBS")
        else:
            plt.plot(ux, uy, 'D', color='blue', markersize=5, label="UAV" if i == 1 else "")

    # Vẽ vùng cảm biến
    sensing_r = compute_sensing_radius(G_St, G_Ur, lambda_S, P_st, P_U_min_r, h)
    for i, (ux, uy, _) in enumerate(uav_positions):
        if i > 0:
            circle = plt.Circle((ux, uy), sensing_r, color='blue', alpha=0.1)
            plt.gca().add_patch(circle)

    # Vẽ đường truyền tối ưu
    uav_dict = {idx: (pos[0], pos[1], pos[2]) for idx, pos in enumerate(uav_positions)}
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
                    plt.plot([sx, ex], [sy, ey], 'g--', linewidth=1)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.pause(1.0)

# -----------------------------
# 8) Main
# -----------------------------
def main():
    # 1) Tham số chung cho PSO
    num_particles = 50  # Số hạt trong PSO
    max_pso_iterations = 300  # Số vòng lặp PSO tối đa
    num_time_steps = 50  # Số time step mô phỏng
    convergence_threshold = 0.1  # Ngưỡng hội tụ (epsilon)
    omega = 0.5  # Hệ số quán tính
    c1 = 1  # cognitive
    c2 = 1 # social
    speed_factor =2.5 # Nhân thêm speed_factor khi cập nhật vận tốc

    # Biên giới vị trí (m)
    x_min, x_max = 0.0, 2000.0
    y_min, y_max = 0.0, 2000.0
    h_fixed = 100.0  # Độ cao UAV cố định

    # Giới hạn vận tốc
    v_min, v_max = -200.0, 200.0  # Tăng gấp đôi so với -100..100

    # Thời gian bắt đầu
    current_time = 0.0

    # VSI recovery interval cho 3 loại cảm biến (đơn vị là Tu)
    vsi_recovery_intervals = {
        "type1": 3,  # R1 = 3Tu
        "type2": 5,  # R2 = 5Tu
        "type3": 7  # R3 = 7Tu
    }

    # 2) Tham số cho total_utility và path_quality_utility
    # Tham số cho VSI
    R = [3, 5, 7]  # VSI recovery interval theo Tu
    Vmax = [10, 15, 20]  # VSI tối đa
    Vmin = [0.1, 0.15, 0.20]  # VSI tối thiểu

    # Tham số cho tính bán kính vùng cảm biến
    G_St = 1.0  # Hệ số khuếch đại ăng-ten cảm biến
    G_Ur = 1.0  # Hệ số khuếch đại ăng-ten UAV
    lambda_S = 3e8 / 2.4e9  # Bước sóng (2.4 GHz)
    P_st = 0.006  # Công suất phát của cảm biến
    P_U_min_r = 10 ** ((-76 - 30) / 10)  # Công suất thu tối thiểu
    h = h_fixed  # Độ cao UAV

    # Tham số cho kết nối UAV-UAV
    P_tx_uav = 0.25 # Công suất phát của UAV (250 mW)
    G_tx_uav = 1.0  # Hệ số khuếch đại ăng-ten phát UAV
    G_rx_uav = 1.0  # Hệ số khuếch đại ăng-ten thu UAV
    N0 = 1e-11  # Công suất nhiễu nền (thay sigma2)

    # Tham số cho path_quality_utility
    SINR_threshold = 4  # Ngưỡng SINR tối thiểu cho liên kết
    Q = 3  # Số lần truyền lại tối đa
    N = 20  # Số bit trong packet
    cm = 0.5  # Hằng số cm
    km = 0.5  # Hằng số km

    # Thông số trung tâm cụm cảm biến và độ lệch chuẩn
    cluster_centers = [
        (400, 1600),
        (1200, 1700),
        (500, 400),
        (1500, 600)
    ]
    std_dev = 150.0

    # Tham số vẽ
    figsize = (8, 6)

    # Khởi tạo
    random.seed(123)
    sensor_positions, sensor_types = generate_sensor_clusters(
        seed=42,
        cluster_centers=cluster_centers,
        std_dev=std_dev
    )
    t_prev = [0.0] * len(sensor_positions)

    # Khởi tạo vị trí UAV ban đầu
    init_uav_positions = [
        (100.0, 100.0, 0.0),  # UAV0 = GBS
        (500.0, 200.0, h_fixed),
        (600.0, 1000.0, h_fixed),
        (1500.0, 500.0, h_fixed),
        (1200.0, 1200.0, h_fixed)
    ]

    # Vẽ trạng thái ban đầu
    plt.figure(figsize=figsize)
    current_uav_positions = deepcopy(init_uav_positions)

    # Tính tiện ích hiện tại
    current_utility, _, _, _= compute_total_utility(
        current_uav_positions, sensor_positions, sensor_types,
        current_time, t_prev,
        R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
        P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
    )
    plot_map(
        current_uav_positions, sensor_positions, sensor_types, t_prev,
        iteration=0, current_time=current_time,
        vsi_recovery_intervals=vsi_recovery_intervals,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        G_St=G_St, G_Ur=G_Ur, lambda_S=lambda_S, P_st=P_st, P_U_min_r=P_U_min_r, h=h,
        P_tx_uav=P_tx_uav, G_tx_uav=G_tx_uav, G_rx_uav=G_rx_uav, lambda_S_plot=lambda_S, N0=N0,
        SINR_threshold=SINR_threshold, Q=Q, N=N, cm=cm, km=km
    )
    print(f"Tiện ích ban đầu: {current_utility:.4f}")




    # Bắt đầu vòng lặp thời gian (mỗi bước là 1 Tu)
    for step in range(1, num_time_steps + 1):
        print(f"\n==== Time Step {step} (Tu = {step}) ====")

        # Tăng thời gian
        current_time += 1

        # 1. Tính tiện ích hiện tại
        current_utility, _, _, per_uav_utilities = compute_total_utility(
            current_uav_positions, sensor_positions, sensor_types,
            current_time, t_prev,
            R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
            P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
        )

        # Thu thập cảm biến ở vị trí hiện tại
        t_prev, collected_sensors = update_sensors_after_collection(
            current_uav_positions, sensor_positions, t_prev, current_time,
            G_St, G_Ur, lambda_S, P_st, P_U_min_r, h
        )
        # 5. Vẽ trạng thái hiện tại
        plot_map(
            current_uav_positions, sensor_positions, sensor_types, t_prev,
            iteration=step, current_time=current_time,
            vsi_recovery_intervals=vsi_recovery_intervals,
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            G_St=G_St, G_Ur=G_Ur, lambda_S=lambda_S, P_st=P_st, P_U_min_r=P_U_min_r, h=h,
            P_tx_uav=P_tx_uav, G_tx_uav=G_tx_uav, G_rx_uav=G_rx_uav, lambda_S_plot=lambda_S, N0=N0,
            SINR_threshold=SINR_threshold, Q=Q, N=N, cm=cm, km=km
        )
        print(f"Số cảm biến thu thập: {len(collected_sensors)}")

        print("Chi tiết tiện ích từng UAV:")
        for u_info in per_uav_utilities:
            print(f"  UAV{u_info['uav_id']}: "
                  f"Sensing={u_info['sensing_util']:.4f}, "
                  f"Path={u_info['path_util']:.4f}, "
                  f"Conn={u_info['conn_util']:.4f}, "
                  f"Total={u_info['total_util']:.4f}")

        print(f"Tiện ích hiện tại: {current_utility:.4f}")

        # 2. Chạy PSO để tìm vị trí tối ưu mới
        print("Đang chạy PSO để tìm vị trí tối ưu...")
        best_fitness, best_positions, is_better = run_pso(
            current_uav_positions, sensor_positions, sensor_types,
            t_prev, current_time, current_utility,
            num_particles, max_pso_iterations, convergence_threshold,
            omega, c1, c2, speed_factor,
            x_min, x_max, y_min, y_max, v_min, v_max, h_fixed,
            R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
            P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
        )

        # 3. Nếu tìm được vị trí tốt hơn, di chuyển UAV
        if is_better and best_positions is not None:
            print("UAV đang di chuyển đến vị trí tối ưu mới...")
            current_uav_positions = deepcopy(best_positions)



        else:
            print("Giữ nguyên vị trí UAV hiện tại (không tìm được vị trí tốt hơn)")





    print("\nKết thúc mô phỏng.")
    plt.show()

if __name__ == "__main__":
    main()