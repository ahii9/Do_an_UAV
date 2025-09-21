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

# -----------------------------
# 4) Lớp Particle (cho PSO)
# -----------------------------
class Particle:
    def __init__(self, num_uavs, init_positions, x_min, x_max, y_min, y_max, v_min, v_max, h_fixed):
        self.num_uavs = num_uavs
        self.position = []
        self.velocity = []
        if init_positions is not None:
            for i in range(num_uavs):
                if i == 0:
                    self.position.append(init_positions[i])
                    self.velocity.append((0.0, 0.0))
                else:
                    x0, y0, z0 = init_positions[i]
                    self.position.append((x0, y0, z0))
                    vx = random.uniform(v_min, v_max)
                    vy = random.uniform(v_min, v_max)
                    self.velocity.append((vx, vy))
        else:
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
    t_prev_copy = deepcopy(t_prev)
    total_util, _, _ ,_= compute_total_utility(
        particle.position, sensor_positions, sensor_types,
        current_time, t_prev_copy,
        R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
        P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
    )
    return total_util

def update_velocity_position(particle, global_best, omega, c1, c2, speed_factor, v_min, v_max, x_min, x_max, y_min, y_max):
    for i in range(particle.num_uavs):
        if i == 0:
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
        new_vx *= speed_factor
        new_vy *= speed_factor
        new_vx = max(v_min, min(new_vx, v_max))
        new_vy = max(v_min, min(new_vy, v_max))
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
    swarm = [Particle(num_uavs=5, init_positions=uav_init_positions,
                      x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                      v_min=v_min, v_max=v_max, h_fixed=h_fixed)
             for _ in range(num_particles)]
    global_best_fitness = -math.inf
    global_best_position = None
    prev_utility_iteration = -math.inf
    for iteration in range(max_iterations):
        current_utility_iteration = -math.inf
        for p in swarm:
            fitness = evaluate_fitness(p, sensor_positions, sensor_types, t_prev, current_time,
                                       R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                                       P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0)
            p.fitness = fitness
            if fitness > p.best_fitness:
                p.best_fitness = fitness
                p.best_position = deepcopy(p.position)
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = deepcopy(p.position)
            current_utility_iteration = max(current_utility_iteration, fitness)
        if global_best_position is not None:
            for p in swarm:
                update_velocity_position(p, global_best_position,
                                         omega, c1, c2, speed_factor,
                                         v_min, v_max, x_min, x_max, y_min, y_max)
        if abs(current_utility_iteration - prev_utility_iteration) < convergence_threshold:
            print(f"PSO hội tụ sau {iteration+1} vòng lặp")
            break
        prev_utility_iteration = current_utility_iteration
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
    sensing_r = compute_sensing_radius(G_St, G_Ur, lambda_S, P_st, P_U_min_r, h)
    uav_sensor_map, _ = sensors_in_uav_area(sensor_positions, uav_positions, sensing_r)
    collected_sensors = []
    for _, sensor_list in uav_sensor_map.items():
        for s_id in sensor_list:
            t_prev[s_id] = current_time
            collected_sensors.append(s_id)
    return t_prev, collected_sensors

# -----------------------------
# 8) Main
# -----------------------------
def main():
    # 1) Tham số chung cho PSO
    num_particles = 50
    max_pso_iterations = 300
    num_time_steps = 50
    convergence_threshold = 0.1
    omega = 0.5
    c1 = 1
    c2 = 1
    speed_factor = 2.5
    x_min, x_max = 0.0, 2000.0
    y_min, y_max = 0.0, 2000.0
    h_fixed = 100.0
    v_min, v_max = -200.0, 200.0
    current_time = 0.0
    R = [3, 5, 7]
    Vmax = [10, 15, 20]
    Vmin = [0.1, 0.15, 0.20]
    G_St = 1.0
    G_Ur = 1.0
    lambda_S = 3e8 / 2.4e9
    P_st = 0.006
    P_U_min_r = 10 ** ((-76 - 30) / 10)
    h = h_fixed
    P_tx_uav = 0.25
    G_tx_uav = 1.0
    G_rx_uav = 1.0
    N0 = 1e-11
    SINR_threshold = 4
    Q = 3
    N = 20
    cm = 0.5
    km = 0.5
    cluster_centers = [(400, 1600), (1200, 1700), (500, 400), (1500, 600)]
    std_dev = 150.0
    random.seed(123)
    sensor_positions, sensor_types = generate_sensor_clusters(seed=42, cluster_centers=cluster_centers, std_dev=std_dev)
    t_prev = [0.0] * len(sensor_positions)
    init_uav_positions = [(100.0, 100.0, 0.0), (500.0, 200.0, h_fixed), (600.0, 1000.0, h_fixed), (1500.0, 500.0, h_fixed), (1200.0, 1200.0, h_fixed)]
    current_uav_positions = deepcopy(init_uav_positions)
    qt_values = []  # Danh sách lưu trữ Qt cho mỗi Tu
    for step in range(1, num_time_steps + 1):
        print(f"\n==== Time Step {step} (Tu = {step}) ====")
        current_time += 1
        current_utility, _, _,_= compute_total_utility(
            current_uav_positions, sensor_positions, sensor_types,
            current_time, t_prev,
            R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
            P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
        )
        qt_values.append(current_utility)  # Lưu trữ Qt
        t_prev, collected_sensors = update_sensors_after_collection(
            current_uav_positions, sensor_positions, t_prev, current_time,
            G_St, G_Ur, lambda_S, P_st, P_U_min_r, h
        )
        print(f"Tiện ích hiện tại: {current_utility:.4f}")
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
        if is_better and best_positions is not None:
            print("UAV đang di chuyển đến vị trí tối ưu mới...")
            current_uav_positions = deepcopy(best_positions)
            print(f"Số cảm biến thu thập: {len(collected_sensors)}")
        else:
            print("Giữ nguyên vị trí UAV hiện tại (không tìm được vị trí tốt hơn)")
            print(f"Số cảm biến thu thập: {len(collected_sensors)}")
    print("\nKết thúc mô phỏng.")
    # Vẽ biểu đồ Qt
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_time_steps + 1), qt_values, marker='o', linestyle='-', color='b')

    plt.xlabel("Tu (Đơn vị thời gian)")
    plt.ylabel("Giá trị hàm mục tiêu tổng hợp")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()