import random
import math
import matplotlib.pyplot as plt
from copy import deepcopy
from path_quality_utility import compute_sensor_SINR_dict_using_sensing, compute_link_sinr
from total_utility import compute_total_utility
from path_quality_utility import generate_candidate_paths, select_optimal_path
from sensing_utility import compute_sensing_radius, sensors_in_uav_area

# -----------------------------
# 1) Sinh dữ liệu cảm biến
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
# 2) Lớp Particle (cho PSO)
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
    total_util, _, _ = compute_total_utility(
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
# 3) Hàm chạy PSO
# -----------------------------
def run_pso(uav_init_positions, sensor_positions, sensor_types, t_prev, current_time, current_utility,
            num_particles, max_iterations, convergence_threshold,
            omega, c1, c2, speed_factor,
            x_min, x_max, y_min, y_max, v_min, v_max, h_fixed,
            R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
            P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0):
    swarm = [Particle(num_uavs=len(uav_init_positions), init_positions=uav_init_positions,
                      x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                      v_min=v_min, v_max=v_max, h_fixed=h_fixed)
             for _ in range(num_particles)]
    global_best_fitness = -math.inf
    global_best_position = None
    best_t_prev = t_prev

    prev_utility_iteration = -math.inf
    iteration_count = 0

    for iteration in range(max_iterations):
        iteration_count += 1
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
            break

        prev_utility_iteration = current_utility_iteration

    better_than_current = global_best_fitness > current_utility
    return global_best_fitness, global_best_position, better_than_current, iteration_count

# -----------------------------
# 4) Hàm reset VSI
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
# 5) Main: Phân tích tác động của tham số
# -----------------------------
def main():
    # Tham số chung
    max_pso_iterations = 100
    num_time_steps = 10  # Giảm số bước để chạy nhanh hơn
    convergence_threshold = 0.001

    x_min, x_max = 0.0, 2000.0
    y_min, y_max = 0.0, 2000.0
    v_min, v_max = -200.0, 200.0

    current_time = 0.0
    vsi_recovery_intervals = {"type1": 3, "type2": 5, "type3": 7}
    R = [3, 5, 7]
    Vmax = [10, 15, 20]
    Vmin = [0.1, 0.15, 0.20]

    G_St = 1.0
    G_Ur = 1.0
    lambda_S = 3e8 / 2.4e9
    P_st = 0.006
    P_U_min_r = 10 ** ((-76 - 30) / 10)
    h = 100.0
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

    # Danh sách vị trí UAV ban đầu đầy đủ (5 UAV, bao gồm GBS)
    full_init_uav_positions = [
        (100.0, 100.0, 0.0),  # GBS
        (500.0, 200.0, h),
        (600.0, 1000.0, h),
        (1500.0, 500.0, h),
        (2000.0, 2000.0, h)
    ]

    # Tham số cần phân tích
    num_particles_values = [30, 50, 70]
    omega_values = [0.5, 0.8, 1.0]
    c1_values = [0.5, 1.0, 1.5]
    c2_values = [1.0, 2.0, 2.5]
    speed_factor_values = [1.0, 1.5, 2.0]
    num_uavs_values = [3, 4, 5]  # Tổng số UAV (bao gồm GBS)

    # Lưu kết quả
    results = {
        "num_particles": {"utility": [], "iterations": []},
        "omega": {"utility": [], "iterations": []},
        "c1": {"utility": [], "iterations": []},
        "c2": {"utility": [], "iterations": []},
        "speed_factor": {"utility": [], "iterations": []},
        "num_uavs": {"utility": [], "iterations": []}
    }

    # 5.1) Phân tích num_particles
    print("Phân tích tác động của num_particles...")
    for num_particles in num_particles_values:
        current_uav_positions = deepcopy(full_init_uav_positions[:4])  # Sử dụng 4 UAV (bao gồm GBS)
        t_prev = [0.0] * len(sensor_positions)
        current_time = 0.0
        total_utility = 0.0
        total_iterations = 0

        for step in range(1, num_time_steps + 1):
            current_time += 1
            current_utility, _, _ = compute_total_utility(
                current_uav_positions, sensor_positions, sensor_types,
                current_time, t_prev,
                R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
            )
            t_prev, _ = update_sensors_after_collection(
                current_uav_positions, sensor_positions, t_prev, current_time,
                G_St, G_Ur, lambda_S, P_st, P_U_min_r, h
            )
            best_fitness, best_positions, is_better, iteration_count = run_pso(
                current_uav_positions, sensor_positions, sensor_types, t_prev, current_time, current_utility,
                num_particles, max_pso_iterations, convergence_threshold,
                omega_values[0], c1_values[1], c2_values[1], speed_factor_values[1],
                x_min, x_max, y_min, y_max, v_min, v_max, h,
                R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
            )
            if is_better and best_positions is not None:
                current_uav_positions = deepcopy(best_positions)
            total_utility += best_fitness
            total_iterations += iteration_count

        avg_utility = total_utility / num_time_steps
        avg_iterations = total_iterations / num_time_steps
        results["num_particles"]["utility"].append(avg_utility)
        results["num_particles"]["iterations"].append(avg_iterations)

    # 5.2) Phân tích omega
    print("Phân tích tác động của omega...")
    for omega in omega_values:
        current_uav_positions = deepcopy(full_init_uav_positions[:4])
        t_prev = [0.0] * len(sensor_positions)
        current_time = 0.0
        total_utility = 0.0
        total_iterations = 0

        for step in range(1, num_time_steps + 1):
            current_time += 1
            current_utility, _, _ = compute_total_utility(
                current_uav_positions, sensor_positions, sensor_types,
                current_time, t_prev,
                R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
            )
            t_prev, _ = update_sensors_after_collection(
                current_uav_positions, sensor_positions, t_prev, current_time,
                G_St, G_Ur, lambda_S, P_st, P_U_min_r, h
            )
            best_fitness, best_positions, is_better, iteration_count = run_pso(
                current_uav_positions, sensor_positions, sensor_types, t_prev, current_time, current_utility,
                num_particles_values[1], max_pso_iterations, convergence_threshold,
                omega, c1_values[1], c2_values[1], speed_factor_values[1],
                x_min, x_max, y_min, y_max, v_min, v_max, h,
                R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
            )
            if is_better and best_positions is not None:
                current_uav_positions = deepcopy(best_positions)
            total_utility += best_fitness
            total_iterations += iteration_count

        avg_utility = total_utility / num_time_steps
        avg_iterations = total_iterations / num_time_steps
        results["omega"]["utility"].append(avg_utility)
        results["omega"]["iterations"].append(avg_iterations)

    # 5.3) Phân tích c1
    print("Phân tích tác động của c1...")
    for c1 in c1_values:
        current_uav_positions = deepcopy(full_init_uav_positions[:4])
        t_prev = [0.0] * len(sensor_positions)
        current_time = 0.0
        total_utility = 0.0
        total_iterations = 0

        for step in range(1, num_time_steps + 1):
            current_time += 1
            current_utility, _, _ = compute_total_utility(
                current_uav_positions, sensor_positions, sensor_types,
                current_time, t_prev,
                R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
            )
            t_prev, _ = update_sensors_after_collection(
                current_uav_positions, sensor_positions, t_prev, current_time,
                G_St, G_Ur, lambda_S, P_st, P_U_min_r, h
            )
            best_fitness, best_positions, is_better, iteration_count = run_pso(
                current_uav_positions, sensor_positions, sensor_types, t_prev, current_time, current_utility,
                num_particles_values[1], max_pso_iterations, convergence_threshold,
                omega_values[0], c1, c2_values[1], speed_factor_values[1],
                x_min, x_max, y_min, y_max, v_min, v_max, h,
                R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
            )
            if is_better and best_positions is not None:
                current_uav_positions = deepcopy(best_positions)
            total_utility += best_fitness
            total_iterations += iteration_count

        avg_utility = total_utility / num_time_steps
        avg_iterations = total_iterations / num_time_steps
        results["c1"]["utility"].append(avg_utility)
        results["c1"]["iterations"].append(avg_iterations)

    # 5.4) Phân tích c2
    print("Phân tích tác động của c2...")
    for c2 in c2_values:
        current_uav_positions = deepcopy(full_init_uav_positions[:4])
        t_prev = [0.0] * len(sensor_positions)
        current_time = 0.0
        total_utility = 0.0
        total_iterations = 0

        for step in range(1, num_time_steps + 1):
            current_time += 1
            current_utility, _, _ = compute_total_utility(
                current_uav_positions, sensor_positions, sensor_types,
                current_time, t_prev,
                R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
            )
            t_prev, _ = update_sensors_after_collection(
                current_uav_positions, sensor_positions, t_prev, current_time,
                G_St, G_Ur, lambda_S, P_st, P_U_min_r, h
            )
            best_fitness, best_positions, is_better, iteration_count = run_pso(
                current_uav_positions, sensor_positions, sensor_types, t_prev, current_time, current_utility,
                num_particles_values[1], max_pso_iterations, convergence_threshold,
                omega_values[0], c1_values[1], c2, speed_factor_values[1],
                x_min, x_max, y_min, y_max, v_min, v_max, h,
                R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
            )
            if is_better and best_positions is not None:
                current_uav_positions = deepcopy(best_positions)
            total_utility += best_fitness
            total_iterations += iteration_count

        avg_utility = total_utility / num_time_steps
        avg_iterations = total_iterations / num_time_steps
        results["c2"]["utility"].append(avg_utility)
        results["c2"]["iterations"].append(avg_iterations)

    # 5.5) Phân tích speed_factor
    print("Phân tích tác động của speed_factor...")
    for speed_factor in speed_factor_values:
        current_uav_positions = deepcopy(full_init_uav_positions[:4])
        t_prev = [0.0] * len(sensor_positions)
        current_time = 0.0
        total_utility = 0.0
        total_iterations = 0

        for step in range(1, num_time_steps + 1):
            current_time += 1
            current_utility, _, _ = compute_total_utility(
                current_uav_positions, sensor_positions, sensor_types,
                current_time, t_prev,
                R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
            )
            t_prev, _ = update_sensors_after_collection(
                current_uav_positions, sensor_positions, t_prev, current_time,
                G_St, G_Ur, lambda_S, P_st, P_U_min_r, h
            )
            best_fitness, best_positions, is_better, iteration_count = run_pso(
                current_uav_positions, sensor_positions, sensor_types, t_prev, current_time, current_utility,
                num_particles_values[1], max_pso_iterations, convergence_threshold,
                omega_values[0], c1_values[1], c2_values[1], speed_factor,
                x_min, x_max, y_min, y_max, v_min, v_max, h,
                R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
            )
            if is_better and best_positions is not None:
                current_uav_positions = deepcopy(best_positions)
            total_utility += best_fitness
            total_iterations += iteration_count

        avg_utility = total_utility / num_time_steps
        avg_iterations = total_iterations / num_time_steps
        results["speed_factor"]["utility"].append(avg_utility)
        results["speed_factor"]["iterations"].append(avg_iterations)

    # 5.6) Phân tích số lượng UAV (num_uavs)
    print("Phân tích tác động của số lượng UAV...")
    for num_uavs in num_uavs_values:
        current_uav_positions = deepcopy(full_init_uav_positions[:num_uavs])  # Lấy số UAV tương ứng (bao gồm GBS)
        t_prev = [0.0] * len(sensor_positions)
        current_time = 0.0
        total_utility = 0.0
        total_iterations = 0

        for step in range(1, num_time_steps + 1):
            current_time += 1
            current_utility, _, _ = compute_total_utility(
                current_uav_positions, sensor_positions, sensor_types,
                current_time, t_prev,
                R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
            )
            t_prev, _ = update_sensors_after_collection(
                current_uav_positions, sensor_positions, t_prev, current_time,
                G_St, G_Ur, lambda_S, P_st, P_U_min_r, h
            )
            best_fitness, best_positions, is_better, iteration_count = run_pso(
                current_uav_positions, sensor_positions, sensor_types, t_prev, current_time, current_utility,
                num_particles_values[1], max_pso_iterations, convergence_threshold,
                omega_values[0], c1_values[1], c2_values[1], speed_factor_values[1],
                x_min, x_max, y_min, y_max, v_min, v_max, h,
                R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
            )
            if is_better and best_positions is not None:
                current_uav_positions = deepcopy(best_positions)
            total_utility += best_fitness
            total_iterations += iteration_count

        avg_utility = total_utility / num_time_steps
        avg_iterations = total_iterations / num_time_steps
        results["num_uavs"]["utility"].append(avg_utility)
        results["num_uavs"]["iterations"].append(avg_iterations)

    # 5.7) Vẽ biểu đồ
    plt.figure(figsize=(15, 10))

    # Biểu đồ cho num_particles
    plt.subplot(2, 3, 1)
    plt.plot(num_particles_values, results["num_particles"]["utility"], 'o-', label="Tổng tiện ích trung bình")
    plt.plot(num_particles_values, results["num_particles"]["iterations"], 's-', label="Số vòng lặp hội tụ trung bình")
    plt.xlabel("Số lượng hạt (num_particles)")
    plt.ylabel("Giá trị")
    plt.title("Tác động của num_particles")
    plt.legend()
    plt.grid(True)

    # Biểu đồ cho omega
    plt.subplot(2, 3, 2)
    plt.plot(omega_values, results["omega"]["utility"], 'o-', label="Tổng tiện ích trung bình")
    plt.plot(omega_values, results["omega"]["iterations"], 's-', label="Số vòng lặp hội tụ trung bình")
    plt.xlabel("Hệ số quán tính (ω)")
    plt.ylabel("Giá trị")
    plt.title("Tác động của ω")
    plt.legend()
    plt.grid(True)

    # Biểu đồ cho c1
    plt.subplot(2, 3, 3)
    plt.plot(c1_values, results["c1"]["utility"], 'o-', label="Tổng tiện ích trung bình")
    plt.plot(c1_values, results["c1"]["iterations"], 's-', label="Số vòng lặp hội tụ trung bình")
    plt.xlabel("Hệ số nhận thức (c1)")
    plt.ylabel("Giá trị")
    plt.title("Tác động của c1")
    plt.legend()
    plt.grid(True)

    # Biểu đồ cho c2
    plt.subplot(2, 3, 4)
    plt.plot(c2_values, results["c2"]["utility"], 'o-', label="Tổng tiện ích trung bình")
    plt.plot(c2_values, results["c2"]["iterations"], 's-', label="Số vòng lặp hội tụ trung bình")
    plt.xlabel("Hệ số xã hội (c2)")
    plt.ylabel("Giá trị")
    plt.title("Tác động của c2")
    plt.legend()
    plt.grid(True)

    # Biểu đồ cho speed_factor
    plt.subplot(2, 3, 5)
    plt.plot(speed_factor_values, results["speed_factor"]["utility"], 'o-', label="Tổng tiện ích trung bình")
    plt.plot(speed_factor_values, results["speed_factor"]["iterations"], 's-', label="Số vòng lặp hội tụ trung bình")
    plt.xlabel("Hệ số tăng tốc (speed_factor)")
    plt.ylabel("Giá trị")
    plt.title("Tác động của speed_factor")
    plt.legend()
    plt.grid(True)

    # Biểu đồ cho num_uavs
    plt.subplot(2, 3, 6)
    plt.plot(num_uavs_values, results["num_uavs"]["utility"], 'o-', label="Tổng tiện ích trung bình")
    plt.plot(num_uavs_values, results["num_uavs"]["iterations"], 's-', label="Số vòng lặp hội tụ trung bình")
    plt.xlabel("Số lượng UAV (bao gồm GBS)")
    plt.ylabel("Giá trị")
    plt.title("Tác động của số lượng UAV")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("parameter_analysis_with_uavs.png")
    plt.show()

if __name__ == "__main__":
    main()