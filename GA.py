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
# 2) Lớp Individual (cho GA)
# -----------------------------
class Individual:
    def __init__(self, num_uavs, init_positions, x_min, x_max, y_min, y_max, h_fixed):
        """
        Khởi tạo một cá thể trong GA, đại diện cho một tập hợp vị trí của các UAV.

        Parameters:
            num_uavs: Số lượng UAV
            init_positions: Vị trí ban đầu của các UAV (nếu None, khởi tạo ngẫu nhiên)
            x_min, x_max, y_min, y_max: Giới hạn vùng di chuyển
            h_fixed: Độ cao cố định của UAV
        """
        self.num_uavs = num_uavs
        self.position = []
        if init_positions is not None:
            for i in range(num_uavs):
                if i == 0:  # GBS cố định
                    self.position.append(init_positions[i])
                else:
                    x0, y0, z0 = init_positions[i]
                    # Thêm nhiễu nhẹ quanh vị trí ban đầu
                    x = x0 + random.uniform(-50, 50)
                    y = y0 + random.uniform(-50, 50)
                    x = max(x_min, min(x, x_max))
                    y = max(y_min, min(y, y_max))
                    self.position.append((x, y, z0))
        else:
            for i in range(num_uavs):
                if i == 0:  # GBS
                    self.position.append((0.0, 0.0, 0.0))
                else:
                    x = random.uniform(x_min, x_max)
                    y = random.uniform(y_min, y_max)
                    self.position.append((x, y, h_fixed))
        self.fitness = -math.inf


def evaluate_fitness(individual, sensor_positions, sensor_types, t_prev, current_time,
                     R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                     P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0):
    """
    Tính giá trị hàm mục tiêu (total utility) cho một cá thể.

    Parameters:
        individual: Đối tượng cá thể cần tính fitness
        sensor_positions, sensor_types: Vị trí và loại cảm biến
        t_prev: Thời gian thu thập gần nhất của mỗi cảm biến
        current_time: Thời gian hiện tại
        Các tham số còn lại: Các tham số tính toán utility

    Returns:
        float: Giá trị tiện ích
    """
    t_prev_copy = deepcopy(t_prev)
    total_util, _, _, _ = compute_total_utility(
        individual.position, sensor_positions, sensor_types,
        current_time, t_prev_copy,
        R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
        P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
    )
    return total_util


def select_parents(population, tournament_size):
    """
    Chọn hai cá thể bố mẹ bằng phương pháp tournament selection.

    Parameters:
        population: Danh sách các cá thể
        tournament_size: Kích thước giải đấu

    Returns:
        tuple: (parent1, parent2)
    """

    def select_one():
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    parent1 = select_one()
    parent2 = select_one()
    return parent1, parent2


def crossover(parent1, parent2, crossover_rate):
    """
    Lai ghép hai cá thể bố mẹ để tạo ra hai cá thể con.

    Parameters:
        parent1, parent2: Hai cá thể bố mẹ
        crossover_rate: Tỷ lệ lai ghép

    Returns:
        tuple: (child1, child2)
    """
    if random.random() > crossover_rate:
        return deepcopy(parent1), deepcopy(parent2)

    child1 = Individual(parent1.num_uavs, None, 0, 0, 0, 0, 0)
    child2 = Individual(parent2.num_uavs, None, 0, 0, 0, 0, 0)
    child1.position = []
    child2.position = []

    for i in range(parent1.num_uavs):
        if i == 0:  # GBS cố định
            child1.position.append(parent1.position[0])
            child2.position.append(parent2.position[0])
        else:
            if random.random() < 0.5:
                child1.position.append(parent1.position[i])
                child2.position.append(parent2.position[i])
            else:
                child1.position.append(parent2.position[i])
                child2.position.append(parent1.position[i])
    return child1, child2


def mutate(individual, mutation_rate, x_min, x_max, y_min, y_max, h_fixed):
    """
    Đột biến một cá thể bằng cách thay đổi ngẫu nhiên vị trí của một số UAV.

    Parameters:
        individual: Cá thể cần đột biến
        mutation_rate: Tỷ lệ đột biến
        x_min, x_max, y_min, y_max: Giới hạn vùng di chuyển
        h_fixed: Độ cao cố định
    """
    for i in range(1, individual.num_uavs):  # Bỏ qua GBS
        if random.random() < mutation_rate:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            individual.position[i] = (x, y, h_fixed)


# -----------------------------
# 3) Hàm chạy GA cho tối ưu hóa vị trí
# -----------------------------

def run_ga(uav_init_positions, sensor_positions, sensor_types, t_prev, current_time, current_utility,
           population_size, max_generations, tournament_size, crossover_rate, mutation_rate,
           x_min, x_max, y_min, y_max, h_fixed,
           R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
           P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0,
           convergence_threshold=0.1, patience=50):
    """
    Chạy GA để tìm vị trí tối ưu cho UAV, với điều kiện hội tụ để dừng sớm.

    Parameters:
        uav_init_positions: Vị trí ban đầu của các UAV
        sensor_positions, sensor_types: Vị trí và loại cảm biến
        t_prev: Thời gian thu thập gần nhất của mỗi cảm biến
        current_time: Thời gian hiện tại
        current_utility: Tiện ích hiện tại
        population_size: Kích thước quần thể
        max_generations: Số thế hệ tối đa
        tournament_size: Kích thước giải đấu khi chọn bố mẹ
        crossover_rate: Tỷ lệ lai ghép
        mutation_rate: Tỷ lệ đột biến
        x_min, x_max, y_min, y_max: Giới hạn vùng di chuyển
        h_fixed: Độ cao cố định của UAV
        convergence_threshold: Ngưỡng hội tụ (mặc định 0.1, giống PSO)
        patience: Số thế hệ kiên nhẫn trước khi dừng (mặc định 50)
        Các tham số còn lại: Các tham số tính toán utility

    Returns:
        tuple: (global_best_fitness, global_best_position, better_than_current)
    """
    # Khởi tạo quần thể với 50 cá thể, mỗi cá thể là một tập hợp vị trí UAV
    population = [Individual(num_uavs=5, init_positions=uav_init_positions,
                             x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, h_fixed=h_fixed)
                  for _ in range(population_size)]
    global_best_fitness = -math.inf  # $Q^T$ tốt nhất, khởi tạo -vô cực
    global_best_position = None  # Vị trí UAV tương ứng
    prev_best_fitness = -math.inf  # $Q^T$ tốt nhất của thế hệ trước
    no_improvement_count = 0  # Đếm số thế hệ không cải thiện $Q^T$

    # Vòng lặp GA, tối đa max_generations thế hệ
    for generation in range(max_generations):
        # Đánh giá $Q^T$ cho từng cá thể
        for ind in population:
            fitness = evaluate_fitness(ind, sensor_positions, sensor_types, t_prev, current_time,
                                       R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                                       P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0)
            ind.fitness = fitness
            # Cập nhật giải pháp tốt nhất nếu $Q^T$ cao hơn
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = deepcopy(ind.position)

        # Tạo thế hệ mới
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, tournament_size)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            mutate(child1, mutation_rate, x_min, x_max, y_min, y_max, h_fixed)
            mutate(child2, mutation_rate, x_min, x_max, y_min, y_max, h_fixed)
            new_population.extend([child1, child2])
        population = new_population[:population_size]

        # Kiểm tra điều kiện hội tụ
        # Nếu $Q^T$ không cải thiện quá convergence_threshold so với trước
        if abs(global_best_fitness - prev_best_fitness) < convergence_threshold:
            no_improvement_count += 1  # Tăng số thế hệ không cải thiện
        else:
            no_improvement_count = 0  # Reset nếu có cải thiện

        prev_best_fitness = global_best_fitness  # Cập nhật $Q^T$ trước

        # Nếu không cải thiện trong patience thế hệ, dừng GA
        if no_improvement_count >= patience:
            print(f"GA hội tụ sau {generation + 1} thế hệ, $Q^T$ tốt nhất = {global_best_fitness:.4f}")
            break

        # print(f"Thế hệ {generation + 1}: Tiện ích tốt nhất = {global_best_fitness:.4f}")

    # Kiểm tra xem $Q^T$ tìm được có tốt hơn tiện ích hiện tại không
    better_than_current = global_best_fitness > current_utility

    if better_than_current:
        print(f"Đã tìm được vị trí tốt hơn: {global_best_fitness:.4f} > {current_utility:.4f}")
    else:
        print(f"Không tìm được vị trí tốt hơn: {global_best_fitness:.4f} <= {current_utility:.4f}")

    return global_best_fitness, global_best_position, better_than_current

# -----------------------------
# 4) Hàm reset VSI cho cảm biến vừa được thu thập
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
# 5) Hàm vẽ
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
        delta_t = current_time - t_prev[i]
        recovery_interval = vsi_recovery_intervals[stype]
        normalized_vsi = min(delta_t / recovery_interval, 1.0)
        color = (1.0 - normalized_vsi, 0.0, normalized_vsi)
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
# 6) Main
# -----------------------------
def main():
    # 1) Tham số chung cho GA
    population_size = 50  # Kích thước quần thể
    max_generations = 300  # Số thế hệ tối đa
    num_time_steps = 50  # Số time step mô phỏng
    tournament_size = 5  # Kích thước giải đấu
    crossover_rate = 0.8  # Tỷ lệ lai ghép
    mutation_rate = 0.1  # Tỷ lệ đột biến

    # Biên giới vị trí (m)
    x_min, x_max = 0.0, 2000.0
    y_min, y_max = 0.0, 2000.0
    h_fixed = 100.0  # Độ cao UAV cố định

    # Thời gian bắt đầu
    current_time = 0.0

    # VSI recovery interval cho 3 loại cảm biến (đơn vị là Tu)
    vsi_recovery_intervals = {
        "type1": 3,  # R1 = 3Tu
        "type2": 5,  # R2 = 5Tu
        "type3": 7  # R3 = 7Tu
    }

    # 2) Tham số cho total_utility và path_quality_utility
    R = [3, 5, 7]  # VSI recovery interval theo Tu
    Vmax = [10, 15, 20]  # VSI tối đa
    Vmin = [0.1, 0.15, 0.20]  # VSI tối thiểu

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
        (100.0, 100.0, 0.0),  # GBS
        (500.0, 200.0, h_fixed),
        (600.0, 1000.0, h_fixed),
        (1500.0, 500.0, h_fixed),
        (1200.0, 1200.0, h_fixed)
    ]

    # Vẽ trạng thái ban đầu
    plt.figure(figsize=figsize)
    current_uav_positions = deepcopy(init_uav_positions)
    current_utility, _, _, _ = compute_total_utility(
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

    # Vòng lặp thời gian
    for step in range(1, num_time_steps + 1):
        print(f"\n==== Time Step {step} (Tu = {step}) ====")
        current_time += 1

        # Tính tiện ích hiện tại
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

        # Vẽ trạng thái hiện tại
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

        # Chạy GA để tìm vị trí tối ưu mới
        print("Đang chạy GA để tìm vị trí tối ưu...")
        best_fitness, best_positions, is_better = run_ga(
            current_uav_positions, sensor_positions, sensor_types,
            t_prev, current_time, current_utility,
            population_size, max_generations, tournament_size, crossover_rate, mutation_rate,
            x_min, x_max, y_min, y_max, h_fixed,
            R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
            P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
        )

        # Di chuyển UAV nếu tìm được vị trí tốt hơn
        if is_better and best_positions is not None:
            print("UAV đang di chuyển đến vị trí tối ưu mới...")
            current_uav_positions = deepcopy(best_positions)
        else:
            print("Giữ nguyên vị trí UAV hiện tại (không tìm được vị trí tốt hơn)")

    print("\nKết thúc mô phỏng.")
    plt.show()


if __name__ == "__main__":
    main()