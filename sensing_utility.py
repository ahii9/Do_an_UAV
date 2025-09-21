# File: sensing_utility.py
import math
import matplotlib.pyplot as plt

from vsi import compute_vsi  # Import hàm compute_vsi từ file vsi.py


# Hàm tính bán kính vùng cảm biến của UAV
def compute_sensing_radius(G_St, G_Ur, lambda_S, P_st, P_U_min_r, h):
    """
    Tính toán bán kính vùng cảm biến của UAV

    Parameters:
        G_St (float): Hệ số khuếch đại ăng-ten cảm biến
        G_Ur (float): Hệ số khuếch đại ăng-ten UAV
        lambda_S (float): Bước sóng
        P_st (float): Công suất phát của cảm biến
        P_U_min_r (float): Công suất thu tối thiểu của UAV
        h (float): Độ cao của UAV

    Returns:
        float: Bán kính vùng cảm biến
    """
    numerator = G_St * G_Ur * (lambda_S ** 2) * P_st
    denominator = (4 * math.pi) ** 2 * P_U_min_r
    d_max = math.sqrt(numerator / denominator)
    radius = math.sqrt(max(d_max ** 2 - h ** 2, 0))
    return radius


# Hàm kiểm tra sensor nằm trong vùng cảm biến UAV nào
def sensors_in_uav_area(sensor_positions, uav_positions, sensing_radius):
    """
    Xác định cảm biến nào nằm trong vùng cảm biến của UAV nào

    Parameters:
        sensor_positions (list): Danh sách vị trí các cảm biến
        uav_positions (list): Danh sách vị trí các UAV
        sensing_radius (float): Bán kính vùng cảm biến

    Returns:
        tuple: (uav_sensor_map, sensor_uav_map)
            - uav_sensor_map: Dict ánh xạ từ UAV đến các cảm biến nằm trong vùng
            - sensor_uav_map: Dict ánh xạ từ cảm biến đến các UAV có thể thu thập nó
    """
    uav_sensor_map = {uav_id: [] for uav_id in range(len(uav_positions))}
    sensor_uav_map = {sensor_id: [] for sensor_id in range(len(sensor_positions))}

    for sensor_id, s_pos in enumerate(sensor_positions):
        for uav_id, u_pos in enumerate(uav_positions):
            distance = math.hypot(s_pos[0] - u_pos[0], s_pos[1] - u_pos[1])
            if distance <= sensing_radius:
                uav_sensor_map[uav_id].append(sensor_id)
                sensor_uav_map[sensor_id].append(uav_id)

    return uav_sensor_map, sensor_uav_map


# Hàm tính solitary index dựa trên số UAV có thể thu thập sensor
def compute_solitary_index(N_ujk):
    """
    Tính toán chỉ số solitary dựa trên số lượng UAV có thể thu thập cảm biến

    Parameters:
        N_ujk (int): Số lượng UAV có thể thu thập cảm biến

    Returns:
        float: Giá trị chỉ số solitary
    """
    return 1.0 / N_ujk if N_ujk > 0 else 0.0


# Tính sensing utility cho sensor type cụ thể
def compute_sensor_type_utility(sensor_values, c, d):
    """
    Tính toán utility của một loại cảm biến cụ thể

    Parameters:
        sensor_values (list): Danh sách giá trị VSI hiệu dụng của các cảm biến
        c (float): Hằng số c trong công thức tính utility
        d (float): Hằng số d trong công thức tính utility

    Returns:
        float: Giá trị utility của loại cảm biến
    """
    total_effective_vsi = sum(sensor_values)
    utility = c * math.log10((total_effective_vsi / d) + 1)
    return utility


# Tính tổng sensing utility của UAV
def compute_total_sensing_utility(uav_sensor_data, c, d):
    """
    Tính toán tổng utility của tất cả các cảm biến được thu thập bởi UAV

    Parameters:
        uav_sensor_data (dict): Dict chứa dữ liệu cảm biến theo loại
            Format: {sensor_type: [(vsi_1, N_ujk_1), (vsi_2, N_ujk_2), ...]}
        c (float): Hằng số c trong công thức tính utility
        d (float): Hằng số d trong công thức tính utility

    Returns:
        float: Tổng utility của UAV
    """
    total_utility = 0.0
    for sensor_type, sensor_list in uav_sensor_data.items():
        effective_vsi_values = []
        for vsi, N_ujk in sensor_list:
            solitary_index = compute_solitary_index(N_ujk)
            effective_vsi = vsi * solitary_index
            effective_vsi_values.append(effective_vsi)
        sensor_type_utility = compute_sensor_type_utility(effective_vsi_values, c, d)
        total_utility += sensor_type_utility
    return total_utility


def plot_sensing_map(sensor_positions, uav_positions, sensing_radius, xlim, ylim, figsize):
    """
    Vẽ bản đồ vùng cảm biến của UAV

    Parameters:
        sensor_positions (list): Danh sách vị trí các cảm biến
        uav_positions (list): Danh sách vị trí các UAV
        sensing_radius (float): Bán kính vùng cảm biến
        xlim (tuple): Giới hạn trục x
        ylim (tuple): Giới hạn trục y
        figsize (tuple): Kích thước hình
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("UAV Sensing Coverage Map")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.grid(True, linestyle='--', alpha=0.6)

    # Vẽ Sensor (màu đen)
    for s_pos in sensor_positions:
        ax.scatter(s_pos[0], s_pos[1], color='black', marker='o',
                   label='Sensor' if 'Sensor' not in ax.get_legend_handles_labels()[1] else "")

    # Vẽ UAV và vòng tròn vùng cảm biến
    for u_pos in uav_positions:
        ax.scatter(u_pos[0], u_pos[1], color='blue', marker='s', s=100,
                   label='UAV' if 'UAV' not in ax.get_legend_handles_labels()[1] else "")
        circle = plt.Circle((u_pos[0], u_pos[1]), sensing_radius, color='blue', alpha=0.2,
                            label='Sensing Area' if 'Sensing Area' not in ax.get_legend_handles_labels()[1] else "")
        ax.add_patch(circle)

    ax.legend()
    plt.show()


if __name__ == "__main__":
    # Khai báo tất cả các tham số hệ thống
    G_St = 1.0
    G_Ur = 1.0
    lambda_S = 3e8 / 2.4e9  # 2.4 GHz
    P_st = 0.006  # 6 mW
    P_U_min_r = 10 ** ((-76 - 30) / 10)  # Convert -76 dBm to W
    h = 100  # độ cao UAV

    # Các hằng số tính utility
    c = 30
    d = 40

    # Các tham số cho đồ thị
    xlim = (0, 1000)
    ylim = (0, 1000)
    figsize = (8, 8)

    # Tính bán kính vùng cảm biến với các tham số đã khai báo
    sensing_radius = compute_sensing_radius(G_St, G_Ur, lambda_S, P_st, P_U_min_r, h)
    print("Sensing Radius:", sensing_radius)

    # Định nghĩa vị trí UAV và cảm biến
    uav_positions = [(200, 160, 100), (200, 0, 100), (400, 400, 100), (1000, 500, 100)]
    sensor_positions = [(50, 50), (51, 51), (150, 150), (250, 250), (300, 300),
                        (350, 350), (400, 450), (450, 400), (500, 500), (550, 550),
                        (600, 650), (650, 600), (700, 700), (750, 750), (800, 800)]

    # Xác định cảm biến nằm trong vùng cảm biến nào
    uav_sensor_map, sensor_uav_map = sensors_in_uav_area(sensor_positions, uav_positions, sensing_radius)

    # Hiển thị thông tin
    for uav_id, sensors in uav_sensor_map.items():
        print(f"UAV {uav_id} sensing area sensors:", sensors)
    for sensor_id, uavs in sensor_uav_map.items():
        print(f"Sensor {sensor_id} collected by UAVs:", uavs, "N_ujk=", len(uavs))

    # Vẽ bản đồ vùng cảm biến
    plot_sensing_map(sensor_positions, uav_positions, sensing_radius, xlim, ylim, figsize)