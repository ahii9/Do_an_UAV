import numpy as np
import matplotlib.pyplot as plt

# Tham số cho ba loại cảm biến
V_max = [20, 13, 10]  # Giá trị VSI tối đa
V_min = [0.2, 0.15, 0.1]  # Giá trị VSI tối thiểu
R = [5, 3, 2]  # Chu kỳ phục hồi
labels = ['Cảm biến nhiệt độ', 'Cảm biến độ ẩm', 'Cảm biến áp suất']
colors = ['r', 'g', 'b']

# Thời gian từ 0 đến 10
t = np.linspace(0, 10, 100)


# Hàm tính VSI theo hàm mũ
def calculate_vsi_exponential(t, t_p, V_max, V_min, R):
    a = (V_max - V_min) / (np.exp(R) - 1)
    b = V_min - a
    vsi = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] - t_p <= R:
            vsi[i] = a * np.exp(t[i] - t_p) + b
        else:
            vsi[i] = V_max
    return vsi


# Hàm tính VSI theo hàm tuyến tính
def calculate_vsi_linear(t, t_p, V_max, V_min, R):
    vsi = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] - t_p <= R:
            vsi[i] = V_max * (t[i] - t_p) / R
        else:
            vsi[i] = V_max
    return vsi


# Vẽ biểu đồ so sánh
plt.figure(figsize=(10, 6))

# Vẽ từng loại cảm biến
for j in range(3):
    vsi_exp = calculate_vsi_exponential(t, 0, V_max[j], V_min[j], R[j])
    vsi_linear = calculate_vsi_linear(t, 0, V_max[j], V_min[j], R[j])

    # Đường VSI hàm mũ
    plt.plot(t, vsi_exp, color=colors[j], linestyle='-', label=f'{labels[j]} (Hàm mũ)')
    # Đường VSI hàm tuyến tính
    plt.plot(t, vsi_linear, color=colors[j], linestyle='--', label=f'{labels[j]} (Hàm tuyến tính)')

# Cấu hình biểu đồ
plt.title('So sánh VSI: Hàm mũ và Hàm tuyến tính', fontsize=14)
plt.xlabel('Thời gian ', fontsize=12)
plt.ylabel('Giá trị Thông tin Cảm biến (VSI)', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Lưu biểu đồ
plt.savefig('vsi_comparison_exponential_linear.png')
plt.show()