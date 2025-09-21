# File: vsi.py
import math


def compute_vsi(t, t_prev, Vmax, Vmin, R):
    """
    Tính giá trị Sensing Information (VSI) cho một cảm biến theo công thức:

        v(t) = { a * exp(t - t_prev) + b, nếu (t - t_prev) <= R
               { Vmax,                    nếu (t - t_prev) > R

    với:
        a = (Vmax - Vmin) / (exp(R) - 1)
        b = Vmin - a

    Tham số:
    - t: Thời điểm hiện tại.
    - t_prev: Thời điểm thu thập dữ liệu của cảm biến lần trước.
    - Vmax: Giá trị tối đa của sensing information cho loại cảm biến.
    - Vmin: Giá trị tối thiểu sau khi dữ liệu được thu thập.
    - R: VSI recovery interval (khoảng thời gian hồi phục VSI).

    Trả về:
    - Giá trị VSI tại thời điểm t.
    """
    delta_t = t - t_prev
    if delta_t <= R:
        a = (Vmax - Vmin) / (math.exp(R) - 1)
        b = Vmin - a
        vsi = a * math.exp(delta_t) + b
    else:
        vsi = Vmax
    return vsi


# Ví dụ sử dụng trong file vsi.py
if __name__ == "__main__":
    # Các tham số ví dụ:
    Vmax = 100.0  # Giá trị tối đa của sensing information
    Vmin = 20.0  # Giá trị tối thiểu sau khi dữ liệu được thu thập
    R = 5.0  # VSI recovery interval (giây)
    t_prev = 0.0  # Thời điểm thu thập dữ liệu lần trước

    # Tính VSI tại các thời điểm khác nhau từ t = 0 đến t = 6 giây
    print("Tính VSI theo thời gian:")
    for t in [0, 1, 2, 3, 4, 5, 6,7,8,9,10,11]:
        vsi_value = compute_vsi(t, t_prev, Vmax, Vmin, R)
        print(f"t = {t} s, VSI = {vsi_value:.2f}")
