
import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu tổng Sensing, Path, và Conn tại mỗi Time Step (bỏ qua UAV0)
time_steps = list(range(1, 51))
sensing_quality = [
    1.8941 + 0.0000 + 2.4817 + 0.0000,  # Time Step 1
    7.7506 + 10.1390 + 6.0482 + 1.8983,  # Time Step 2
    3.3441 + 3.8988 + 2.2086 + 0.8663,  # Time Step 3
    11.7455 + 13.3160 + 6.2417 + 21.7847,  # Time Step 4
    12.7336 + 20.9730 + 29.1900 + 22.5023,  # Time Step 5
    0.7516 + 2.1646 + 2.8802 + 2.6196,  # Time Step 6
    38.0315 + 9.4547 + 30.2002 + 19.8650,  # Time Step 7
    28.3115 + 41.9657 + 21.3700 + 27.9304,  # Time Step 8
    1.5725 + 2.6095 + 0.9589 + 1.6260,  # Time Step 9
    24.4940 + 46.3435 + 5.2827 + 40.5409,  # Time Step 10
    1.1594 + 4.2284 + 0.0750 + 3.5327,  # Time Step 11
    41.0248 + 33.0530 + 0.0000 + 18.7889,  # Time Step 12
    29.1870 + 26.8041 + 50.4178 + 24.7132,  # Time Step 13
    1.4437 + 1.9244 + 3.8830 + 1.9546,  # Time Step 14
    35.1728 + 22.1080 + 27.0322 + 24.4463,  # Time Step 15
    2.9666 + 0.9285 + 1.7673 + 2.8977,  # Time Step 16
    26.4567 + 16.3220 + 18.9835 + 24.9423,  # Time Step 17
    1.4983 + 1.0840 + 1.7073 + 1.5141,  # Time Step 18
    40.6973 + 35.1376 + 0.0000 + 40.8328,  # Time Step 19
    22.7021 + 16.6461 + 51.4801 + 36.8173,  # Time Step 20
    2.0719 + 3.3352 + 4.4526 + 2.2339,  # Time Step 21
    31.6828 + 22.9094 + 29.3293 + 19.9650,  # Time Step 22
    1.8486 + 1.4701 + 2.1183 + 1.3670,  # Time Step 23
    17.2507 + 25.6328 + 22.5856 + 12.9528,  # Time Step 24
    37.2861 + 28.1328 + 23.0586 + 47.9303,  # Time Step 25
    3.0985 + 2.8181 + 1.3700 + 3.0831,  # Time Step 26
    25.6164 + 35.0553 + 17.0061 + 28.3752,  # Time Step 27
    2.6095 + 3.1158 + 1.5601 + 2.6196,  # Time Step 28
    25.1196 + 28.2248 + 35.5119 + 32.3244,  # Time Step 29
    1.4983 + 2.4867 + 2.4891 + 2.0578,  # Time Step 30
    38.4282 + 23.7455 + 35.5696 + 38.3612,  # Time Step 31
    2.7659 + 1.7764 + 3.4441 + 2.6930,  # Time Step 32
    31.8609 + 28.4207 + 36.9600 + 36.1496,  # Time Step 33
    2.3882 + 1.9546 + 2.5181 + 2.2472,  # Time Step 34
    36.9613 + 45.3339 + 34.9529 + 19.2669,  # Time Step 35
    1.8486 + 3.6108 + 1.5445 + 0.7516,  # Time Step 36
    23.6732 + 24.6663 + 32.6803 + 33.3608,  # Time Step 37
    1.3960 + 0.8547 + 2.4891 + 2.7672,  # Time Step 38
    0.0000 + 33.2953 + 47.2229 + 36.4384,  # Time Step 39
    0.0000 + 2.6874 + 4.2014 + 3.2433,  # Time Step 40
    10.0393 + 33.1653 + 33.9334 + 16.5758,  # Time Step 41
    0.2545 + 2.8181 + 2.5095 + 1.9231,  # Time Step 42
    23.0326 + 31.4222 + 21.3700 + 21.9513,  # Time Step 43
    44.3136 + 19.2806 + 24.6805 + 38.5055,  # Time Step 44
    3.2147 + 0.7516 + 2.7962 + 2.7989,  # Time Step 45
    33.6386 + 15.5984 + 34.0558 + 27.3857,  # Time Step 46
    1.9873 + 1.5153 + 4.4797 + 1.3422,  # Time Step 47
    28.7112 + 14.1232 + 22.0682 + 39.0834,  # Time Step 48
    1.8410 + 1.3700 + 1.8506 + 3.3707,  # Time Step 49
    35.5629 + 9.5682 + 41.8144 + 26.0080  # Time Step 50
]

communication_path_quality = [
    0.9266 + 0.0000 + 0.8348 + 0.0000,  # Time Step 1
    0.9314 + 0.7817 + 0.8855 + 0.7971,  # Time Step 2
    0.9314 + 0.7817 + 0.8855 + 0.7971,  # Time Step 3
    0.9428 + 0.7269 + 0.8597 + 0.7860,  # Time Step 4
    0.8871 + 0.8678 + 0.8552 + 0.7969,  # Time Step 5
    0.8871 + 0.8678 + 0.8552 + 0.7969,  # Time Step 6
    0.9280 + 0.8470 + 0.8600 + 0.7972,  # Time Step 7
    0.9129 + 0.7590 + 0.8434 + 0.7578,  # Time Step 8
    0.9129 + 0.7590 + 0.8434 + 0.7578,  # Time Step 9
    0.9528 + 0.8576 + 0.8868 + 0.8226,  # Time Step 10
    0.9528 + 0.8576 + 0.8868 + 0.8226,  # Time Step 11
    0.9286 + 0.8620 + 0.0000 + 0.8556,  # Time Step 12
    0.9399 + 0.8415 + 0.8407 + 0.7695,  # Time Step 13
    0.9399 + 0.8415 + 0.8407 + 0.7695,  # Time Step 14
    0.9268 + 0.8797 + 0.8636 + 0.7864,  # Time Step 15
    0.9268 + 0.8797 + 0.8636 + 0.7864,  # Time Step 16
    0.9190 + 0.8689 + 0.8657 + 0.8064,  # Time Step 17
    0.9190 + 0.8689 + 0.8657 + 0.8064,  # Time Step 18
    0.9349 + 0.8659 + 0.0000 + 0.8291,  # Time Step 19
    0.9213 + 0.8764 + 0.8377 + 0.8467,  # Time Step 20
    0.9213 + 0.8764 + 0.8377 + 0.8467,  # Time Step 21
    0.9335 + 0.8484 + 0.8522 + 0.8630,  # Time Step 22
    0.9335 + 0.8484 + 0.8522 + 0.8630,  # Time Step 23
    0.9037 + 0.8358 + 0.8515 + 0.8439,  # Time Step 24
    0.9260 + 0.8442 + 0.7557 + 0.7868,  # Time Step 25
    0.9260 + 0.8442 + 0.7557 + 0.7868,  # Time Step 26
    0.9253 + 0.8405 + 0.8466 + 0.8399,  # Time Step 27
    0.9253 + 0.8405 + 0.8466 + 0.8399,  # Time Step 28
    0.9204 + 0.8586 + 0.8569 + 0.8009,  # Time Step 29
    0.9204 + 0.8586 + 0.8569 + 0.8009,  # Time Step 30
    0.9369 + 0.8579 + 0.7208 + 0.7865,  # Time Step 31
    0.9369 + 0.8579 + 0.7208 + 0.7865,  # Time Step 32
    0.9248 + 0.8510 + 0.7816 + 0.8011,  # Time Step 33
    0.9248 + 0.8510 + 0.7816 + 0.8011,  # Time Step 34
    0.9344 + 0.8892 + 0.8433 + 0.9033,  # Time Step 35
    0.9344 + 0.8892 + 0.8433 + 0.9033,  # Time Step 36
    0.9207 + 0.8877 + 0.8786 + 0.9268,  # Time Step 37
    0.9207 + 0.8877 + 0.8786 + 0.9268,  # Time Step 38
    0.0000 + 0.8399 + 0.8710 + 0.9329,  # Time Step 39
    0.0000 + 0.8399 + 0.8710 + 0.9329,  # Time Step 40
    0.8445 + 0.8575 + 0.8396 + 0.9134,  # Time Step 41
    0.8445 + 0.8575 + 0.8396 + 0.9134,  # Time Step 42
    0.8817 + 0.8337 + 0.8681 + 0.9193,  # Time Step 43
    0.7917 + 0.9118 + 0.8377 + 0.9354,  # Time Step 44
    0.7917 + 0.9118 + 0.8377 + 0.9354,  # Time Step 45
    0.8121 + 0.9273 + 0.8566 + 0.9292,  # Time Step 46
    0.8121 + 0.9273 + 0.8566 + 0.9292,  # Time Step 47
    0.8313 + 0.9252 + 0.8669 + 0.9288,  # Time Step 48
    0.8313 + 0.9252 + 0.8669 + 0.9288,  # Time Step 49
    0.8349 + 0.9117 + 0.8619 + 0.9313  # Time Step 50
]

network_connectivity = [
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 1
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 2
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 3
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 4
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 5
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 6
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 7
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 8
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 9
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 10
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 11
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 12
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 13
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 14
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 15
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 16
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 17
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 18
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 19
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 20
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 21
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 22
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 23
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 24
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 25
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 26
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 27
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 28
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 29
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 30
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 31
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 32
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 33
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 34
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 35
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 36
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 37
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 38
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 39
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 40
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 41
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 42
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 43
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 44
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 45
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 46
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 47
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 48
    1.0000 + 1.0000 + 1.0000 + 1.0000,  # Time Step 49
    1.0000 + 1.0000 + 1.0000 + 1.0000  # Time Step 50
]
collected_sensors = [
    30,  # Time Step 1
    60,  # Time Step 2
    60,  # Time Step 3
    50,  # Time Step 4
    51,  # Time Step 5
    51,  # Time Step 6
    42,  # Time Step 7
    43,  # Time Step 8
    43,  # Time Step 9
    56,  # Time Step 10
    56,  # Time Step 11
    54,  # Time Step 12
    54,  # Time Step 13
    54,  # Time Step 14
    53,  # Time Step 15
    53,  # Time Step 16
    34,  # Time Step 17
    34,  # Time Step 18
    53,  # Time Step 19
    67,  # Time Step 20
    67,  # Time Step 21
    44,  # Time Step 22
    44,  # Time Step 23
    43,  # Time Step 24
    62,  # Time Step 25
    62,  # Time Step 26
    60,  # Time Step 27
    60,  # Time Step 28
    46,  # Time Step 29
    46,  # Time Step 30
    61,  # Time Step 31
    61,  # Time Step 32
    52,  # Time Step 33
    52,  # Time Step 34
    51,  # Time Step 35
    51,  # Time Step 36
    44,  # Time Step 37
    44,  # Time Step 38
    61,  # Time Step 39
    61,  # Time Step 40
    43,  # Time Step 41
    43,  # Time Step 42
    46,  # Time Step 43
    59,  # Time Step 44
    59,  # Time Step 45
    54,  # Time Step 46
    54,  # Time Step 47
    49,  # Time Step 48
    49,  # Time Step 49
    50   # Time Step 50
]

# Vẽ đồ thị riêng cho Sensing Quality
plt.figure(figsize=(8, 5))
plt.plot(time_steps, sensing_quality, label='', color='black', marker='o', markersize=3, linestyle='-')
plt.xlabel('Tu (Đơn vị thời gian)')
plt.ylabel('Qs (Giá trị chất lượng cảm biến)')
plt.title('Sensing Quality over Time (Excluding GBS)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('sensing_quality_plot.png', dpi=300)
plt.close()

# Vẽ đồ thị riêng cho Communication Path Quality
plt.figure(figsize=(8, 5))
plt.plot(time_steps, communication_path_quality, label='', color='black', marker='o', markersize=3, linestyle='-')
plt.xlabel('Tu (Đơn vị thời gian)')
plt.ylabel('Qp (Giá trị chất lượng đường truyền thông tin)')
plt.title('Communication Path Quality over Time (Excluding GBS)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('communication_path_quality_plot.png', dpi=300)
plt.close()

# Vẽ đồ thị riêng cho Network Connectivity
plt.figure(figsize=(8, 5))
plt.plot(time_steps, network_connectivity, label='', color='black', marker='o', markersize=3, linestyle='-')
plt.xlabel('Tu (Đơn vị thời gian)')
plt.ylabel('Qc (khả năng kết nối mạng)')
plt.title('Network Connectivity over Time (Excluding GBS)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('network_connectivity_plot.png', dpi=300)
plt.close()
# Vẽ đồ thị riêng cho số cảm biến thu thập
plt.figure(figsize=(8, 5))
plt.plot(time_steps, collected_sensors, label='', color='black', marker='o', markersize=3, linestyle='-')
plt.xlabel('Tu (Đơn vị thời gian)')
plt.ylabel('Số cảm biến được thu thập dữ liệu ')
plt.title('Number of Collected Sensors over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('collected_sensors_plot.png', dpi=300)
plt.close()
print("Đã lưu các đồ thị vào các file:")
print("- sensing_quality_plot.png")
print("- communication_path_quality_plot.png")
print("- network_connectivity_plot.png")
