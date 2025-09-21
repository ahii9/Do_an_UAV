import matplotlib.pyplot as plt

# Data
time_steps = range(1, 51)
sensing_quality = [4.3758, 25.8361, 10.3178, 53.0879, 85.3989, 8.4160, 97.5514, 119.5776, 6.7669, 116.6611, 8.9955, 92.8667, 131.1221, 9.2057, 108.7593, 8.5601, 86.7045, 5.8037, 116.6677, 127.6456, 12.0936, 103.8865, 6.8040, 78.4220, 136.4078, 10.3697, 106.0530, 9.9050, 121.1807, 8.5319, 136.1045, 10.6794, 133.3912, 8.1079, 136.5149, 7.7555, 114.3806, 7.5069, 116.9566, 10.1321, 93.7138, 7.5052, 97.7761, 126.7802, 9.5614, 110.6785, 9.3045, 103.9860, 8.4323, 112.9535]
collected_sensors = [30, 60, 60, 50, 51, 51, 42, 43, 43, 56, 56, 54, 54, 54, 53, 53, 34, 34, 53, 67, 67, 44, 44, 43, 62, 62, 60, 60, 46, 46, 61, 61, 52, 52, 51, 51, 44, 44, 61, 61, 43, 43, 46, 59, 59, 54, 54, 49, 49, 50]

# Hình 4.9: Total VSI and Collected Sensors
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(time_steps, sensing_quality, 'b-', label='Giá trị chất lượng cảm biến')
ax1.set_xlabel('Tu (Đơn vị thời gian)')
ax1.set_ylabel('Giá trị chất lượng cảm biến', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(time_steps, collected_sensors, 'r-', label='Số cảm biến thu thập')
ax2.set_ylabel('Số cảm biến thu thập', color='r')
ax2.tick_params(axis='y', labelcolor='r')

fig.suptitle('')
fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.99), ncol=1)
plt.savefig('vsi_and_sensors_time.png', bbox_inches='tight')
plt.close()