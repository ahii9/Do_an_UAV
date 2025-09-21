import numpy as np
import matplotlib.pyplot as plt
from total_utility import calculate_total_utility
from path_quality_utility import compute_SNR_values
from sensing_utility import calculate_d_max

# Cấu hình hệ thống
NUM_UAV = 4
NUM_SENSORS = 300
NUM_CLUSTERS = 4
NUM_PARTICLES = 30
MAX_ITER = 100
SNR_THRESHOLD = 3
GBS_POS = (0, 15, 100)


def create_sensor_clusters():
    clusters = []
    cluster_centers = [(500, 500, 0), (1500, 500, 0), (500, 1500, 0), (1500, 1500, 0)]
    for center in cluster_centers:
        cluster = []
        for _ in range(NUM_SENSORS // NUM_CLUSTERS):
            x = center[0] + np.random.normal(0, 100)
            y = center[1] + np.random.normal(0, 100)
            sensor_type = np.random.randint(1, 4)
            cluster.append((x, y, 0, sensor_type))
        clusters.append(cluster)
    return clusters, cluster_centers


class PSO:
    def __init__(self, sensors, cluster_centers):
        self.sensors = sensors
        self.cluster_centers = cluster_centers
        self.gbest_position = None
        self.gbest_value = -np.inf
        self.particles = []
        self.velocities = []
        self.pbest_positions = []
        self.pbest_values = []

        for _ in range(NUM_PARTICLES):
            particle = [(np.random.rand() * 2000, np.random.rand() * 2000, 100) for _ in range(NUM_UAV)]
            self.particles.append(particle)
            self.velocities.append([(0, 0, 0) for _ in range(NUM_UAV)])
            self.pbest_positions.append(particle.copy())
            self.pbest_values.append(-np.inf)

        self.w = 0.5
        self.c1 = 1.4
        self.c2 = 1.4

    def fitness(self, positions, time):
        UAV_positions = {0: GBS_POS}
        for i in range(NUM_UAV):
            UAV_positions[i + 1] = positions[i]

        d_max = calculate_d_max(G_S_t=10, G_U_r=5, f=2.4e9, P_S_t=0.1, P_U_r_min=1e-9)
        SNR_values = compute_SNR_values(UAV_positions)

        '''cluster_coverage = [0] * NUM_CLUSTERS
        for sensor in self.sensors:
            distances = [np.linalg.norm(np.array(sensor[:3]) - np.array(UAV_positions[i])) for i in range(1, NUM_UAV + 1)]
            min_distance = min(distances)
            cluster_idx = np.argmin([np.linalg.norm(np.array(sensor[:3]) - np.array(center)) for center in self.cluster_centers])
            if min_distance <= d_max:
                cluster_coverage[cluster_idx] += 1

        coverage_penalty = sum(1 for count in cluster_coverage if count == 0) * -1000  # Phạt nếu có cụm bị bỏ qua'''

        sensor_data = [{'t_p': time * np.random.uniform(0.5, 1.5)} for _ in self.sensors]

        return (calculate_total_utility(
            t=time,
            UAV_positions=UAV_positions,
            sensors=self.sensors,
            sensor_data=sensor_data,
            SNR_values=SNR_values,
            N=20,
            cm=0.5,
            km=0.5,
            Q=3,
            SNR_threshold=SNR_THRESHOLD,
            d_max=d_max,
            R_j=10.0,
            V_max_j=15,
            V_min_j=0.1,
            c=30,
            d=40,
            sensor_SNR={}
        )
                #+ coverage_penalty
         )

    def optimize(self):
        trajectories = {i: [] for i in range(NUM_UAV)}
        time_intervals = np.linspace(1, 100, MAX_ITER)

        for iter in range(MAX_ITER):
            current_time = time_intervals[iter]
            for i in range(NUM_PARTICLES):
                current_value = self.fitness(self.particles[i], current_time)
                if current_value > self.pbest_values[i]:
                    self.pbest_values[i] = current_value
                    self.pbest_positions[i] = self.particles[i].copy()
                if current_value > self.gbest_value:
                    self.gbest_value = current_value
                    self.gbest_position = self.particles[i].copy()

            for i in range(NUM_PARTICLES):
                for uav_idx in range(NUM_UAV):
                    r1 = np.random.rand(3)
                    r2 = np.random.rand(3)
                    cognitive = self.c1 * r1 * (np.array(self.pbest_positions[i][uav_idx]) - np.array(self.particles[i][uav_idx]))
                    social = self.c2 * r2 * (np.array(self.gbest_position[uav_idx]) - np.array(self.particles[i][uav_idx]))
                    new_velocity = self.w * np.array(self.velocities[i][uav_idx]) + cognitive + social
                    self.velocities[i][uav_idx] = tuple(new_velocity)
                    new_position = np.array(self.particles[i][uav_idx]) + new_velocity
                    self.particles[i][uav_idx] = tuple(new_position)

            for idx, pos in enumerate(self.gbest_position):
                trajectories[idx].append(pos)

            print(f"Iteration {iter + 1} | Time: {current_time:.2f}s | Utility: {self.gbest_value:.2f}")
            for uav_idx, pos in enumerate(self.gbest_position):
                print(f"  UAV {uav_idx + 1}: {pos}")

        self.plot_trajectories(trajectories)

    def plot_trajectories(self, trajectories):
        plt.figure(figsize=(12, 10))
        colors = ['blue', 'green', 'orange', 'purple']
        for sensor in self.sensors:
            plt.scatter(sensor[0], sensor[1], c='gray', alpha=0.5, s=10)
        plt.scatter(GBS_POS[0], GBS_POS[1], marker='s', s=200, c='red', label='GBS')

        for uav_idx, uav_trajectory in trajectories.items():
            uav_positions = np.array(uav_trajectory)
            plt.plot(uav_positions[:, 0], uav_positions[:, 1], c=colors[uav_idx], label=f'UAV {uav_idx + 1}')
            plt.scatter(uav_positions[-1, 0], uav_positions[-1, 1], c=colors[uav_idx], edgecolor='k', s=100)

        plt.title("UAV Trajectories Over Updates")
        plt.xlabel("X-coordinate (m)")
        plt.ylabel("Y-coordinate (m)")
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.show()

if __name__ == "__main__":
    clusters, cluster_centers = create_sensor_clusters()
    all_sensors = [sensor for cluster in clusters for sensor in cluster]
    pso = PSO(all_sensors, cluster_centers)
    pso.optimize()
