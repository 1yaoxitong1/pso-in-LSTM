import random
import numpy as np

# 目标函数
def objective_function(x):
    return x[0] ** 2 + x[1] ** 2 + 1

# 初始化粒子群
def initialize_particles(num_particles, dimensions):
    particles = []
    for _ in range(num_particles):
        particle = {
            'position': np.random.uniform(-100, 100, dimensions),
            'velocity': np.random.uniform(-1, 1, dimensions),  # 调整速度范围
            'best_position': None,
            'best_value': float('inf')
        }
        particles.append(particle)
    return particles

# 更新粒子速度和位置
def update_particles(particles, global_best_position, w=0.5, c1=1, c2=2):
    for particle in particles:
        r1, r2 = random.random(), random.random()
        velocity = (w * particle['velocity'] +
                    c1 * r1 * (particle['best_position'] - particle['position']) +
                    c2 * r2 * (global_best_position - particle['position']))
        position = particle['position'] + velocity
        # 添加边界检查
        position = np.clip(position, -100, 100)
        particle['velocity'] = velocity
        particle['position'] = position

# 主函数
def pso(num_particles, dimensions, num_iterations):
    particles = initialize_particles(num_particles, dimensions)
    global_best_position = particles[0]['position'].copy()
    global_best_value = float('inf')

    for iteration in range(num_iterations):
        for particle in particles:
            value = objective_function(particle['position'])
            if value < particle['best_value']:
                particle['best_value'] = value
                particle['best_position'] = particle['position'].copy()
            if value < global_best_value:
                global_best_value = value
                global_best_position = particle['position'].copy()

        update_particles(particles, global_best_position)
        print(f"Iteration {iteration + 1}/{num_iterations}, Best Value: {global_best_value}")

    return global_best_position, global_best_value

# 参数设置
num_particles = 60
dimensions = 2  # 修改为2维问题
num_iterations = 100

# 运行PSO算法
best_position, best_value = pso(num_particles, dimensions, num_iterations)
print("Best Position:", best_position)
print("Best Value:", best_value)
