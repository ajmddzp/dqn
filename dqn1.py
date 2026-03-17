import flappy_bird_gymnasium
import gymnasium

env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
obs, _ = env.reset()

print("观测值的形状:", obs.shape)
print("观测值的数据类型:", type(obs))
print("具体的观测值:", obs)
print("观测值包含的元素数量:", len(obs))

env.close()
