import gymnasium as gym
from stable_baselines3 import A2C
import maze_pomdp

env = gym.make("MazeEnv")

model = A2C("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)