import gymnasium as gym
from cart_agent import CartAgent
import numpy as np

env = gym.make("MountainCar", render_mode="human")

params = [np.random.uniform(-1,1) for _ in range(0,6)]
agent = CartAgent(params)

observation, info = env.reset()
episode_over = False

while not(episode_over):
    action = agent.chooseAction(observation, verbose=True)
    observation, time_step_reward, terminated, truncated, info = env.step(action)
    episode_over = (terminated or truncated)
env.close()