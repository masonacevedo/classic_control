import gymnasium as gym
import numpy as np
from agent import Agent
np.random.seed(42)

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

ourAgent = Agent(a = np.random.uniform(-1,1),
                 b = np.random.uniform(-1,1),
                 c = np.random.uniform(-1,1),
                 d = np.random.uniform(-1,1))

episode_over = False
total_reward = 0

while not episode_over:
    action = ourAgent.chooseAction(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()