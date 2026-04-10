import gymnasium as gym
from cart_agent import CartAgent
import numpy as np



def runEpisode(agentToUse, env):
    observation, info = env.reset()
    episode_over = False

    states = [observation]
    actions = []
    rewards = []
    while not(episode_over):
        action = agentToUse.chooseAction(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        states.append(observation)
        actions.append(action)
        rewards.append(reward)
        episode_over = (terminated or truncated)

    env.close()

    return states, actions, rewards


params = [np.random.uniform(-1,1) for _ in range(0,6)]
agent = CartAgent(params)
env = gym.make("MountainCar", render_mode="human")
states, actions, rewards = runEpisode(agent, env)