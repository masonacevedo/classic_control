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

def calculateReturns(rewards, gamma):
    returnsList = []
    for r in reversed(rewards):
        if len(returnsList) == 0:
            returnsList.append(r)
        else:
            returnsList.append(r + gamma*returnsList[-1])
    return list(reversed(returnsList))

env = gym.make("MountainCar", render_mode=None)
params = [np.random.uniform(-1,1) for _ in range(0,6)]
agent = CartAgent(params)
gamma = .99

numEpisodes = 10
for episodeNumber in range(0, numEpisodes):
    states, actions, rewards = runEpisode(agent, env)
    allReturns = calculateReturns(rewards, gamma)
    print("allReturns:", allReturns)
    print("len(allReturns):", len(allReturns))
    print("len(rewards):   ", len(rewards))
    # for i in range(0, len(rewards)):
    #     state, action, reward = states[i], actions[i], rewards[i]
