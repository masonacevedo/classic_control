import gymnasium as gym
from cart_agent import CartAgent
import numpy as np
import time
import torch
import matplotlib.pyplot as plt

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


env = gym.make("MountainCar-v0", render_mode=None)
agent = CartAgent()
gamma = .99

learning_rate = 0.001
optimizer = torch.optim.Adam(agent.parameters(), lr = learning_rate)

numEpisodes = 1000
lastPrint = time.time()

episodes = []
rewardsPerEpisode = []
for episodeNumber in range(0, numEpisodes):
    if time.time() - lastPrint > 30:
        print("episode:     ", episodeNumber)
        print("numEpisodes: ", numEpisodes)
        print("percentage:  ", 100*episodeNumber/numEpisodes)
        print()
        lastPrint = time.time()

    states, actions, rewards = runEpisode(agent, env)
    allReturns = calculateReturns(rewards, gamma)
    for i in range(0, len(rewards)):
        state, action, totalReturn = states[i], actions[i], allReturns[i]

        optimizer.zero_grad()

        probabilities = agent.forward(state)
        prob_of_action = probabilities[action]
        log_prob = torch.log(prob_of_action)
        loss = log_prob * totalReturn

        loss.backward()

        optimizer.step()

    episodes.append(episodeNumber)
    rewardsPerEpisode.append(sum(rewards))


# env = gym.make("MountainCar-v0", render_mode=None")
# env.reset()
# runEpisode(agent, env)

plt.plot(episodes, rewardsPerEpisode, ".")
plt.show()