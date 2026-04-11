import gymnasium as gym
from cart_agent import CartAgent
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import os

def rollingAverage(data, windowSize):
    kernel = np.ones(windowSize) / windowSize
    return np.convolve(data, kernel, mode='valid')

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

    return states, actions, rewards

def calculateReturns(rewards, gamma):
    returnsList = []
    for r in reversed(rewards):
        if len(returnsList) == 0:
            returnsList.append(r)
        else:
            returnsList.append(r + gamma*returnsList[-1])
    return list(reversed(returnsList))

weights_file_path = "agent_weights.pth"
agent = CartAgent()
if os.path.exists(weights_file_path):
    agent.load_state_dict(torch.load(weights_file_path))
    print("loaded agent from existing weights")
else:
    print("agent beginning with random weights")


env = gym.make("MountainCar-v0", render_mode=None, max_episode_steps=400)
gamma = .99

learning_rate = 0.0001
optimizer = torch.optim.Adam(agent.parameters(), lr = learning_rate)

numEpisodes = 200
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
        loss = -1*log_prob * totalReturn

        loss.backward()

        optimizer.step()

    episodes.append(episodeNumber)
    rewardsPerEpisode.append(sum(rewards))



torch.save(agent.state_dict(), weights_file_path.replace(".pth", "_2.pth"))

env = gym.make("MountainCar-v0", render_mode="human", max_episode_steps=400)
env.reset()
states, actions, rewards = runEpisode(agent, env)
    
window = 20
averaged = rollingAverage(rewardsPerEpisode, window)
plt.plot(episodes, rewardsPerEpisode, ".", alpha=0.3, label="raw")
plt.plot(episodes[window-1:], averaged, label="rolling average")
plt.legend()
plt.show()