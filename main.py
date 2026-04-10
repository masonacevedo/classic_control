import gymnasium as gym
import numpy as np
from agent import Agent
import time
import matplotlib.pyplot as plt
import torch

def runEpisode(agentToUse, env):

    observation, info = env.reset()
    episode_over = False
    all_rewards = []
    all_actions = []
    all_states = [observation]
    while not episode_over:
        action = agentToUse.chooseAction(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        all_actions.append(action)
        all_rewards.append(reward)
        all_states.append(observation)
        episode_over = (terminated or truncated)
    env.close()
    return all_states, all_actions, all_rewards

# this iterative method of calculating rewards is slow. 
# better to calculate them all at once using the recursive formula.
# O(n) rather than O(n^2)
def calculateReturn(startIndex, rewards, gamma):
    totalReturn = 0
    count = 0
    for i in range(startIndex, len(rewards)):
        totalReturn += (gamma**count) * rewards[i]
        count += 1
    return totalReturn


agent = Agent()
gamma = 0.99

optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

numEpisodes = 2000

episodeNumbers = []
rewardsPerEpisode = []

lastPrint = time.time()
env = gym.make("CartPole-v1", render_mode=None)

for episodeCount in range(0, numEpisodes):

    states, actions, rewards = runEpisode(agent, env)
    for i in range(0, len(rewards)):

        optimizer.zero_grad()

        if time.time() - lastPrint > 30:
            print("epoch:     ", episodeCount)
            print("numEpochs: ", numEpisodes)
            print("percentage:", 100*episodeCount/numEpisodes)
            print()
            lastPrint = time.time()
        reward = rewards[i]
        action = actions[i]
        state = states[i]
        returnFromHere = calculateReturn(i, rewards, gamma)
        probability = agent.forward(state)[action]
        log_prob = torch.log(probability)

        loss = -1*returnFromHere * log_prob
        loss.backward()
        optimizer.step()

    episodeNumbers.append(episodeCount)
    rewardsPerEpisode.append(sum(rewards))


env = gym.make("CartPole-v1", render_mode="human")
_,_, rewards = runEpisode(agent, env)
print("totalReward:", sum(rewards))

plt.plot(episodeNumbers, rewardsPerEpisode)
plt.show()

# for each timestep:
#     compute G_t
#     plug in state, compute gradient of action we took with respect to learned weights
#     compute nudge = learning rate * G_t * gradient
#     weights += nudge
