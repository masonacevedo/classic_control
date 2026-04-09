import gymnasium as gym
import numpy as np
from agent import Agent
import time
import matplotlib.pyplot as plt

def evaluateAgent(agentToTest, numEpisodes=10, showRender=False):
    total_reward_across_episodes = 0
    for _ in range(0, numEpisodes):
        if showRender:
            env = gym.make("CartPole-v1", render_mode="human")
        else:
            env = gym.make("CartPole-v1", render_mode=None)
        observation, info = env.reset()

        episode_over = False
        episode_reward = 0
        while not episode_over:
            action = agentToTest.chooseAction(observation)
            observation, time_step_reward, terminated, truncated, info = env.step(action)
            episode_reward += time_step_reward
            episode_over = terminated or truncated

        # print(f"Episode finished! Episode reward: {episode_reward}")
        env.close()
        total_reward_across_episodes += episode_reward
    return total_reward_across_episodes/numEpisodes

def runEpisode(agentToUse, showRender=False):
    if showRender:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make("CartPole-v1", render_mode=None)

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

weights = [np.random.uniform(-1,1) for _ in range(0,4)]
agent = Agent(weights)
gamma = 0.99
learning_rate = 0.001
numEpisodes = 2000

episodeNumbers = []
rewardsPerEpisode = []

lastPrint = time.time()
for episodeCount in range(0, numEpisodes):
    states, actions, rewards = runEpisode(agent)
    for i in range(0, len(rewards)):

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
        p = agent.rightProbability(state)

        # if we moved left
        if action == 0:
            p = agent.rightProbability(state)
            gradient_vector = (-p) * state

        # if we moved right
        else:
            p = agent.rightProbability(state)
            gradient_vector = (1-p) * state

        nudge = learning_rate * returnFromHere * gradient_vector
        agent.updateWeights(nudge)

    episodeNumbers.append(episodeCount)
    rewardsPerEpisode.append(sum(rewards))



_,_, rewards = runEpisode(agent, showRender=True)
print("totalReward:", sum(rewards))

plt.plot(episodeNumbers, rewardsPerEpisode)
plt.show()

# for each timestep:
#     compute G_t
#     plug in state, compute gradient of action we took with respect to learned weights
#     compute nudge = learning rate * G_t * gradient
#     weights += nudge
