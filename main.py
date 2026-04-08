import gymnasium as gym
import numpy as np
from agent import Agent
import time

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

currentParams = [np.random.uniform(-1,1) for _ in range(0,4)]
currentEvaluation = evaluateAgent(Agent(currentParams))
lastPrint = time.time()
n=400
for i in range(0, n):
    if (time.time() - lastPrint) > 10:
        lastPrint = time.time()
        print("i/n:",100*i/n)
        print()
    adjustments = [np.random.uniform(-0.1, 0.1) for _ in range(0,4)]
    newParams = [param + adjustment for param, adjustment in zip(currentParams, adjustments)]
    newEvaluation = evaluateAgent(Agent(newParams), numEpisodes=30)
    if newEvaluation > currentEvaluation:
        currentParams = newParams
        currentEvaluation = newEvaluation

bestAgent = Agent(currentParams)

bestReward = evaluateAgent(bestAgent, numEpisodes=1, showRender=True)
print("bestReward:", bestReward)