import gymnasium as gym
import numpy as np
from agent import Agent
import time

def evaluateAgent(a,b,c,d, numEpisodes=10, showRender=False):
    total_reward_across_episodes = 0
    for _ in range(0, numEpisodes):
        if showRender:
            env = gym.make("CartPole-v1", render_mode="human")
        else:
            env = gym.make("CartPole-v1", render_mode=None)
        observation, info = env.reset()

        episode_over = False
        episode_reward = 0
        agentToTest = Agent(a,b,c,d)
        while not episode_over:
            action = agentToTest.chooseAction(observation)
            observation, time_step_reward, terminated, truncated, info = env.step(action)
            episode_reward += time_step_reward
            episode_over = terminated or truncated

        # print(f"Episode finished! Episode reward: {episode_reward}")
        env.close()
        total_reward_across_episodes += episode_reward
    return total_reward_across_episodes/numEpisodes


a_vals = np.linspace(-1,1, 3)
b_vals = np.linspace(-1,1, 3)
c_vals = np.linspace(-1,1, 3)
d_vals = np.linspace(-1,1, 3)
n = len(a_vals) * len(b_vals) * len(c_vals) * len(d_vals)

# evaluateAgent(0,0,0,0,1)
lastPrinted = time.time()
i = 0

results = {}

for a in a_vals:
    for b in b_vals:
        for c in c_vals:
            for d in d_vals:
                if time.time() - lastPrinted > 30:
                    print("i:", i)
                    print("n:", n)
                    print("i/n:", 100*i/n)
                    lastPrinted = time.time()
                result = evaluateAgent(a,b,c,d)
                k = (a,b,c,d)
                results[k] = result
                i += 1


sortedResults = sorted(results.items(), key= lambda i: i[1], reverse=True)
bestResults = sortedResults[0:20]

for params, reward in bestResults:
    print(params, "|", reward)

bestResult = sortedResults[0][0]

evaluateAgent(bestResult[0], bestResult[1], bestResult[2], bestResult[3], numEpisodes=1, showRender=True)