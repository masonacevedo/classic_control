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

def runEpisode(agentToUse, showRender=False):
    if showRender:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make("CartPole-v1", render_mode=None)

    observation, info = env.reset()
    episode_over = False
    all_rewards = []
    all_actions = []
    while not episode_over:
        action = agentToUse.chooseAction(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        all_actions.append(action)
        all_rewards.append(reward)
        episode_over = (terminated or truncated)
    env.close()
    return all_actions, all_rewards

# this iterative method of calculating rewards is slow. 
# better to calculate them all at once using the recursive formula.
# O(n) rather than O(n^2)
def calculateReturn(startIndex, rewards, gamma):
    totalReturn = 0
    count = 0
    for i in range(startIndex, len(rewards)):
        totalReturn += (gamma**count) * rewards[index]
    return totalReturn

weights = [np.random.uniform(-1,1) for _ in range(0,4)]
a = Agent(weights)
actions, rewards = runEpisode(a, True)
print("actions:", actions)
print("len(actions):", len(actions))
print("len(rewards):", len(rewards))
# gamma = 0.9
# learning_rate = 0.1


# for _ in range(0, 10):
#     rewardsReceived = runEpisode(a)
#     for i in range(0, len(rewardsReceived)):
#         reward = rewardsReceived[i]
#         returnFromHere = calculateReturn(i, rewardsReceived, gamma)
#         gradient_vector = 





# for each timestep:
#     compute G_t
#     plug in state, compute gradient of action we took with respect to learned weights
#     compute nudge = learning rate * G_t * gradient
#     weights += nudge
