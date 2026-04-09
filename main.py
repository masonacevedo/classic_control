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
    while not episode_over:
        action = agentToUse.chooseAction(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        all_rewards.append(reward)
        episode_over = (terminated or truncated)
    env.close()
    return all_rewards

weights = [np.random.uniform(-1,1) for _ in range(0,4)]
rewards = runEpisode(Agent(weights), showRender=False)
print("rewards:", rewards)
print("sum(rewards):", sum(rewards))
# for each timestep:
#     compute G_t
#     plug in state, compute gradient of action we took with respect to learned weights
#     compute nudge = learning rate * G_t * gradient
#     weights += nudge
