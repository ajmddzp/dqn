#import gym
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
from agent import Agent

env = gym.make("CartPole-v1")
#s = env.reset()
s, info = env.reset()

EPSILON_DECAY = 10000
EPSILON_START = 1.0
EPSILON_END = 0.02
TARGET_UPDATE_FREQUENCY = 10


n_episode = 5000
n_time_step = 1000

n_state = len(s)
n_action = env.action_space.n

agent = Agent(n_input=n_state, n_output=n_action)

REWARD_BUFFER = np.empty(shape=n_episode)
for episode_i in range(n_episode):
    episode_reward = 0
    for step_i in range(n_time_step):
        epsilon = np.interp(episode_i * n_time_step + step_i, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        random_sample = random.random()

        if random_sample <= epsilon:
            a = env.action_space.sample()
        else:
            # a = agent.online_net.action(s)#todo
            a = agent.online_net.act(s)  # todo

        #s_, r, done, info = env.step(a)
        s_, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        agent.memo.add_memo(s, a, r, done, s_)#todo
        s = s_
        episode_reward += r

        if done:
            s, info = env.reset()
            #s_ = env.reset()[0]
            REWARD_BUFFER[episode_i] = episode_reward
            break

        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()

        #compute targets
        target_q_values = agent.target_net(batch_s_)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = batch_r + agent.GAMMA * (1-batch_done) * max_target_q_values

        #compute q_values
        q_values = agent.online_net(batch_s)
        a_q_values = torch.gather(input=q_values, dim=1, index=batch_a)

        #compute loss
        loss = nn.functional.smooth_l1_loss(targets, a_q_values)

        #gradient descent
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())#todo

        print("Episode:{}".format(episode_i))
        print("Avg. Reword:{}".format(np.mean(REWARD_BUFFER[:episode_i])))