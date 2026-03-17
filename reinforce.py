import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time
import matplotlib.pyplot as plt

# Hyper Parameters
LR = 0.01                   # learning rate
GAMMA = 0.99                # reward discount
# env = gym.make('CartPole-v1', render_mode='human')
env = gym.make('CartPole-v1')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
print(N_ACTIONS)


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_scores = self.out(x)
        return F.softmax(action_scores, dim=-1)  # 只在这里添加softmax


class PolicyGradient(object):

    def __init__(self):
        self.policy_net = PolicyNetwork()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)

        # 用于存储轨迹
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_net(state)  # 获取动作概率分布

        # 根据概率分布采样动作
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob

    def store_transition(self, state, action, log_prob, reward):
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_log_probs.append(log_prob)
        self.episode_rewards.append(reward)

    def learn(self):
        # 计算每个时间步的折扣回报
        returns = []
        R = 0
        for r in reversed(self.episode_rewards):
            R = r + GAMMA * R
            returns.insert(0, R)

        # 标准化回报（减少方差）
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 标准化

        # 计算策略梯度损失
        policy_loss = []
        for log_prob, R in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * R)  # 负号因为我们要最大化回报

        # 梯度更新
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        # 清空当前轨迹
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []


# 创建策略梯度智能体
pg_agent = PolicyGradient()

print('\nTraining with Policy Gradient...')
time.sleep(2)

X = []
Y = []

for i_episode in range(500):
    s, info = env.reset()
    ep_r = 0
    episode_length = 0

    while True:
        env.render()

        # 选择动作
        a, log_prob = pg_agent.choose_action(s)

        # 执行动作
        s_, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # 修改奖励函数（可选，保持与原始代码一致）
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / \
            env.theta_threshold_radians - 0.5
        r = r1 + r2

        # 存储转换
        pg_agent.store_transition(s, a, log_prob, r)

        ep_r += r
        episode_length += 1
        s = s_

        if done or ep_r >= 400:
            # 每个episode结束后进行学习
            pg_agent.learn()
            X.append(ep_r)
            Y.append(i_episode)
            print(f'Ep: {i_episode:3d} | '
                  f'Ep_r: {round(ep_r, 2):6.2f} | '
                  f'Length: {episode_length:3d}')
            break

plt.plot(Y, X)
plt.show()

env.close()
