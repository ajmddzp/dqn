# PPO-Penalty: 通过KL散度惩罚实现策略更新，动态调整惩罚系数以近似满足KL约束
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time
from torch.distributions import Categorical

# Hyper Parameters
BATCH_SIZE = 4000
LR = 0.01  # 价值网络学习率
LR_POLICY = 0.0003  # 策略网络学习率（PPO通常使用较小的学习率）
GAMMA = 0.99
LAMBDA = 0.95
TARGET_KL = 0.01  # 目标KL散度（对应TRPO的MAX_KL）
KL_PENALTY_COEF = 1.0  # KL惩罚系数初始值
ADAPTIVE_KL_TARGET = 1.5  # 自适应KL调整的倍数阈值
EPOCHS = 10  # 每次更新时对同一批数据迭代次数
EPS = 1e-8

env = gym.make('CartPole-v1', render_mode='human')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(
    env.action_space.sample(), int) else env.action_space.sample().shape


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_logits = self.out(x)
        return F.softmax(action_logits, dim=-1)


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        state_value = self.out(x)
        return state_value


class PPO_Penalty:
    def __init__(self):
        self.policy_net = PolicyNet()
        self.value_net = ValueNet()
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=LR_POLICY)
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=LR)
        self.kl_penalty_coef = KL_PENALTY_COEF  # 动态调整的KL惩罚系数

    def choose_action(self, x):  # 与AC一致，得出概率后采样
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        probs = self.policy_net(x)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def compute_advantages(self, rewards, values, dones, next_value):
        advantages = np.zeros(len(rewards))
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1] * (1 - dones[t])

            delta = rewards[t] + GAMMA * next_val - values[t]
            advantages[t] = delta + GAMMA * LAMBDA * \
                (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        return advantages

    def update_value_net(self, states, returns):
        states = torch.FloatTensor(states)
        returns = torch.FloatTensor(returns).unsqueeze(1)

        for _ in range(10):
            values = self.value_net(states)
            value_loss = F.mse_loss(values, returns)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

    def update_policy(self, states, actions, log_probs_old, advantages):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs_old)
        advantages = torch.FloatTensor(advantages)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + EPS)

        # 获取旧策略的概率分布（用于KL计算）
        with torch.no_grad():
            old_probs = self.policy_net(states)
            old_log_probs = torch.log(old_probs.gather(
                1, actions.unsqueeze(1))).squeeze()
            old_probs_clone = old_probs.clone()

        total_policy_loss = 0.0
        total_kl = 0.0

        # 对同一批数据进行多轮优化
        for _ in range(EPOCHS):
            # 计算新策略
            probs = self.policy_net(states)
            log_probs = torch.log(probs.gather(
                1, actions.unsqueeze(1))).squeeze()

            # 计算比率 r(θ)
            ratio = torch.exp(log_probs - log_probs_old)

            # 计算KL散度 (平均每个样本)
            kl = (old_probs_clone * (torch.log(old_probs_clone + EPS) -
                  torch.log(probs + EPS))).sum(dim=1).mean()

            # PPO-Penalty 损失函数: L = -E[ r(θ) * A ] + β * KL
            policy_loss = -(ratio * advantages).mean() + \
                self.kl_penalty_coef * kl

            # 更新策略网络
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), max_norm=0.5)  # 梯度裁剪
            self.policy_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_kl += kl.item()

        # 计算平均KL并动态调整惩罚系数
        avg_kl = total_kl / EPOCHS
        if avg_kl < TARGET_KL / ADAPTIVE_KL_TARGET:
            self.kl_penalty_coef *= 0.5  # KL太小，减小惩罚
        elif avg_kl > TARGET_KL * ADAPTIVE_KL_TARGET:
            self.kl_penalty_coef *= 2.0  # KL太大，增大惩罚

        return avg_kl


ppo = PPO_Penalty()

print('\nTraining with PPO-Penalty...')
time.sleep(2)

for i_episode in range(400):
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

    s, info = env.reset()
    ep_r = 0

    while True:
        env.render()
        a, log_prob = ppo.choose_action(s)

        s_, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # modify the reward (same as before)
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / \
            env.theta_threshold_radians - 0.5
        r = r1 + r2

        # store transition
        states.append(s)
        actions.append(a)
        log_probs.append(log_prob.item())
        rewards.append(r)
        dones.append(done)

        # value estimate
        with torch.no_grad():
            state_tensor = torch.FloatTensor(s).unsqueeze(0)
            value = ppo.value_net(state_tensor).item()
            values.append(value)

        ep_r += r

        if done:
            # 计算最后一个状态的value
            with torch.no_grad():
                if terminated:
                    next_value = 0.0
                else:
                    next_value = ppo.value_net(
                        torch.FloatTensor(s_).unsqueeze(0)).item()

            # 计算优势函数
            advantages = ppo.compute_advantages(
                rewards, values, dones, next_value)

            # 计算回报
            returns = advantages + values

            # 更新价值网络
            ppo.update_value_net(states, returns)

            # 更新策略网络
            avg_kl = ppo.update_policy(states, actions, log_probs, advantages)

            print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2), '| Avg_KL: ', round(
                avg_kl, 6), '| Beta: ', round(ppo.kl_penalty_coef, 4))
            break

        s = s_

env.close()
