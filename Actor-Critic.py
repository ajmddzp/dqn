import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
GAMMA = 0.9                 # reward discount
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v1', render_mode='human')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]

# Actor网络 - 基于原始网络结构，输出动作概率


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_scores = self.out(x)
        return F.softmax(action_scores, dim=-1)  # 输出动作概率

# Critic网络 - 基于原始网络结构，输出状态价值


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, 1)  # 输出单个状态价值
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        state_value = self.out(x)
        return state_value


class ActorCritic(object):
    def __init__(self):
        self.actor_net = ActorNet()
        self.critic_net = CriticNet()

        self.actor_optimizer = torch.optim.Adam(
            self.actor_net.parameters(), lr=LR)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_net.parameters(), lr=LR)

        # 经验回放缓冲区
        self.memory_counter = 0
        self.memory = np.zeros(
            (MEMORY_CAPACITY, N_STATES * 2 + 3))  # s, a, r, s_, done

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        action_probs = self.actor_net(x)

        # 根据概率分布采样动作
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob

    def store_transition(self, s, a, log_prob, r, s_, done):
        if isinstance(s, tuple):
            s = s[0]
        if isinstance(s_, tuple):
            s_ = s_[0]

        transition = np.hstack((s, [a, log_prob.item(), r], s_, [done]))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.memory_counter < BATCH_SIZE:
            return

        # 随机采样经验
        sample_index = np.random.choice(
            min(self.memory_counter, MEMORY_CAPACITY), BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_log_prob = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_r = torch.FloatTensor(b_memory[:, N_STATES+2:N_STATES+3])
        b_s_ = torch.FloatTensor(b_memory[:, N_STATES+3:N_STATES+3+N_STATES])
        b_done = torch.BoolTensor(b_memory[:, -1].astype(bool))

        # Critic学习：计算TD误差
        current_values = self.critic_net(b_s).squeeze()
        next_values = self.critic_net(b_s_).squeeze().detach()
        target_values = b_r.squeeze() + GAMMA * next_values * (~b_done).float()
        # 状态价值函数值

        # Critic损失：价值函数拟合
        critic_loss = F.mse_loss(current_values, target_values)

        # Actor学习：使用优势函数
        # TD误差作为优势函数，利用critic评价进行梯度下降
        advantage = (target_values - current_values).detach()
        actor_loss = -(b_log_prob.squeeze() * advantage).mean()

        # 更新网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


# 创建Actor-Critic智能体
ac_agent = ActorCritic()

print('\nTraining with Actor-Critic...')
time.sleep(2)

for i_episode in range(400):
    s, info = env.reset()
    ep_r = 0
    step_count = 0

    while True:
        env.render()

        # 选择动作
        a, log_prob = ac_agent.choose_action(s)

        # 执行动作
        s_, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # 修改奖励（保持与原始代码一致）
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / \
            env.theta_threshold_radians - 0.5
        r = r1 + r2

        # 存储经验
        ac_agent.store_transition(s, a, log_prob, r, s_, done)

        # 学习
        ac_agent.learn()

        ep_r += r
        step_count += 1
        s = s_

        if done:
            print(f'Ep: {i_episode:3d} | '
                  f'Ep_r: {round(ep_r, 2):6.2f} | '
                  f'Steps: {step_count:3d}')
            break

env.close()
