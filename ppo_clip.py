# PPO: Proximal Policy Optimization
'''
PPO是对TRPO的简化改进，通过裁剪(crop)目标函数来限制策略更新步长。
核心思想：如果新旧策略的比率超出[1-ε, 1+ε]范围，就进行裁剪，防止更新过大。
相比TRPO，PPO实现更简单，计算效率更高，且效果相当或更好。
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# Hyper Parameters
BATCH_SIZE = 128  # 批量大小
LR_POLICY = 3e-4  # 策略网络学习率
LR_VALUE = 1e-3   # 价值网络学习率
GAMMA = 0.99      # 奖励折扣
LAMBDA = 0.95     # GAE参数
EPS_CLIP = 0.2    # 裁剪参数
K_EPOCHS = 4      # 每次更新迭代次数
UPDATE_TIMESTEP = 2000  # 每收集这么多步数更新一次

env = gym.make('CartPole-v1', render_mode='human')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(
    env.action_space.sample(), int) else env.action_space.sample().shape


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, N_ACTIONS)

        # 初始化权重
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.out.weight, gain=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.out(x)
        return F.softmax(action_logits, dim=-1)


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

        # 初始化权重
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.out.weight, gain=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        state_value = self.out(x)
        return state_value


class PPO:
    def __init__(self):
        self.policy_net = PolicyNet()
        self.value_net = ValueNet()
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=LR_POLICY)
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=LR_VALUE)

        self.training_step = 0

    def choose_action(self, x):  # 与AC一致，得出概率后采样
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        probs = self.policy_net(x)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def compute_gae(self, rewards, values, dones, next_value):
        """计算广义优势估计(GAE)"""
        # 优势函数：Q(s)-V(s)
        advantages = np.zeros(len(rewards))
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_value
            else:
                next_value = values[t + 1] * (1 - dones[t])

            delta = rewards[t] + GAMMA * next_value - values[t]
            advantages[t] = delta + GAMMA * LAMBDA * \
                (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        returns = advantages + values
        return advantages, returns

    def update(self, buffer):
        """PPO核心更新算法"""
        # 解包buffer数据
        states = torch.FloatTensor(np.array(buffer['states']))
        actions = torch.LongTensor(np.array(buffer['actions'])).unsqueeze(1)
        old_log_probs = torch.FloatTensor(
            np.array(buffer['log_probs'])).unsqueeze(1)
        advantages = torch.FloatTensor(
            np.array(buffer['advantages'])).unsqueeze(1)
        returns = torch.FloatTensor(np.array(buffer['returns'])).unsqueeze(1)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)

        # 多次迭代更新
        for _ in range(K_EPOCHS):
            # 使用小批量采样
            sampler = BatchSampler(
                SubsetRandomSampler(range(len(states))),
                batch_size=BATCH_SIZE,
                drop_last=False
            )

            for indices in sampler:
                # 获取当前批次数据
                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_old_log_probs = old_log_probs[indices]
                batch_advantages = advantages[indices]
                batch_returns = returns[indices]

                # 计算当前策略的log概率
                probs = self.policy_net(batch_states)
                m = Categorical(probs)
                log_probs = m.log_prob(batch_actions.squeeze()).unsqueeze(1)

                # 计算比率
                ratios = torch.exp(log_probs - batch_old_log_probs)

                # PPO裁剪目标函数
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - EPS_CLIP,
                                    1 + EPS_CLIP) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 更新策略网络
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.policy_optimizer.step()

                # 计算价值损失并更新价值网络
                values = self.value_net(batch_states)
                value_loss = F.mse_loss(values, batch_returns)

                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.value_optimizer.step()

                self.training_step += 1


# 初始化PPO
ppo = PPO()

print('\nTraining with PPO...')
print(
    f'PPO Parameters: clip_epsilon={EPS_CLIP}, k_epochs={K_EPOCHS}, batch_size={BATCH_SIZE}')
time.sleep(2)

# 训练循环
buffer = {
    'states': [],
    'actions': [],
    'log_probs': [],
    'rewards': [],
    'dones': [],
    'values': [],
    'advantages': [],
    'returns': []
}

timestep = 0
episode_count = 0

while episode_count < 400:
    s, info = env.reset()
    ep_r = 0
    episode_states, episode_actions, episode_log_probs = [], [], []
    episode_rewards, episode_dones, episode_values = [], [], []

    while True:
        env.render()

        # 选择动作
        a, log_prob = ppo.choose_action(s)

        # 执行动作
        s_, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # 修改奖励（保持与原代码一致）
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / \
            env.theta_threshold_radians - 0.5
        r = r1 + r2

        # 存储transition
        episode_states.append(s)
        episode_actions.append(a)
        episode_log_probs.append(log_prob.item())
        episode_rewards.append(r)
        episode_dones.append(done)

        # 计算状态价值
        with torch.no_grad():
            state_tensor = torch.FloatTensor(s).unsqueeze(0)
            value = ppo.value_net(state_tensor).item()
            episode_values.append(value)

        ep_r += r
        timestep += 1

        s = s_

        if done:
            # 计算最后一个状态的value
            with torch.no_grad():
                if terminated:
                    next_value = 0.0
                else:
                    next_value = ppo.value_net(
                        torch.FloatTensor(s_).unsqueeze(0)).item()

            # 计算GAE优势和回报
            advantages, returns = ppo.compute_gae(
                episode_rewards, episode_values, episode_dones, next_value)

            # 添加到主buffer
            buffer['states'].extend(episode_states)
            buffer['actions'].extend(episode_actions)
            buffer['log_probs'].extend(episode_log_probs)
            buffer['rewards'].extend(episode_rewards)
            buffer['dones'].extend(episode_dones)
            buffer['values'].extend(episode_values)
            buffer['advantages'].extend(advantages.tolist())
            buffer['returns'].extend(returns.tolist())

            print('Episode: ', episode_count, '| Ep_r: ', round(ep_r, 2),
                  '| Steps: ', timestep, '| Update steps: ', ppo.training_step)

            episode_count += 1
            break

    # 当收集到足够多的步数时进行更新
    if timestep >= UPDATE_TIMESTEP:
        print(f'\nUpdating PPO at timestep {timestep}...')
        ppo.update(buffer)

        # 清空buffer
        buffer = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'advantages': [],
            'returns': []
        }
        timestep = 0

env.close()
print('Training completed!')
