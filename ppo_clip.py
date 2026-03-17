"""PPO (Proximal Policy Optimization) on CartPole-v1.

核心思想：
1. 用策略比率 r_t = pi(a|s) / pi_old(a|s) 衡量新旧策略差异。
2. 对 r_t 做裁剪（clip）限制更新幅度，避免策略一次更新过大导致训练崩掉。
3. 价值网络估计 V(s)，并用 GAE 计算优势函数，降低方差、提升稳定性。
"""

import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# =========================
# 超参数（可按需调节）
# =========================
BATCH_SIZE = 128  # 每个 mini-batch 的样本数
LR_POLICY = 3e-4  # 策略网络学习率
LR_VALUE = 1e-3  # 价值网络学习率
GAMMA = 0.99  # 奖励折扣因子
LAMBDA = 0.95  # GAE 的 lambda 参数
EPS_CLIP = 0.2  # PPO 裁剪范围 epsilon
K_EPOCHS = 4  # 每次更新时，对同一批数据重复训练轮数
UPDATE_TIMESTEP = 2000  # 累积多少步交互后执行一次参数更新

# 创建环境。render_mode='human' 会弹窗实时显示训练过程。
env = gym.make("CartPole-v1", render_mode="human")
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
# 连续动作场景下才会用到动作形状；CartPole 是离散动作，这里最终是 0。
ENV_A_SHAPE = (
    0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
)


class PolicyNet(nn.Module):
    """策略网络：输入状态，输出每个离散动作的概率分布。"""

    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, N_ACTIONS)

        # 使用正交初始化，通常可提升策略梯度方法的训练稳定性。
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        # 输出层使用更小 gain，避免初始策略过于“自信”。
        nn.init.orthogonal_(self.out.weight, gain=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.out(x)
        # 对离散动作空间输出 softmax 概率。
        return F.softmax(action_logits, dim=-1)


class ValueNet(nn.Module):
    """价值网络：输入状态，输出状态价值 V(s)。"""

    def __init__(self):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

        # 价值网络也采用正交初始化，输出层 gain 可略大。
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
        # 分别维护策略网络与价值网络，并使用独立优化器。
        self.policy_net = PolicyNet()
        self.value_net = ValueNet()
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=LR_POLICY
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=LR_VALUE
        )

        # 仅用于日志统计：记录参数更新了多少个 mini-batch。
        self.training_step = 0

    def choose_action(self, x):
        """按当前策略采样动作，并返回该动作在当前策略下的 log_prob。

        这里存下 old_log_prob，后续 PPO 更新时会和新策略 log_prob 做比率。
        """
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        probs = self.policy_net(x)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def compute_gae(self, rewards, values, dones, next_value):
        # 优势函数，价值函数的pro plus版本，具体怎么算的就不管了
        """计算广义优势估计（GAE）和回报。

        advantages[t] 近似 A_t
        returns[t] = advantages[t] + values[t]
        """
        # 从后往前递推计算优势，能自然利用 bootstrapping。
        advantages = np.zeros(len(rewards))
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # 轨迹最后一步使用外部给定的 next_value（终止时一般为 0）。
                next_val = next_value
            else:
                # 非最后一步，next_value 来自下一个状态价值。
                # 若 done=True，后续价值不应继续传播（乘 0）。
                next_val = values[t + 1] * (1 - dones[t])

            # TD 残差 delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + GAMMA * next_val - values[t]
            # GAE 递推：A_t = delta_t + gamma * lambda * A_{t+1}
            advantages[t] = delta + GAMMA * LAMBDA * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        # 回报目标：R_t = A_t + V(s_t)，用于训练价值网络。
        returns = advantages + values
        return advantages, returns

    def update(self, buffer):
        """PPO 核心更新过程。"""
        # 从 buffer 取出并转成 Tensor。
        states = torch.FloatTensor(np.array(buffer["states"]))
        actions = torch.LongTensor(np.array(buffer["actions"])).unsqueeze(1)
        old_log_probs = torch.FloatTensor(np.array(buffer["log_probs"])).unsqueeze(1)
        advantages = torch.FloatTensor(np.array(buffer["advantages"])).unsqueeze(1)
        returns = torch.FloatTensor(np.array(buffer["returns"])).unsqueeze(1)

        # 优势标准化是常见技巧，可降低梯度方差。
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 同一批采样数据重复优化 K 次（PPO 的典型做法）。
        for _ in range(K_EPOCHS):
            # 打乱索引并按 mini-batch 采样。
            sampler = BatchSampler(
                SubsetRandomSampler(range(len(states))),
                batch_size=BATCH_SIZE,
                drop_last=False,
            )

            for indices in sampler:
                # 切出当前 mini-batch。
                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_old_log_probs = old_log_probs[indices]
                batch_advantages = advantages[indices]
                batch_returns = returns[indices]

                # 在“当前策略”下重新计算动作 log_prob。
                probs = self.policy_net(batch_states)
                m = Categorical(probs)
                log_probs = m.log_prob(batch_actions.squeeze()).unsqueeze(1)

                # 重要性采样比率 r_t(theta)。
                ratios = torch.exp(log_probs - batch_old_log_probs)

                # PPO 裁剪目标：
                # surr1: 原始目标
                # surr2: 裁剪后的保守目标
                # 取 min 等价于对收益过高的更新做“刹车”。
                surr1 = ratios * batch_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # 更新策略网络。
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                # 梯度裁剪：进一步提升训练稳定性。
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.policy_optimizer.step()

                # 更新价值网络（回归 returns）。
                values = self.value_net(batch_states)
                value_loss = F.mse_loss(values, batch_returns)

                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.value_optimizer.step()

                self.training_step += 1


# 初始化 PPO 智能体
ppo = PPO()

print("\nTraining with PPO...")
print(
    f"PPO Parameters: clip_epsilon={EPS_CLIP}, k_epochs={K_EPOCHS}, batch_size={BATCH_SIZE}"
)
time.sleep(2)

# 经验缓冲区：先收集轨迹，再批量更新。
buffer = {
    "states": [],
    "actions": [],
    "log_probs": [],
    "rewards": [],
    "dones": [],
    "values": [],
    "advantages": [],
    "returns": [],
}

timestep = 0
episode_count = 0

# 主训练循环：按 episode 与环境交互，按 timestep 触发更新。
while episode_count < 400:
    s, info = env.reset()
    ep_r = 0
    # 先暂存一个 episode 的数据，结束时统一计算 GAE。
    episode_states, episode_actions, episode_log_probs = [], [], []
    episode_rewards, episode_dones, episode_values = [], [], []

    while True:
        env.render()

        # 根据当前策略采样动作。
        a, log_prob = ppo.choose_action(s)

        # 与环境交互一步。
        s_, r, terminated, truncated, info = env.step(a)
        # Gymnasium 下，terminated(任务失败/成功) 或 truncated(时间截断) 都视为结束。
        done = terminated or truncated

        # 奖励塑形（沿用常见 CartPole 改写方式）：
        # 让小车偏移和杆子角度越小，奖励越高。
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (
            env.theta_threshold_radians - abs(theta)
        ) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        # 存储 transition 信息。
        episode_states.append(s)
        episode_actions.append(a)
        episode_log_probs.append(log_prob.item())
        episode_rewards.append(r)
        episode_dones.append(done)

        # 记录当前状态价值，后续用于 GAE。
        with torch.no_grad():
            state_tensor = torch.FloatTensor(s).unsqueeze(0)
            value = ppo.value_net(state_tensor).item()
            episode_values.append(value)

        ep_r += r
        timestep += 1

        s = s_

        if done:
            # 计算末状态价值：
            # - 若自然终止（terminated），下一状态价值按 0 处理；
            # - 若仅时间截断（truncated），可继续 bootstrap。
            with torch.no_grad():
                if terminated:
                    next_value = 0.0
                else:
                    next_value = ppo.value_net(
                        torch.FloatTensor(s_).unsqueeze(0)
                    ).item()

            # 按当前 episode 计算优势与回报。
            advantages, returns = ppo.compute_gae(
                episode_rewards, episode_values, episode_dones, next_value
            )

            # 追加到全局 buffer，等待达到 UPDATE_TIMESTEP 后统一训练。
            buffer["states"].extend(episode_states)
            buffer["actions"].extend(episode_actions)
            buffer["log_probs"].extend(episode_log_probs)
            buffer["rewards"].extend(episode_rewards)
            buffer["dones"].extend(episode_dones)
            buffer["values"].extend(episode_values)
            buffer["advantages"].extend(advantages.tolist())
            buffer["returns"].extend(returns.tolist())

            print(
                "Episode: ",
                episode_count,
                "| Ep_r: ",
                round(ep_r, 2),
                "| Steps: ",
                timestep,
                "| Update steps: ",
                ppo.training_step,
            )

            episode_count += 1
            break

    # 累积到指定步数后执行一次 PPO 更新。
    if timestep >= UPDATE_TIMESTEP:
        print(f"\nUpdating PPO at timestep {timestep}...")
        ppo.update(buffer)

        # 更新后清空 buffer，开始下一轮数据采样。
        buffer = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "advantages": [],
            "returns": [],
        }
        timestep = 0

env.close()
print("Training completed!")
