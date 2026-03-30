"""GRPO (Group Relative Policy Optimization) on CartPole-v1.

核心思想：
1. 对每个状态采样多个动作，形成动作组
2. 用组内奖励的相对值（归一化）作为优势估计
3. 不需要价值网络，简化架构
4. 仍然使用PPO的裁剪机制限制策略更新
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
GAMMA = 0.99  # 奖励折扣因子
EPS_CLIP = 0.2  # PPO 裁剪范围 epsilon
K_EPOCHS = 4  # 每次更新时，对同一批数据重复训练轮数
UPDATE_TIMESTEP = 2000  # 累积多少步交互后执行一次参数更新
GROUP_SIZE = 8  # GRPO：对每个状态采样的动作数量

# 创建环境。render_mode='human' 会弹窗实时显示训练过程。
env = gym.make("CartPole-v1", render_mode="human")
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
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


class GRPO:
    def __init__(self):
        # GRPO只需要策略网络，不需要价值网络
        self.policy_net = PolicyNet()
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=LR_POLICY
        )

        # 仅用于日志统计：记录参数更新了多少个 mini-batch。
        self.training_step = 0

    def choose_action(self, x):
        """按当前策略采样动作，并返回该动作在当前策略下的 log_prob。"""
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        probs = self.policy_net(x)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def compute_group_advantages(self, rewards):
        """
        GRPO的核心：在组内计算相对优势
        对一组轨迹（同一个状态的不同动作采样）的回报进行组内归一化

        参数:
            rewards: 列表的列表，每个元素是一组动作对应的累积奖励
                    形状: [group_size, trajectory_length]

        返回:
            advantages: 组内归一化后的优势
        """

        """
        # 如果资源充足，追求稳定性 → 选GRPO
            if GPU_memory_limited and want_stability:
                use_grpo(group_size=8)  # 多花计算时间，但更稳定

        # 如果资源受限，需要快速迭代 → 选PPO
            if time_limited and compute_limited:
                use_ppo()  # 每步计算量小，但需要小心调价值网络

        # 如果有完美模拟器，可以并行 → GRPO优势更大
            if parallel_envs_available:
                use_grpo(group_size=16)  # 并行执行，不增加时间成本
        """
        # 样本均值，估计期望
        # 有偏估计
        # 计算每组的总回报（可以取和或平均，这里用和）
        group_returns = [sum(group_rewards) for group_rewards in rewards]
        group_returns = torch.FloatTensor(group_returns)

        # 组内归一化作为优势

        # 在这里计算差值，得到相对优势
        # 示例demo，计算组间平均与计算组内相对的策略是一样的
        advantages = (group_returns - group_returns.mean()) / (
            group_returns.std() + 1e-8
        )

        return advantages

    def update(self, buffer):
        """GRPO 核心更新过程。"""
        # 从 buffer 取出并转成 Tensor
        # 注意：GRPO的buffer结构不同，包含组信息
        states = torch.FloatTensor(np.array(buffer["states"]))  # [n_groups, state_dim]

        # 展平所有动作和log_probs用于训练
        all_actions = []
        all_old_log_probs = []
        all_advantages = []

        # 为每组计算优势
        # 遍历每一组采样数据（每组通常对应同一个 state 的多次动作采样）
        for i in range(len(buffer["group_rewards"])):
            # 当前组奖励，通常长度为 GROUP_SIZE
            group_rewards = buffer["group_rewards"][i]
            # 当前组动作，与 group_rewards 一一对应
            group_actions = buffer["group_actions"][i]
            # 当前组动作在旧策略下的 log_prob，与 group_actions 一一对应
            group_log_probs = buffer["group_log_probs"][i]

            # 计算组内优势
            # 函数期望输入是“组列表”，因此传入 [group_rewards]（多包一层列表）
            # 返回张量 shape 约为 [1]，后续会用 group_advantages[0].item() 取标量
            group_advantages = self.compute_group_advantages([group_rewards])

            # 基线的优势函数，用样本均值来估计真实的均值

            # 同一组内的所有样本共享同一个优势值
            # 当前组样本数（一般等于 GROUP_SIZE）
            n_samples = len(group_actions)
            # extend 会把 group_actions 中的元素逐个追加到 all_actions（扁平化拼接）
            all_actions.extend(group_actions)
            # 同样把旧策略 log_prob 按样本逐个拼接到总列表
            all_old_log_probs.extend(group_log_probs)
            # 当前实现中：同一组内样本共享一个优势值，复制 n_samples 次后再拼接
            all_advantages.extend([group_advantages[0].item()] * n_samples)

        actions = torch.LongTensor(np.array(all_actions)).unsqueeze(1)
        old_log_probs = torch.FloatTensor(np.array(all_old_log_probs)).unsqueeze(1)
        advantages = torch.FloatTensor(np.array(all_advantages)).unsqueeze(1)

        # 同一批采样数据重复优化 K 次（PPO 的典型做法）。
        for _ in range(K_EPOCHS):
            # 打乱索引并按 mini-batch 采样。
            sampler = BatchSampler(
                SubsetRandomSampler(range(len(actions))),
                batch_size=BATCH_SIZE,
                drop_last=False,
            )

            for indices in sampler:
                # 切出当前 mini-batch。
                batch_states = states[
                    indices // GROUP_SIZE
                ]  # 根据索引映射回原始状态，重现对应关系
                batch_actions = actions[indices]
                batch_old_log_probs = old_log_probs[indices]
                batch_advantages = advantages[indices]

                # 在“当前策略”下重新计算动作 log_prob。
                probs = self.policy_net(batch_states)
                m = Categorical(probs)
                log_probs = m.log_prob(batch_actions.squeeze()).unsqueeze(1)

                # 重要性采样比率 r_t(theta)。
                ratios = torch.exp(log_probs - batch_old_log_probs)

                # PPO 裁剪目标（GRPO使用相同的裁剪机制）
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

                self.training_step += 1


# 初始化 GRPO 智能体
grpo = GRPO()

print("\nTraining with GRPO...")
print(
    f"GRPO Parameters: clip_epsilon={EPS_CLIP}, k_epochs={K_EPOCHS}, "
    f"batch_size={BATCH_SIZE}, group_size={GROUP_SIZE}"
)
time.sleep(2)

# GRPO缓冲区：按组存储数据
buffer = {
    "states": [],  # 每个组对应的状态
    "group_rewards": [],  # 每组对应的奖励序列
    "group_actions": [],  # 每组采样的动作
    "group_log_probs": [],  # 每组动作的log概率
}

timestep = 0
episode_count = 0

# 主训练循环
while episode_count < 400:
    s, info = env.reset()
    ep_r = 0

    # GRPO: 对每个状态采样多个动作
    # 存储当前组的数据
    group_states = []
    group_rewards_list = []
    group_actions_list = []
    group_log_probs_list = []

    while True:
        env.render()

        # GRPO核心：对当前状态采样多个动作
        # 其实还是多次前向传播，蒙特卡洛方法，为计算组间均值做准备
        state_tensor = torch.FloatTensor(s).unsqueeze(0)
        probs = grpo.policy_net(state_tensor)
        m = Categorical(probs)

        # 采样GROUP_SIZE个动作
        group_actions = []
        group_log_probs = []

        for _ in range(GROUP_SIZE):
            action = m.sample()
            group_actions.append(action.item())
            group_log_probs.append(m.log_prob(action).item())

        # 执行动作（这里使用第一个采样的动作与环境交互）
        # 实际应用中可能需要更复杂的策略，这里简化处理
        a = group_actions[0]
        s_, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # 奖励塑形（沿用常见 CartPole 改写方式）：
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (
            env.theta_threshold_radians - abs(theta)
        ) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        # 存储组数据
        group_states.append(s)
        group_actions_list.append(group_actions)
        group_log_probs_list.append(group_log_probs)

        # 为组内每个动作记录相同的奖励（简化处理）
        group_rewards = [r] * GROUP_SIZE
        group_rewards_list.append(group_rewards)

        ep_r += r
        timestep += 1

        s = s_

        if done:
            # 将当前episode的所有组数据添加到buffer
            buffer["states"].extend(group_states)
            buffer["group_rewards"].extend(group_rewards_list)
            buffer["group_actions"].extend(group_actions_list)
            buffer["group_log_probs"].extend(group_log_probs_list)

            print(
                "Episode: ",
                episode_count,
                "| Ep_r: ",
                round(ep_r, 2),
                "| Steps: ",
                timestep,
                "| Update steps: ",
                grpo.training_step,
            )

            episode_count += 1
            break

    # 累积到指定步数后执行一次 GRPO 更新。
    if timestep >= UPDATE_TIMESTEP:
        print(f"\nUpdating GRPO at timestep {timestep}...")
        grpo.update(buffer)

        # 更新后清空 buffer，开始下一轮数据采样。
        buffer = {
            "states": [],
            "group_rewards": [],
            "group_actions": [],
            "group_log_probs": [],
        }
        timestep = 0

env.close()
print("Training completed!")
