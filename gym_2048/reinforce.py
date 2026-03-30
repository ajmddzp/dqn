from __future__ import annotations

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Game2048

# Hyper Parameters
LR = 1e-4
GAMMA = 0.99
RENDER_EVERY_EPISODES = 100
N_ACTIONS = 4
N_STATES = 16
NUM_EPISODES = 12000
MAX_TILE_MILESTONE_BONUSES = [
    (128, 0.2),
    (256, 0.4),
    (512, 0.8),
    (1024, 1.2),
    (2048, 2.0),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self):
        """初始化策略网络结构。"""
        super().__init__()
        self.fc1 = nn.Linear(N_STATES, 512)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(512, 512)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(512, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        """前向计算动作概率分布。"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_scores = self.out(x)
        return F.softmax(action_scores, dim=-1)


class PolicyGradient:
    def __init__(self):
        """初始化策略梯度智能体与优化器。"""
        self.device = device
        self.policy_net = PolicyNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.episode_rewards = []
        self.episode_log_probs = []

    @staticmethod
    def _to_state_vector(s):
        """将棋盘状态转为长度16的log2特征向量。"""
        if isinstance(s, tuple):
            s = s[0]
        arr = np.asarray(s, dtype=np.float32).reshape(-1)
        mask = arr > 0
        arr[mask] = np.log2(arr[mask])
        return arr

    @staticmethod
    def _to_board(s):
        """将状态整理为4x4整数棋盘。"""
        if isinstance(s, tuple):
            s = s[0]
        return np.asarray(s, dtype=np.int64).reshape(4, 4)

    @staticmethod
    def _move_row_left(row: np.ndarray) -> np.ndarray:
        """模拟单行向左移动与合并。"""
        non_zero = row[row != 0]
        merged = []
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged.append(int(non_zero[i] * 2))
                i += 2
            else:
                merged.append(int(non_zero[i]))
                i += 1
        if len(merged) < 4:
            merged.extend([0] * (4 - len(merged)))
        return np.asarray(merged, dtype=np.int64)

    @classmethod
    def _move_left(cls, board: np.ndarray) -> np.ndarray:
        """模拟整盘向左移动。"""
        return np.vstack([cls._move_row_left(row) for row in board])

    @classmethod
    def _move_right(cls, board: np.ndarray) -> np.ndarray:
        """模拟整盘向右移动。"""
        flipped = np.fliplr(board)
        moved = cls._move_left(flipped)
        return np.fliplr(moved)

    @classmethod
    def _apply_action_to_board(cls, board: np.ndarray, action: int) -> np.ndarray:
        """按动作id模拟执行一次移动。"""
        # user action mapping:
        # 0=up, 1=down, 2=left, 3=right
        if action == 0:
            return cls._move_left(board.T).T
        if action == 1:
            return cls._move_right(board.T).T
        if action == 2:
            return cls._move_left(board)
        if action == 3:
            return cls._move_right(board)
        raise ValueError("action must be one of 0, 1, 2, 3")

    @classmethod
    def get_valid_action_mask(cls, state) -> np.ndarray:
        """返回当前状态下4个动作是否合法的布尔掩码。"""
        board = cls._to_board(state)
        valid = np.zeros(N_ACTIONS, dtype=bool)
        for action in range(N_ACTIONS):
            moved = cls._apply_action_to_board(board, action)
            valid[action] = not np.array_equal(board, moved)
        return valid

    def choose_action(self, state, valid_mask=None):
        """根据策略分布采样动作，并返回对应log_prob。"""
        state_vec = self._to_state_vector(state)
        state_t = torch.as_tensor(
            state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        action_probs = self.policy_net(state_t).squeeze(0)

        if valid_mask is not None:
            mask_t = torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device)
            if torch.count_nonzero(mask_t) > 0:
                masked_probs = torch.where(mask_t, action_probs, torch.zeros_like(action_probs))
                prob_sum = masked_probs.sum()
                if prob_sum.item() > 1e-8:
                    action_probs = masked_probs / prob_sum
                else:
                    action_probs = mask_t.float() / mask_t.float().sum()
            else:
                action_probs = torch.full_like(action_probs, 1.0 / N_ACTIONS)

        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return int(action.item()), log_prob

    def store_transition(self, log_prob, reward):
        """缓存当前step的log概率与奖励。"""
        self.episode_log_probs.append(log_prob)
        self.episode_rewards.append(float(reward))

    def learn(self):
        """用整条episode轨迹执行一次REINFORCE更新。"""
        if not self.episode_rewards:
            return None

        returns = []
        R = 0.0
        for r in reversed(self.episode_rewards):
            R = r + GAMMA * R
            returns.insert(0, R)

        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        if returns_t.numel() > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs_t = torch.stack(self.episode_log_probs)
        loss = -(log_probs_t * returns_t).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.episode_rewards.clear()
        self.episode_log_probs.clear()
        return float(loss.item())


def main():
    """训练主循环：采样、累计回报并更新策略。"""
    game = Game2048(
        seed=42, fps=30, window_title="2048 Policy Gradient Train", render_enabled=False
    )
    pg_agent = PolicyGradient()

    print(f"Using device: {device}")
    print("\nTraining with Policy Gradient...")
    time.sleep(2)

    episode_returns = []
    episode_ids = []

    try:
        for i_episode in range(NUM_EPISODES):
            game.set_render_enabled((i_episode + 1) % RENDER_EVERY_EPISODES == 0)
            s, info = game.reset(seed=42 + i_episode)
            done = False
            ep_r = 0.0
            ep_steps = 0

            old_empty_cells = int(info.get("empty_cells", 0))
            old_max_block = int(info.get("max_block", np.max(s)))

            while not done:
                valid_mask = pg_agent.get_valid_action_mask(s)
                a, log_prob = pg_agent.choose_action(s, valid_mask=valid_mask)
                s_, env_reward, done, info = game.step(a)

                max_block = int(info.get("max_block", np.max(s_)))
                is_success = bool(info.get("is_success", False))
                empty_cells = int(info.get("empty_cells", 0))

                # Dense reward shaping:
                # - merge reward as the main signal
                # - max tile / empty cells as auxiliary signals
                # - penalize invalid moves and failed terminations
                r_merge = 0.3 * math.log1p(max(0.0, float(env_reward)))
                r_max = 0.5 * (math.log2(max_block) - math.log2(old_max_block))
                r_empty = 0.02 * (empty_cells - old_empty_cells)
                r_milestone = 0.0

                if max_block > old_max_block:
                    for tile_value, tile_bonus in MAX_TILE_MILESTONE_BONUSES:
                        if old_max_block < tile_value <= max_block:
                            r_milestone += tile_bonus

                invalid_move = np.array_equal(s_, s)
                r_invalid = -0.2 if invalid_move else 0.0
                r_done_fail = -0.8 if (done and not is_success) else 0.0
                r_success = 2.0 if is_success else 0.0

                r = (
                    r_merge
                    + r_max
                    + r_empty
                    + r_milestone
                    + r_invalid
                    + r_done_fail
                    + r_success
                )

                r = float(np.clip(r, -2.0, 2.0))

                pg_agent.store_transition(log_prob, r)
                ep_r += r
                ep_steps += 1

                s = s_
                old_empty_cells = empty_cells
                old_max_block = max_block

                if done:
                    loss = pg_agent.learn()
                    episode_returns.append(ep_r)
                    episode_ids.append(i_episode)
                    print(
                        f"Ep: {i_episode:4d} | "
                        f"Ep_r: {ep_r:7.2f} | "
                        f"Length: {ep_steps:4d} | "
                        f"Loss: {('None' if loss is None else f'{loss:.4f}')} | "
                        f"Max Block: {max_block:5d} | "
                    )
                    break
    finally:
        game.close()

    plt.plot(episode_ids, episode_returns)
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title("REINFORCE on 2048")
    plt.show()


if __name__ == "__main__":
    main()
