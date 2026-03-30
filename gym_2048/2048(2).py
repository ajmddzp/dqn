from __future__ import annotations

from model import Game2048, run
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from model import to_env_action

# Hyper Parameters
BATCH_SIZE = 64
LR = 1e-4  # learning rate
EPSILON = 0.95  # greedy policy
GAMMA = 0.99  # reward discount
TARGET_REPLACE_ITER = 200  # target update frequency
MEMORY_CAPACITY = 10000
RENDER_EVERY_EPISODES = 100
game = Game2048(seed=42, fps=30, window_title="2048 DQN Train", render_enabled=False)
env = game.env
N_ACTIONS = 4
N_STATES = 16
ENV_A_SHAPE = (
    0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
)  # to confirm the shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):  #####################
    def __init__(
        self,
    ):  ################
        """初始化Q网络结构。"""
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 512)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(512, 512)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.fc3 = nn.Linear(512, 512)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(512, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):  ####################
        """前向计算每个动作的Q值。"""
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):

    def __init__(self):
        """初始化DQN双网络、经验池与优化器。"""
        self.device = device
        self.eval_net, self.target_net = Net().to(self.device), Net().to(
            self.device
        )  # 双网络初始化

        self.learn_step_counter = 0  # 学习步数初始化       # for target updating
        self.memory_counter = 0  # 经验存储位置计数器            # for storing memory
        # [s(16), a(1), r(1), done(1), s'(16)]
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3))  # initialize memory
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=LR
        )  # 使用Adam优化器
        self.loss_func = nn.MSELoss()  # 均方差损失函数

    @staticmethod
    def _to_state_vector(s):
        """将棋盘状态转为长度16的log2特征向量。"""
        if isinstance(s, tuple):
            s = s[0]
        arr = np.asarray(s, dtype=np.float32).reshape(-1)
        # Normalize board values: 0 stays 0, others map to log2(tile).
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

    def choose_action(self, x):  ##################
        """用epsilon-greedy在合法动作中选择动作。"""
        valid_mask = self.get_valid_action_mask(x)
        valid_actions = np.flatnonzero(valid_mask)

        x = self._to_state_vector(x)
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            with torch.no_grad():
                actions_value = self.eval_net(x).squeeze(0)  # 前向传播
                if valid_actions.size > 0:
                    mask_t = torch.as_tensor(
                        valid_mask, dtype=torch.bool, device=self.device
                    )
                    actions_value = actions_value.masked_fill(~mask_t, -1e9)
                action = int(torch.argmax(actions_value, dim=0).item())
        else:  # random
            if valid_actions.size > 0:
                action = int(np.random.choice(valid_actions))
            else:
                action = int(np.random.randint(0, N_ACTIONS))
        action = action if ENV_A_SHAPE == 0 else np.array(action).reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_, done):
        """将一条转移样本写入经验回放池。"""
        s = self._to_state_vector(s)
        s_ = self._to_state_vector(s_)

        transition = np.hstack((s, [int(a), r, float(bool(done))], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        """采样批量经验并执行一次DQN参数更新。"""
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(
                self.eval_net.state_dict()
            )  # 更新目标网络，从eval中拷贝参数
        self.learn_step_counter += 1  # 计数器

        # sample batch transitions
        sample_index = np.random.choice(
            MEMORY_CAPACITY, BATCH_SIZE
        )  # 从2000个经验中选取32个索引
        b_memory = self.memory[sample_index, :]  # 从内存中取得经验

        b_s = torch.as_tensor(
            b_memory[:, :N_STATES], dtype=torch.float32, device=self.device
        )  # 数据转换类型
        # 采样的经验，状态
        b_a = torch.as_tensor(
            b_memory[:, N_STATES : N_STATES + 1].astype(int),
            dtype=torch.long,
            device=self.device,
        )
        # 采样的经验，动作
        b_r = torch.as_tensor(
            b_memory[:, N_STATES + 1 : N_STATES + 2],
            dtype=torch.float32,
            device=self.device,
        )
        # 采样的经验，done标志
        b_done = torch.as_tensor(
            b_memory[:, N_STATES + 2 : N_STATES + 3],
            dtype=torch.float32,
            device=self.device,
        )
        # 采样的经验，奖励
        b_s_ = torch.as_tensor(
            b_memory[:, -N_STATES:], dtype=torch.float32, device=self.device
        )  # 负索引，表示从后往前
        # 采样的经验，下一个状态

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        # 前向传播，并获得索引为[1],[b_a]的值
        with torch.no_grad():
            q_next = self.target_net(b_s_)
            q_target = b_r + (1.0 - b_done) * GAMMA * q_next.max(1)[0].view(
                BATCH_SIZE, 1
            )  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)  # 计算loss

        self.optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 反向传播计算梯度，求loss对每个变量的梯度
        self.optimizer.step()  # 由梯度下降法优化参数
        return float(loss.item())


dqn = DQN()
print(f"Using device: {device}")

print("\nCollecting experience...")
time.sleep(2)
for i_episode in range(12000):  ###########
    game.set_render_enabled((i_episode + 1) % RENDER_EVERY_EPISODES == 0)
    s, info = game.reset(seed=42 + i_episode)
    done = False
    ep_r = 0.0
    ep_steps = 0
    last_loss = None

    old_empty_cells = 0  # 上一轮的空格数量
    old_max_block = 2  # 上一轮的最大块

    while not done:
        a = dqn.choose_action(s)  # 选择动作
        env_action = to_env_action(int(a))

        s_, env_reward, done, info = game.step(int(a))

        # 读取 model.step() 返回的 info 字段
        end_value = info.get("end_value", 0)  # 全局数字之和
        max_block = info.get("max_block", 0)  # 最大块
        is_success = info.get("is_success", False)  # 是否达到2048
        empty_cells = info.get("empty_cells", 0)  # 空格
        step_id = info.get("step_id", 0)  # step_id

        # Dense reward shaping:
        # - merge reward as the main signal
        # - max tile / empty cells as auxiliary signals
        # - penalize invalid moves and failed terminations
        r_merge = 0.3 * math.log1p(max(0.0, float(env_reward)))
        r_max = 0.3 * (math.log2(max_block) - math.log2(old_max_block))
        r_empty = 0.02 * (empty_cells - old_empty_cells)
        invalid_move = np.array_equal(s_, s)
        r_invalid = -0.2 if invalid_move else 0.0
        r_done_fail = -0.8 if (done and not is_success) else 0.0
        r_success = 2.0 if is_success else 0.0
        r = r_merge + r_max + r_empty + r_invalid + r_done_fail + r_success
        r = np.clip(r, -2.0, 2.0)

        dqn.store_transition(s, a, r, s_, done)  # 经验存储
        ep_r += r
        ep_steps += 1

        if dqn.memory_counter > MEMORY_CAPACITY:
            last_loss = dqn.learn()

        if done:
            loss_text = "None" if last_loss is None else f"{last_loss:.6f}"
            print(
                f"Ep: {i_episode} | Steps: {ep_steps} | Ep_r: {ep_r:.2f} | Loss: {loss_text} | Max_block: {max_block}"
            )
            break
        s = s_
        old_empty_cells = empty_cells
        old_max_block = max_block

game.close()
