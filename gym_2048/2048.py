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
LR = 1e-3  # learning rate
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
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 512)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(512, 512)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(512, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):  ####################
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):

    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()  # 双网络初始化

        self.learn_step_counter = 0  # 学习步数初始化       # for target updating
        self.memory_counter = 0  # 经验存储位置计数器            # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=LR
        )  # 使用Adam优化器
        self.loss_func = nn.MSELoss()  # 均方差损失函数

    @staticmethod
    def _to_state_vector(s):
        if isinstance(s, tuple):
            s = s[0]
        return np.asarray(s, dtype=np.float32).reshape(-1)

    def choose_action(self, x):  ##################
        x = self._to_state_vector(x)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net.forward(x)  # 前向传播
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = (
                action[0] if ENV_A_SHAPE == 0 else np.array(action).reshape(ENV_A_SHAPE)
            )  # return the argmax index
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
            action = (
                action if ENV_A_SHAPE == 0 else np.array(action).reshape(ENV_A_SHAPE)
            )
        return action

    def store_transition(self, s, a, r, s_):
        s = self._to_state_vector(s)
        s_ = self._to_state_vector(s_)

        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
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

        b_s = torch.FloatTensor(b_memory[:, :N_STATES])  # 数据转换类型
        # 采样的经验，状态
        b_a = torch.LongTensor(b_memory[:, N_STATES : N_STATES + 1].astype(int))
        # 采样的经验，动作
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1 : N_STATES + 2])
        # 采样的经验，奖励
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])  # 负索引，表示从后往前
        # 采样的经验，下一个状态

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        # 前向传播，并获得索引为[1],[b_a]的值
        q_next = self.target_net(
            b_s_
        ).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(
            BATCH_SIZE, 1
        )  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)  # 计算loss

        self.optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 反向传播计算梯度，求loss对每个变量的梯度
        self.optimizer.step()  # 由梯度下降法优化参数
        return float(loss.item())


dqn = DQN()

print("\nCollecting experience...")
time.sleep(2)
for i_episode in range(400):  ###########
    game.set_render_enabled((i_episode + 1) % RENDER_EVERY_EPISODES == 0)
    s, info = game.reset(seed=42 + i_episode)
    done = False
    ep_r = 0.0
    ep_steps = 0
    last_loss = None
    while not done:
        a = dqn.choose_action(s)  # 选择动作
        env_action = to_env_action(int(a))
        can_move = game.env.unwrapped.is_action_possible(env_action)
        if not can_move:
            reward = -10.0  # 不合法动作惩罚
            s_ = s  # 状态不变
            done = False  # 不结束
            info = {
                "empty_cells": int(np.count_nonzero(s == 0)),
                "max_block": int(np.max(s)),
            }
            max_block = max(2, info["max_block"])
            empty_cells = info["empty_cells"]
            r = reward + empty_cells + math.log2(max_block)
        else:
            s_, reward, done, info = game.step(int(a))

            reward = min(reward, 0)  # 惩罚，如果这个动作没有带来任何的数字合并

            # 读取 model.step() 返回的 info 字段
            end_value = info.get("end_value", 0)  # 全局数字之和
            max_block = info.get("max_block", 0)  # 最大块
            is_success = info.get("is_success", False)  # 是否达到2048
            empty_cells = info.get("empty_cells", 0)  # 空格
            step_id = info.get("step_id", 0)  # step_id

            r = reward + empty_cells + math.log2(max_block)
            # 奖励函数：奖励 + 空格数量 + log2(最大块)，鼓励玩家获得更多空格和更大的块

        dqn.store_transition(s, a, r, s_)  # 经验存储
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
game.close()
