import random
import numpy as np
import torch
import torch.nn as nn

"""经验回放库"""


class Replaymemory:
    def __init__(self, n_s, n_a):
        self.n_s = n_s  # 状态维度
        self.n_a = n_a  # 动作数量
        self.MEMORY_SIZE = 1000  # 记忆库容量
        self.BATCH_SIZE = 1  # 每次学习的样本数量

        # 存储1000个状态，每个状态是n_s维向量，例如：[ 0.012, -0.045, 0.123, -0.089 ]
        self.all_s = np.empty(
            shape=(self.MEMORY_SIZE, self.n_s), dtype=np.float32)
        # 存储1000个动作（0到n_a-1的整数）
        # self.all_a = np.random.randint(low=0, high=n_a, self.MEMORY_SIZE, dtype=np.uint8)
        self.all_a = np.random.randint(0, n_a, self.MEMORY_SIZE, np.uint8)
        # 存储1000个奖励值
        self.all_r = np.empty(self.MEMORY_SIZE, dtype=np.float32)
        # 存储1000个"是否结束"标志
        # self.all_done = np.random.randint(low=0, high=n_a, self.MEMORY_SIZE, dtype=np.uint8)
        self.all_done = np.random.randint(0, n_a, self.MEMORY_SIZE, np.uint8)
        # 存储1000个下一个状态
        self.all_s_ = np.empty(
            shape=(self.MEMORY_SIZE, self.n_s), dtype=np.float32)
        self.t_memo = 0
        self.t_max = 0

    def add_memo(self, s, a, r, done, s_):
        """添加经验，把新经验存到当前位置"""
        self.all_s[self.t_memo] = s  # 当前状态
        self.all_a[self.t_memo] = a  # 执行的动作
        self.all_r[self.t_memo] = r  # 获得的奖励
        self.all_done[self.t_memo] = done  # 是否结束
        self.all_s_[self.t_memo] = s_  # 下一个状态

        # 更新指针（循环使用记忆库）
        self.t_max = max(self.t_max, self.t_memo + 1)
        self.t_memo = (self.t_memo + 1) % self.MEMORY_SIZE

    def sample(self):

        if self.t_max >= self.BATCH_SIZE:
            idxes = random.sample(range(0, self.t_max), self.BATCH_SIZE)
        else:
            idxes = range(0, self.t_max)
        # 从记忆库中随机抽取BATCH_SIZE个经验
        idxes = random.sample(range(self.t_max), self.BATCH_SIZE)

        # # 检查是否有足够样本
        # if self.t_max < self.BATCH_SIZE:
        #     return None  # 样本不足，返回None

        # # 只有样本足够时才采样
        # idxes = random.sample(range(self.t_max), self.BATCH_SIZE)

        batch_s = []
        batch_a = []
        batch_r = []
        batch_done = []
        batch_s_ = []

        for idx in idxes:
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            batch_done.append(self.all_done[idx])
            batch_s_.append(self.all_s_[idx])
        # 把这些经验转换成PyTorch张量
        batch_s_tensor = torch.as_tensor(
            np.asarray(batch_s), dtype=torch.float32)
        batch_a_tensor = torch.as_tensor(np.asarray(
            batch_a), dtype=torch.int64).unsqueeze(-1)
        batch_r_tensor = torch.as_tensor(np.asarray(
            batch_r), dtype=torch.float32).unsqueeze(-1)
        batch_done_tensor = torch.as_tensor(np.asarray(
            batch_done), dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(
            np.asarray(batch_s_), dtype=torch.float32)

        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_done_tensor, batch_s__tensor


class DQN(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=n_input, out_features=88),  # 输入层→隐藏层
            nn.Tanh(),                                        # 激活函数
            nn.Linear(in_features=88, out_features=n_output)  # 隐藏层→输出层
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)  # 转成张量
        q_value = self(obs_tensor.unsqueeze(0))                 # 神经网络计算价值
        max_q_idx = torch.argmax(input=q_value)                 # 选价值最高的动作
        action = max_q_idx.detach().item()                        # 转成普通数字
        return action


class Agent:
    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output

        self.GAMMA = 0.99           # 折扣因子（未来奖励的重要性）
        self.learning_rate = 1e-3   # 学习速度

        self.memo = Replaymemory(self.n_input, self.n_output)   # 记忆库

        # 两个神经网络：在线网络（实时学习）和目标网络（稳定学习
        self.online_net = DQN(self.n_input, self.n_output)  # todo
        self.target_net = DQN(self.n_input, self.n_output)  # todo

        # 优化器：调整神经网络参数
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=self.learning_rate)  # todo
