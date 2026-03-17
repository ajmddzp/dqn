import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make("CartPole-v1", render_mode="human")
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = (
    0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
)  # to confirm the shape


class Net(nn.Module):
    def __init__(
        self,
    ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DoubleDQN(object):

    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()  # 双网络初始化

        self.learn_step_counter = 0  # 学习步数初始化
        self.memory_counter = 0  # 经验存储位置计数器
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=LR
        )  # 使用Adam优化器
        self.loss_func = nn.MSELoss()  # 均方差损失函数

    def choose_action(self, x):
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
        if isinstance(s, tuple):
            s = s[0]
        if isinstance(s_, tuple):
            s_ = s_[0]

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

        b_s = torch.FloatTensor(b_memory[:, :N_STATES])  # 采样的经验，状态
        b_a = torch.LongTensor(
            b_memory[:, N_STATES : N_STATES + 1].astype(int)
        )  # 采样的经验，动作
        b_r = torch.FloatTensor(
            b_memory[:, N_STATES + 1 : N_STATES + 2]
        )  # 采样的经验，奖励
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])  # 采样的经验，下一个状态

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)

        # Double DQN 修改部分
        # 使用评估网络选择下一个状态的最佳动作
        q_eval_next = self.eval_net(b_s_)
        best_actions = q_eval_next.max(1)[1].view(BATCH_SIZE, 1)

        # 使用目标网络评估该动作的Q值
        q_next = self.target_net(b_s_).detach()
        q_target_next = q_next.gather(1, best_actions)

        # 计算目标Q值
        q_target = b_r + GAMMA * q_target_next

        loss = self.loss_func(q_eval, q_target)  # 计算loss

        self.optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 反向传播计算梯度，求loss对每个变量的梯度
        self.optimizer.step()  # 由梯度下降法优化参数


dqn = DoubleDQN()  # 修改类名

print("\nCollecting experience...")
time.sleep(2)
for i_episode in range(400):
    s, info = env.reset()  # Gymnasium 返回 (state, info)
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_action(s)  # 选择动作

        # take action
        s_, r, terminated, truncated, info = env.step(a)  # Gymnasium 返回 5 个值
        done = terminated or truncated  # 合并终止条件

        # modify the reward
        x, x_dot, theta, theta_dot = s_  # 下一状态的四个量
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (
            env.theta_threshold_radians - abs(theta)
        ) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)  # 经验存储

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()  # 如果存储达到了一定量，则利用经验进行学习

        if done:
            print("Ep: ", i_episode, "| Ep_r: ", round(ep_r, 2))
            break
        s = s_
env.close()
