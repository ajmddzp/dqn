import flappy_bird_gymnasium
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time
import matplotlib.pyplot as plt

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.001                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 10000
env = gymnasium.make("FlappyBird-v0", render_mode='human', use_lidar=False)
# env = gymnasium.make("FlappyBird-v0", use_lidar=False)
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]  # 状态变量数
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(
), int) else env.action_space.sample().shape     # to confirm the shape


class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(N_STATES, 128)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization,服从正态分布N(0,1)
        self.fc2 = nn.Linear(128, 50)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):

    def __init__(self, device):
        self.device = device
        self.eval_net = Net(device).to(device)  # 移动到GPU
        self.target_net = Net(device).to(device)  # 移动到GPU

        self.learn_step_counter = 0  # 学习步数初始化       # for target updating
        self.memory_counter = 0  # 经验存储位置计数器            # for storing memory
        # initialize memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=LR)  # 使用Adam优化器
        self.loss_func = nn.MSELoss()  # 均方差损失函数

    def choose_action(self, x, epoch):

        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(self.device)  # 移动到GPU
        # input only one sample
        if np.random.uniform() < EPSILON or epoch > 100:   # greedy
            actions_value = self.eval_net.forward(x)  # 前向传播
            action = torch.max(actions_value, 1)[
                1].cpu().data.numpy()  # 移回CPU处理
            action = action[0] if ENV_A_SHAPE == 0 else np.array(
                action).reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else np.array(
                action).reshape(ENV_A_SHAPE)
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
                self.eval_net.state_dict())  # 更新目标网络，从eval中拷贝参数
        self.learn_step_counter += 1  # 计数器

        # sample batch transitions
        sample_index = np.random.choice(
            MEMORY_CAPACITY, BATCH_SIZE)  # 从2000个经验中选取32个索引
        b_memory = self.memory[sample_index, :]  # 从内存中取得经验

        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(
            self.device)  # 数据转换类型
        # 采样的经验，状态
        b_a = torch.LongTensor(
            b_memory[:, N_STATES:N_STATES+1].astype(int)).to(self.device)
        # 采样的经验，动作
        b_r = torch.FloatTensor(
            b_memory[:, N_STATES+1:N_STATES+2]).to(self.device)
        # 采样的经验，奖励
        b_s_ = torch.FloatTensor(
            b_memory[:, -N_STATES:]).to(self.device)  # 负索引，表示从后往前
        # 采样的经验，下一个状态

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        # 前向传播，并获得索引为[1],[b_a]的值
        # detach from graph, don't backpropagate
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * \
            q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)  # 计算loss

        self.optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 反向传播计算梯度，求loss对每个变量的梯度
        self.optimizer.step()  # 由梯度下降法优化参数


# print(N_ACTIONS)
# print(N_STATES)
print('\nCollecting experience...')
print(N_STATES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn = DQN(device)

x = []
y = []
time.sleep(2)
for i_episode in range(4000):
    s, _ = env.reset()
    ep_r = 0
    while True:
        # Next action:
        # (feed the observation to your agent here)
        # env.render()

        action = dqn.choose_action(s, i_episode)
    # Processing:
        s_, reward, terminated, _, info = env.step(action)
        # ep_r += reward
        last_pop_top = s_[1]
        last_pop_bot = s_[2]
        last_pop_pos = (last_pop_top+last_pop_bot)/2

        pip_x = s_[0]

        # next_pop_top = s_[4]
        # next_pop_bot = s_[5]
        # next_pop_pos = (next_pop_top+next_pop_bot)/2

        pos_actor = s_[9]
        vec_actor = s_[10]
        # print(pip_x)
        # print(abs(pos_actor-last_pop_pos))
        # print(abs(pos_actor-next_pop_pos))
        # print(last_pop_top, last_pop_bot)
        # break

        r = reward*2 - abs(pos_actor-last_pop_pos)
        ep_r += r

        dqn.store_transition(s, action, r, s_)

        if dqn.memory_counter > MEMORY_CAPACITY:
            # print("learn!!!")
            # 如果存储达到了一定量，则利用经验进行学习
            dqn.learn()
    # Checking if the player is still alive
        if terminated:
            x.append(ep_r)
            y.append(i_episode)
            print('Ep: ', i_episode,
                  '| Ep_r: ', round(ep_r, 2))
            break
        s = s_

plt.plot(y, x, color="green")
plt.show()
env.close()
