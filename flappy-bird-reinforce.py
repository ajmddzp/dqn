import flappy_bird_gymnasium
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time
import matplotlib as plt

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.001                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 10000
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
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

        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs = self.net(state)  # 获取动作概率分布

        # 根据概率分布采样动作
        action_dist = torch.distributions.Categorical(action_probs).to(device)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob

    def store_transition(self, state, action, log_prob, reward):
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_log_probs.append(log_prob)
        self.episode_rewards.append(reward)

    def learn(self):
        # 计算每个时间步的折扣回报
        returns = []
        R = 0
        for r in reversed(self.episode_rewards):
            R = r + GAMMA * R
            returns.insert(0, R)

        # 标准化回报（减少方差）
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 标准化

        # 计算策略梯度损失
        policy_loss = []
        for log_prob, R in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * R)  # 负号因为我们要最大化回报

        # 梯度更新
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        # 清空当前轨迹
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []


# print(N_ACTIONS)
# print(N_STATES)
print('\nCollecting experience...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn = DQN(device)

x = []
y = []
time.sleep(2)
for i_episode in range(4000):
    s, _ = env.reset()
    ep_r = 0
    episode_length = 0

    while True:
        # Next action:
        # (feed the observation to your agent here)
        env.render()

        action, log_prob = dqn.choose_action(s)
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

        dqn.store_transition(s, action, log_prob, r)

        # Checking if the player is still alive
        if terminated:
            dqn.learn()
            x.append(reward)
            y.append(i_episode)
            print(f'Ep: {i_episode:3d} | '
                  f'Ep_r: {round(ep_r, 2):6.2f} | '
                  f'Length: {episode_length:3d}')
            break
        s = s_
    plt.plot(x, y, color="green")
    plt.show()
env.close()
