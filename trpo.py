# TRPO:Actor-Critic强化版，基本思想与AC一致，使用了不同的迭代更新方法。
'''
KL 散度（Kullback-Leibler Divergence）精确地衡量了当我们使用一个近似概率分布 Q 来建模或描述一个真实概率分布 P 时，所引入的信息损失。
简而言之，它量化了“近似”与“真实”之间的差距。KL 散度值越小，意味着分布 Q 对分布 P 的拟合程度越高。
kl散度用来计算策略之间的差别，如果KL散度小于MAX_KL，则策略更新。
如果KL散度大于MAX_KL，则策略更新停止，防止步长过大从而导致扯到蛋。
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time
from torch.distributions import Categorical

# Hyper Parameters
BATCH_SIZE = 4000  # TRPO通常使用更大的批量大小
LR = 0.01  # 学习率（主要用于价值网络）
GAMMA = 0.99  # 奖励折扣
LAMBDA = 0.95  # GAE参数
MAX_KL = 0.01  # 最大KL散度
DAMPING = 0.1  # 阻尼系数
env = gym.make('CartPole-v1', render_mode='human')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(
    env.action_space.sample(), int) else env.action_space.sample().shape


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_logits = self.out(x)
        return F.softmax(action_logits, dim=-1)


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        state_value = self.out(x)
        return state_value


class TRPO:
    def __init__(self):
        self.policy_net = PolicyNet()
        self.value_net = ValueNet()
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=LR)

    def choose_action(self, x):  # 与AC一致，得出概率后采样
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        probs = self.policy_net(x)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def compute_advantages(self, rewards, values, dones, next_value):
        advantages = np.zeros(len(rewards))
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_value
            else:
                next_value = values[t + 1] * (1 - dones[t])
                # dons：是否为终止态，开关

            delta = rewards[t] + GAMMA * next_value - values[t]
            advantages[t] = delta + GAMMA * LAMBDA * \
                (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        return advantages

    def update_value_net(self, states, returns):
        states = torch.FloatTensor(states)
        returns = torch.FloatTensor(returns).unsqueeze(1)

        for _ in range(10):  # 多次更新价值网络
            # 与AC一致，利用均方差损失函数来更新critic网络
            values = self.value_net(states)
            value_loss = F.mse_loss(values, returns)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

    def update_policy(self, states, actions, log_probs_old, advantages):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs_old)
        advantages = torch.FloatTensor(advantages)

        # 计算旧策略的概率
        with torch.no_grad():
            old_probs = self.policy_net(states)
            old_log_probs = torch.log(old_probs.gather(
                1, actions.unsqueeze(1))).squeeze()

        # 计算损失函数的梯度
        def get_loss(volatile=False):
            if volatile:
                with torch.no_grad():
                    probs = self.policy_net(states)
                    log_probs = torch.log(probs.gather(
                        1, actions.unsqueeze(1))).squeeze()
            else:
                probs = self.policy_net(states)
                log_probs = torch.log(probs.gather(
                    1, actions.unsqueeze(1))).squeeze()

            ratio = torch.exp(log_probs - log_probs_old)
            loss = (ratio * advantages).mean()
            return loss

        # 计算KL散度
        def get_kl():
            with torch.no_grad():
                probs = self.policy_net(states)
                log_probs = torch.log(probs)
                old_log_probs_data = torch.log(old_probs)
                kl = (old_probs * (old_log_probs_data - log_probs)
                      ).sum(dim=1).mean()
            return kl

        # 计算梯度
        loss = get_loss()
        grads = torch.autograd.grad(loss, self.policy_net.parameters())
        flat_grad = torch.cat([grad.view(-1) for grad in grads])

        # 计算Fisher-vector乘积
        def fisher_vector_product(v):
            kl = get_kl()
            grads = torch.autograd.grad(
                kl, self.policy_net.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * v).sum()
            grads = torch.autograd.grad(kl_v, self.policy_net.parameters())
            flat_grad_grad_kl = torch.cat(
                [grad.contiguous().view(-1) for grad in grads])

            return flat_grad_grad_kl + v * DAMPING

        # 共轭梯度法
        step_dir = self.conjugate_gradient(
            fisher_vector_product, flat_grad.data, nsteps=10)

        # 计算自然梯度
        shs = 0.5 * (step_dir * fisher_vector_product(step_dir)
                     ).sum(0, keepdim=True)
        lm = torch.sqrt(shs / MAX_KL)
        full_step = step_dir / lm[0]

        # 更新策略网络参数
        old_params = self.get_flat_params()
        self.set_flat_params(old_params + full_step)

        # 检查KL散度约束
        if get_kl() > MAX_KL * 1.5:
            self.set_flat_params(old_params)

    def conjugate_gradient(self, f_Ax, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)

        for i in range(nsteps):
            Ap = f_Ax(p)
            alpha = rdotr / torch.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            p = r + (new_rdotr / rdotr) * p
            rdotr = new_rdotr

        return x

    def get_flat_params(self):
        params = []
        for param in self.policy_net.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)

    def set_flat_params(self, flat_params):
        offset = 0
        for param in self.policy_net.parameters():
            numel = param.numel()
            param.data.copy_(
                flat_params[offset:offset+numel].view(param.size()))
            offset += numel


trpo = TRPO()

print('\nTraining with TRPO...')
time.sleep(2)

for i_episode in range(400):
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

    s, info = env.reset()
    ep_r = 0

    while True:
        env.render()
        a, log_prob = trpo.choose_action(s)  # 进行一次行动

        # take action
        s_, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # modify the reward (same as before)
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / \
            env.theta_threshold_radians - 0.5
        r = r1 + r2

        # store transition
        states.append(s)
        actions.append(a)
        log_probs.append(log_prob.item())
        rewards.append(r)
        dones.append(done)

        # value estimate
        with torch.no_grad():
            state_tensor = torch.FloatTensor(s).unsqueeze(0)
            value = trpo.value_net(state_tensor).item()
            values.append(value)

        ep_r += r

        if done:
            # 计算最后一个状态的value
            with torch.no_grad():
                if terminated:
                    next_value = 0.0
                else:
                    next_value = trpo.value_net(
                        torch.FloatTensor(s_).unsqueeze(0)).item()

            # 计算优势函数
            advantages = trpo.compute_advantages(
                rewards, values, dones, next_value)

            # 计算回报
            returns = advantages + values

            # 更新价值网络
            trpo.update_value_net(states, returns)

            # 更新策略网络
            trpo.update_policy(states, actions, log_probs, advantages)

            print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2))
            break

        s = s_

env.close()
