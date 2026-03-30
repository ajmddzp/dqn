import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model as G

# Hyper Parameters
BATCH_SIZE = 32
LR = 1e-4
GAMMA = 0.99
TARGET_REPLACE_ITER = 500
MEMORY_CAPACITY = 20000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 80000
MAX_EPISODES = 1200
MAX_STEPS_PER_EPISODE = 3000
RENDER_EVERY_EPISODES = 100
RENDER_FPS = 30

N_ACTIONS = 4
N_STATES = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_STATES, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.out = nn.Linear(256, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


class DQN:
    def __init__(self):
        self.eval_net = Net().to(device)
        self.target_net = Net().to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.action_step_counter = 0
        # [s(16), a(1), r(1), done(1), s'(16)]
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3), dtype=np.float32)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.SmoothL1Loss()
        self.epsilon = EPSILON_START

    @staticmethod
    def _to_numpy_state(s):
        if isinstance(s, tuple):
            s = s[0]
        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().numpy()
        return np.asarray(s, dtype=np.float32).reshape(-1)

    def choose_action(self, x):
        x = self._to_numpy_state(x)
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)

        self.action_step_counter += 1
        ratio = min(1.0, self.action_step_counter / EPSILON_DECAY_STEPS)
        self.epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * ratio

        if np.random.uniform() < self.epsilon:
            return int(np.random.randint(0, N_ACTIONS))
        with torch.no_grad():
            q_values = self.eval_net(x)
            return int(q_values.argmax(dim=1).item())

    def store_transition(self, s, a, r, s_, done):
        s = self._to_numpy_state(s)
        s_ = self._to_numpy_state(s_)
        transition = np.hstack((s, [a, r, float(bool(done))], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        current_size = min(self.memory_counter, MEMORY_CAPACITY)
        if current_size < BATCH_SIZE:
            return None

        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(current_size, BATCH_SIZE, replace=False)
        batch = self.memory[sample_index, :]

        b_s = torch.FloatTensor(batch[:, :N_STATES]).to(device)
        b_a = torch.LongTensor(batch[:, N_STATES : N_STATES + 1].astype(int)).to(device)
        b_r = torch.FloatTensor(batch[:, N_STATES + 1 : N_STATES + 2]).to(device)
        b_done = torch.FloatTensor(batch[:, N_STATES + 2 : N_STATES + 3]).to(device)
        b_s_ = torch.FloatTensor(batch[:, -N_STATES:]).to(device)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        with torch.no_grad():
            q_next = self.target_net(b_s_).max(1)[0].view(BATCH_SIZE, 1)
            q_target = b_r + (1.0 - b_done) * GAMMA * q_next
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())


def train(render=True, render_every=RENDER_EVERY_EPISODES, render_fps=RENDER_FPS):
    # Start with render off, and only enable it on scheduled episodes.
    G.set_render(False, fps=render_fps)
    dqn = DQN()
    print("\nCollecting experience...")
    time.sleep(1)

    start_time = time.time()
    last_render_state = False
    for i_episode in range(MAX_EPISODES):
        if render:
            if render_every <= 1:
                should_render = True
            else:
                should_render = (i_episode + 1) % render_every == 0
        else:
            should_render = False

        if should_render != last_render_state:
            G.set_render(should_render, fps=render_fps)
            last_render_state = should_render
        s = G.reset()
        ep_r = 0.0
        steps = 0
        last_loss = None

        while True:
            a = dqn.choose_action(s)
            step_out = G.RL_step(a)
            if len(step_out) == 4:
                s_, r, done, current_max = step_out
            elif len(step_out) == 3:
                s_, r, done = step_out
                current_max = int(max(max(row) for row in G.gameMap))
            else:
                raise ValueError(f"Unexpected RL_step return length: {len(step_out)}")
            dqn.store_transition(s, a, r, s_, done)
            loss = dqn.learn()
            if loss is not None:
                last_loss = loss

            ep_r += float(r)
            s = s_
            steps += 1

            if done or steps >= MAX_STEPS_PER_EPISODE:
                loss_text = "None" if last_loss is None else f"{last_loss:.4f}"
                print(
                    f"Ep: {i_episode:4d} | Steps: {steps:4d} | "
                    f"Ep_r: {ep_r:8.2f} | Eps: {dqn.epsilon:.3f} | Loss: {loss_text} | "
                    f"Ep_Max: {current_max}"
                )
                break

    if last_render_state:
        G.set_render(False, fps=render_fps)
    print("time:", time.time() - start_time)


if __name__ == "__main__":
    train()
