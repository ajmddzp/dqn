import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time

# Hyper Parameters
LR = 0.001  # learning rate
GAMMA = 0.99  # reward discount
ENTROPY_BETA = 0.01  # entropy regularization coefficient
MAX_EPISODES = 400
MAX_STEPS = 300
env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make('CartPole-v1')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # Shared feature extraction layers
        self.fc1 = nn.Linear(N_STATES, 128)
        self.fc2 = nn.Linear(128, 128)

        # Actor head - outputs action probabilities
        self.actor = nn.Linear(128, N_ACTIONS)

        # Critic head - outputs state value
        self.critic = nn.Linear(128, 1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Actor: action probabilities (logits)
        action_probs = F.softmax(self.actor(x), dim=-1)

        # Critic: state value
        state_value = self.critic(x)

        return action_probs, state_value


class A2C:
    def __init__(self):
        self.net = ActorCritic()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)

        # For storing episode transitions
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)

        # Get action probabilities and state value from network
        with torch.no_grad():
            action_probs, state_value = self.net(state)

        # Create a categorical distribution over actions
        dist = torch.distributions.Categorical(action_probs)

        # Sample an action
        action = dist.sample()

        # Store for later training
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.item(), log_prob, state_value, entropy

    def store_transition(self, log_prob, value, reward, entropy):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.entropies.append(entropy)

    def learn(self, next_state, done):
        if done:
            next_value = 0
        else:
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            with torch.no_grad():
                _, next_value = self.net(next_state)
            next_value = next_value.item()

        # Calculate returns and advantages
        returns = []
        advantages = []

        # Calculate returns using GAE (Generalized Advantage Estimation)
        R = next_value
        for r in reversed(self.rewards):
            R = r + GAMMA * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)
        values = torch.cat(self.values).squeeze()

        # Calculate advantages
        advantages = returns - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate losses
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0

        for log_prob, value, advantage, entropy, R in zip(
            self.log_probs, values, advantages, self.entropies, returns
        ):
            # Actor loss (policy gradient)
            actor_loss += -log_prob * advantage.detach()

            # Critic loss (value function MSE)
            critic_loss += F.mse_loss(value, R.unsqueeze(0))

            # Entropy regularization
            entropy_loss += -entropy * ENTROPY_BETA

        # Total loss
        total_loss = actor_loss + critic_loss + entropy_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)  # Gradient clipping
        self.optimizer.step()

        # Clear buffers
        self.clear_buffers()

        return actor_loss.item(), critic_loss.item(), entropy_loss.item()

    def clear_buffers(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []


# Initialize A2C agent
a2c = A2C()

print("\nTraining A2C...")
time.sleep(2)

start_time = time.time()
success_count = 0

for i_episode in range(MAX_EPISODES):
    state, info = env.reset()
    ep_reward = 0
    step_count = 0

    while True:
        step_count += 1
        env.render()

        # Choose action
        action, log_prob, value, entropy = a2c.choose_action(state)

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Modify reward (optional, same as DQN version)
        x, x_dot, theta, theta_dot = next_state
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (
            env.theta_threshold_radians - abs(theta)
        ) / env.theta_threshold_radians - 0.5
        modified_reward = r1 + r2

        # Store transition
        a2c.store_transition(log_prob, value, modified_reward, entropy)

        ep_reward += modified_reward

        # Update at the end of episode or when max steps reached
        if done or step_count >= MAX_STEPS:
            if done:
                # Episode ended naturally (pole fell)
                print(
                    f"Episode {i_episode:3d} | Steps: {step_count:3d} | Reward: {ep_reward:6.2f} | Failed"
                )
                success_count = 0
            else:
                # Reached max steps without failing
                success_count += 1
                print(
                    f"Episode {i_episode:3d} | Steps: {step_count:3d} | Reward: {ep_reward:6.2f} | Success #{success_count}"
                )

            # Learn from the episode
            if len(a2c.rewards) > 0:
                actor_loss, critic_loss, entropy_loss = a2c.learn(next_state, done)
                if i_episode % 20 == 0:
                    print(
                        f"  Losses - Actor: {actor_loss:.4f}, Critic: {critic_loss:.4f}, Entropy: {entropy_loss:.4f}"
                    )
            break

        state = next_state

    # Check for convergence (similar to DQN)
    if success_count >= 20:
        print(
            f"\nConverged! Successfully balanced for {success_count} consecutive episodes."
        )
        break

end_time = time.time()
time_elapsed = end_time - start_time
print(f"\nTraining completed in {time_elapsed:.2f} seconds")
env.close()
