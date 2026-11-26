import torch
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn as nn

from model import Actor, Critic

class PPO:
    def __init__(self,
            obs_dim,
            act_dim,
            hidden_dim,
            num_hidden,
            gamma,
            lmbda,
            eps,
            ent_coef,
            value_coef,
            epochs,
            policy_lr,
            value_lr,
            minibatch,
            discrete=True,
            decay_actor_lr=False,
            decay_critic_lr=False,
            decay_ent=False,
            kl_target=None):

        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.epochs = epochs
        self.minibatch = minibatch
        self.discrete = discrete
        self.decay_actor_lr = decay_actor_lr
        self.decay_critic_lr = decay_critic_lr
        self.decay_ent = decay_ent
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.kl_target = kl_target

        self.obs_dim = obs_dim
        self.actor = Actor(obs_dim, act_dim, hidden_dim, num_hidden)
        self.critic = Critic(obs_dim, 1, hidden_dim, num_hidden)

        self.policy_optim = optim.Adam(self.actor.parameters(), lr=self.policy_lr, eps=1e-5)
        self.value_optim = optim.Adam(self.critic.parameters(), lr=self.value_lr, eps=1e-5)

    def gae(self, buffer):

        with torch.inference_mode():
            next_values = self.critic.get_value(buffer.next_states)

        bootstrap_mask = (buffer.done <= 0)
        TD = buffer.rewards + self.gamma * next_values * bootstrap_mask - buffer.values

        # Advantage At = TD_t + gamma*lambda*At+1
        advantages = torch.zeros_like(buffer.rewards)
        At_1 = 0
        for t in reversed(range(buffer.steps)):
            cont = (buffer.done[t] == 0)
            At = TD[t] + self.gamma * self.lmbda * cont * At_1

            advantages[t] = At
            At_1 = At

        returns = advantages + buffer.values
        return advantages, returns

    def update(self, advantage, returns, buffer):
        rollout_steps, obs_dim, n_envs = buffer.steps, self.obs_dim, buffer.n_envs

        # 1: // Batch Processing //
        states = buffer.states.reshape(rollout_steps * n_envs, obs_dim)
        actions = buffer.actions.reshape(rollout_steps * n_envs)
        old_log_probs = buffer.log_probs.reshape(rollout_steps * n_envs)
        advantages = advantage.reshape(rollout_steps * n_envs)
        returns = returns.reshape(rollout_steps * n_envs)

        # redundant but maybe needed for tracking
        values = buffer.values.reshape(rollout_steps * n_envs)
        rewards = buffer.rewards.reshape(rollout_steps * n_envs)

        # 2: // Minibatching //
        indices = torch.randperm(self.minibatch)

        # 3: // Training Loop //
        for epoch in range(self.epochs):


            for start in range(0, rollout_steps, self.minibatch):
                end = start + self.minibatch
                mb_idx = indices[start:end]

                batch_states = states[mb_idx]
                batch_actions = actions[mb_idx]
                batch_old_log = old_log_probs[mb_idx]
                batch_adv = advantage[mb_idx]
                batch_returns = returns[mb_idx]

                # // New Policy and Value Outputs //
                out = self.actor.policy(batch_states)
                dist = Categorical(logits=out)

                new_log = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                values = self.critic.value(batch_states).squeeze(-1)

                # // Calculate Losses //
                r = torch.exp(new_log - batch_old_log)
                policy_loss = -torch.min(r * batch_adv, torch.clamp(r, 1 - self.eps, 1 + self.eps) * batch_adv).mean()
                value_loss = nn.functional.mse_loss(values, batch_returns)

                total_loss = policy_loss + value_loss * self.value_coef - entropy * self.ent_coef

                # // Backpropagate //
                self.policy_optim.zero_grad()
                self.value_optim.zero_grad()

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)

                self.policy_optim.step()
                self.value_optim.step()
