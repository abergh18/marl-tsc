from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .base_trainer import BaseGraphTrainer


def get_grad_norm(parameters):
    """Compute the total L2 norm of gradients for a set of parameters."""
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2
    )
    return total_norm.item()

class RunningMeanStd:

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x)

        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = len(x)

        delta = batch_mean - self.mean

        total_count = self.count + batch_count

        new_mean = (
            self.mean
            + delta * batch_count / total_count
        )

        m_a = self.var * self.count
        m_b = batch_var * batch_count

        m2 = (
            m_a
            + m_b
            + delta**2
            * self.count
            * batch_count
            / total_count
        )

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    @property
    def std(self):
        return np.sqrt(self.var)


class GraphCTDETrainer(BaseGraphTrainer):

    def __init__(
        self,
      env,
      policy,
      actor_optimizer,
      critic_optimizer,
      rollout_steps=64,
      gae_lambda=0.95,

    ):
        super().__init__(
            env=env,
            policy=policy,
            #optimizer=optimizer,
            rollout_steps=rollout_steps,
            gae_lambda=gae_lambda,
            #actor_optimizer = actor_optimizer,
            #critic_optimizer = critic_optimizer
        )
        self.value_normalizer = RunningMeanStd()
        self.actor_encoder_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

    def update(
        self,
        rollout_batch,
        advantage_batch
    ):

        advantages = advantage_batch.advantages  # (T, N)
        returns = advantage_batch.returns        # (T, N)

        # -----------------------------
        # Value normalization, using PRE-update stats, then update
        # the normalizer afterward (ordering fix from earlier).
        #
        # CHANGED: returns is now (T, N) instead of (T,). The
        # normalizer itself stays scalar (single running mean/std
        # across all agents and timesteps combined) -- there's no
        # need for a per-agent normalizer, since all agents share
        # the same reward scale/distribution by construction. Only
        # the shape of what gets normalized has changed.
        # -----------------------------
        normalized_returns = (
            returns
            - self.value_normalizer.mean
        ) / (
            self.value_normalizer.std
            + 1e-8
        )

        # Flatten before updating the normalizer's running stats,
        # since RunningMeanStd.update() expects a 1D array.
        self.value_normalizer.update(
            returns.flatten().cpu().numpy()
        )

        print(
            f"ValueNorm mean={self.value_normalizer.mean:.3f} "
            f"std={self.value_normalizer.std:.3f}"
        )

        # CHANGED: advantage normalization now operates across the
        # full (T, N) tensor -- mean/std computed over all agents and
        # timesteps together. This keeps advantages on a comparable
        # scale across agents, which matters since some agents may
        # have systematically higher-variance local rewards than
        # others (e.g. busier intersections).
        advantages = (
            advantages
            - advantages.mean()
        ) / (
            advantages.std()
            + 1e-8
        )

        # -----------------------------
        # Replay observations through current network
        # -----------------------------

        actor_losses = []
        critic_losses = []
        entropy_losses = []

        entropy_coef = 0.01

        for t, graph_obs in enumerate(
            rollout_batch.observations
        ):

            output = self.policy(
                graph_obs
            )

            dist = Categorical(
                logits=output.logits
            )
            entropy = dist.entropy().sum()

            logit_gap = (
                output.logits.max(dim=-1).values
                - output.logits.min(dim=-1).values
            ).mean()

            probs = torch.softmax(
                output.logits,
                dim=-1
            )

            max_prob = probs.max(dim=-1).values.mean()

            print(
                f"entropy={entropy.item():.4f} | "
                f"logit_gap={logit_gap.item():.4f} | "
                f"max_prob={max_prob.item():.4f}"
            )

            actions = rollout_batch.actions[t]  # (N,)

            log_probs = dist.log_prob(
                actions
            )  # (N,)

            #
            # Actor
            #
            # CHANGED: per-agent advantage multiplied against each
            # agent's own log-prob BEFORE summing, instead of summing
            # log-probs first and multiplying by one shared scalar.
            # This is the actual credit-assignment fix -- each agent's
            # gradient now reflects its own advantage.
            #
            actor_loss = -(
                log_probs
                * advantages[t].detach()
            ).sum()

            entropy_loss = dist.entropy().sum()

            print(
                f"value={output.value.mean().item():.3f} "
                f"target={normalized_returns[t].mean().item():.3f}"
            )

            #
            # Critic
            #
            # CHANGED: output.value is now (N, 1) from the per-node
            # CriticHead. Squeeze to (N,) and compare directly against
            # normalized_returns[t], which is also (N,) -- no more
            # .mean() collapse needed since shapes now match natively.
            #
            critic_loss = F.mse_loss(
                output.value.squeeze(-1),
                normalized_returns[t].detach(),
            )

            actor_losses.append(
                actor_loss
            )

            critic_losses.append(
                critic_loss
            )

            entropy_losses.append(
                entropy_loss
            )

        actor_loss = torch.stack(
            actor_losses
        ).mean()

        critic_loss = torch.stack(
            critic_losses
        ).mean()

        entropy_loss = torch.stack(
            entropy_losses
        ).mean()

        total_loss = (
            actor_loss
            + critic_loss
            - entropy_coef * entropy_loss
        )
        self.critic_optimizer.zero_grad()
        self.actor_encoder_optimizer.zero_grad()

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.critic_head.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.encoder.parameters()) + list(self.policy.actor_head.parameters()),
            max_norm=0.5,
        )

        self.critic_optimizer.step()
        self.actor_encoder_optimizer.step()

        '''self.optimizer.zero_grad()

        total_loss.backward()

        # Diagnostic: check gradient norms before clipping
        critic_grad_norm = get_grad_norm(self.policy.critic_head.parameters())
        actor_grad_norm = get_grad_norm(self.policy.actor_head.parameters())
        encoder_grad_norm = get_grad_norm(self.policy.encoder.parameters())

        print(f"GRAD NORMS — critic: {critic_grad_norm:.6f}  actor: {actor_grad_norm:.6f}  encoder: {encoder_grad_norm:.6f}")


        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            max_norm=0.5,
        )
        
        actor_grad = 0.0

        for name, param in self.policy.named_parameters():

            if "actor" in name and param.grad is not None:

                actor_grad += (
                    param.grad.norm().item()
                )

        print(
            f"Actor grad norm={actor_grad:.6f}"
        )'''

        #self.optimizer.step()

        result = {
            "actor_loss": float(
                actor_loss.detach()
            ),
            "critic_loss": float(
                critic_loss.detach()
            ),
            "entropy_loss": float(
                entropy_loss.detach()
            ),
            "total_loss": float(
                total_loss.detach()
            ),
            "rollout_length": len(
                rollout_batch.observations
            ),
            "mean_training_reward": float(
                rollout_batch.rewards.mean()
            ),
        }

        print("UPDATE RETURN:", result)

        return result
