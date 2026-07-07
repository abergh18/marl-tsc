"""
true_mappo_trainer.py

True MAPPO trainer using a graph-based actor and centralised critic.

What distinguishes this from graph_mappo_trainer.py (soon to be graph_ippo_trainer.py)
---------------------------------------------------------------------------------------
- Critic is centralised: a single V(s_t) computed from the full global state,
  shared across all agents. This is the defining feature of MAPPO.
- Actor path is unchanged: per-agent logits from local graph embeddings.
- Value loss regresses the scalar V(s_t) against mean return across agents.
- collect_batch() is overridden here to use global_value for GAE bootstrap,
  keeping BaseGraphTrainer untouched.

Reference: Yu et al. (2021) "The Surprising Effectiveness of PPO in
Cooperative Multi-Agent Games"
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .base_trainer import BaseGraphTrainer
from .graph_rollout import GraphRollout
from .advantage_estimator import AdvantageEstimator


class TrueMAPPOTrainer(BaseGraphTrainer):
    """
    Graph-based true MAPPO trainer.

    Actor  : per-agent, consumes local graph embeddings (GAT encoder output)
    Critic : centralised MLP, consumes global_state = flatten(all agent obs)
    Update : PPO clipping + entropy regularisation, multi-epoch
    """

    def __init__(
        self,
        env,
        policy,
        optimizer,
        rollout_steps=64,
        gae_lambda=0.95,
        gamma=0.99,
        clip_ratio=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=3,
    ):
        super().__init__(
            env=env,
            policy=policy,
            rollout_steps=rollout_steps,
            gae_lambda=gae_lambda,
        )

        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs

    # ── Override collect_batch to use centralised global_value ───────────────

    def collect_batch(self):
        """
        Identical to BaseGraphTrainer.collect_batch except bootstrap_value
        comes from policy.global_value (centralised critic) rather than
        policy.value (per-agent critic heads).
        """
        transitions = self.runner.collect_rollout(
            num_steps=self.rollout_steps,
        )

        rollout_batch = GraphRollout.from_transitions(
            transitions,
            self.env.agent_ids,
        )

        last_observation = self.runner.last_observation

        with torch.no_grad():
            bootstrap_output = self.policy(last_observation)

            # Centralised critic produces a single scalar V(s).
            # Expand to (num_agents,) so AdvantageEstimator.compute_gae
            # receives the shape it expects.
            num_agents = len(self.env.agent_ids)
            bootstrap_value = bootstrap_output.global_value.expand(num_agents)

        advantage_batch = AdvantageEstimator.compute_gae(
            rollout_batch,
            bootstrap_value=bootstrap_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        return rollout_batch, advantage_batch

    # ── PPO update ────────────────────────────────────────────────────────────

    def update(self, rollout_batch, advantage_batch):
        """
        PPO update step.

        Parameters
        ----------
        rollout_batch : RolloutBatch
            Collected transitions. rollout_batch.observations are
            GraphObservation objects containing both .graph (for actor)
            and .global_state (for centralised critic).
        advantage_batch : AdvantageBatch
            Computed advantages and returns.

        Returns
        -------
        dict
            Training statistics.
        """
        advantages = advantage_batch.advantages    # (T, num_agents)
        returns = advantage_batch.returns          # (T, num_agents)

        # Normalise across all agents and timesteps
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses = []
        value_losses = []
        entropy_losses = []
        policy_clip_fracs = []

        for epoch in range(self.update_epochs):

            for t, graph_obs in enumerate(rollout_batch.observations):

                output = self.policy(graph_obs)

                actions = rollout_batch.actions[t].to(output.logits.device)
                old_log_probs = rollout_batch.log_probs[t]

                # ── Actor ─────────────────────────────────────────────────────
                dist = Categorical(logits=output.logits)
                new_log_probs = dist.log_prob(actions)   # (num_agents,)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs)
                adv_t = advantages[t]                    # (num_agents,)

                surr1 = ratio * adv_t
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio,
                ) * adv_t

                actor_loss = -torch.min(surr1, surr2).mean()

                # ── Centralised critic ────────────────────────────────────────
                # output.global_value : (1,)  scalar V(s_t)
                # returns[t]          : (num_agents,) per-agent returns
                #
                # Regress V(s_t) against mean return across agents.
                # Valid for homogeneous agents sharing the same reward scale,
                # which holds for the 4x4 SUMO grid.
                target_return = returns[t].mean()        # scalar
                value_loss = F.mse_loss(
                    output.global_value.squeeze(),       # scalar
                    target_return,
                )

                # ── Total loss ────────────────────────────────────────────────
                total_loss = (
                    actor_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > self.clip_ratio)
                        .float()
                        .mean()
                    )

                actor_losses.append(actor_loss.detach())
                value_losses.append(value_loss.detach())
                entropy_losses.append(entropy.detach())
                policy_clip_fracs.append(clip_frac.detach())

        return {
            "actor_loss": float(torch.stack(actor_losses).mean()),
            "critic_loss": float(torch.stack(value_losses).mean()),
            "entropy_loss": float(torch.stack(entropy_losses).mean()),
            "policy_clip_fraction": float(torch.stack(policy_clip_fracs).mean()),
            "mean_training_reward": float(rollout_batch.rewards.mean()),
            "rollout_length": len(rollout_batch.observations),
            "update_epochs": self.update_epochs,
        }
