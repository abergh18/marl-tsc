"""
true_mappo_trainer.py

True MAPPO trainer using a graph-based actor and centralised critic.
Supports optional zero-sum gifting via separate gifting PPO loss.

What changed from original
---------------------------
- Gifting loss computed and logged separately from traffic loss when
  rollout_batch.gifting_log_probs is not None.
- Gifting clip fraction tracked independently.
- Gifting stats aggregated from rollout_batch.gifting_stats and added
  to returned stats dict.
- Non-gifting runs are completely unaffected — all gifting logic is
  gated on rollout_batch.gifting_log_probs is not None.
- gifting_entropy_coef added as separate hyperparameter.
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

    Gifting support
    ---------------
    When the rollout batch contains gifting_log_probs (i.e. collected by
    GiftingGraphRunner), a separate gifting PPO loss is computed and added
    to the total loss. The gifting head is trained jointly with the actor
    and critic but its gradient signal is tracked independently.
    Non-gifting runs are completely unaffected.
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
        gifting_entropy_coef=0.01,
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
        self.gifting_entropy_coef = gifting_entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs

    def collect_batch(self):
        """
        Override to use global_value for GAE bootstrap.
        Identical to base except bootstrap uses global_value.
        """
        transitions = self.runner.collect_rollout(
            num_steps=self.rollout_steps,
        )
        #print(transitions[0].reward_dict.keys())
        #print(self.env.agent_ids)
        rollout_batch = GraphRollout.from_transitions(
            transitions,
            self.env.agent_ids,
        )

        last_observation = self.runner.last_observation

        with torch.no_grad():
            bootstrap_output = self.policy(last_observation)
            num_agents = len(self.env.agent_ids)
            bootstrap_value = bootstrap_output.global_value.expand(num_agents)

        advantage_batch = AdvantageEstimator.compute_gae(
            rollout_batch,
            bootstrap_value=bootstrap_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        return rollout_batch, advantage_batch

    def update(self, rollout_batch, advantage_batch):
        """
        PPO update step with optional separate gifting loss.

        Parameters
        ----------
        rollout_batch : RolloutBatch
        advantage_batch : AdvantageBatch

        Returns
        -------
        dict
            Training statistics including gifting metrics when applicable.
        """
        advantages = advantage_batch.advantages    # (T, num_agents)
        returns = advantage_batch.returns          # (T, num_agents)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        is_gifting = rollout_batch.gifting_log_probs is not None

        actor_losses = []
        value_losses = []
        entropy_losses = []
        policy_clip_fracs = []

        gifting_losses = []
        gifting_entropy_losses = []
        gifting_clip_fracs = []

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns_mean = returns.mean()
        returns_std  = returns.std() + 1e-8
        returns_norm = (returns - returns_mean) / returns_std

        for epoch in range(self.update_epochs):

            for t, graph_obs in enumerate(rollout_batch.observations):

                output = self.policy(graph_obs)

                actions = rollout_batch.actions[t].to(output.logits.device)
                old_log_probs = rollout_batch.log_probs[t]

                # ── Traffic actor loss ────────────────────────────────────────
                dist = Categorical(logits=output.logits)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs)
                adv_t = advantages[t]

                surr1 = ratio * adv_t
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio,
                ) * adv_t

                actor_loss = -torch.min(surr1, surr2).mean()

                # ── Centralised critic loss ───────────────────────────────────
                target_return = returns_norm[t].mean()
                value_loss = F.mse_loss(
                    output.global_value.squeeze(),
                    target_return,
                )

                # ── Gifting loss (when applicable) ────────────────────────────
                gifting_loss = torch.tensor(0.0, device=output.logits.device)
                gifting_entropy = torch.tensor(0.0, device=output.logits.device)

                if is_gifting:
                    gifting_actions_t = rollout_batch.gifting_actions[t].to(
                        output.logits.device
                    )
                    old_gifting_log_probs = rollout_batch.gifting_log_probs[t]

                    gifting_logits = self.policy.branches[1](
                        output.encoder_output
                        if isinstance(output.encoder_output, torch.Tensor)
                        else output.encoder_output.node_embeddings
                    )
                    dist_g = Categorical(logits=gifting_logits)
                    new_gifting_log_probs = dist_g.log_prob(gifting_actions_t)
                    gifting_ent = dist_g.entropy().mean()

                    gifting_ratio = torch.exp(
                        new_gifting_log_probs - old_gifting_log_probs
                    )

                    g_surr1 = gifting_ratio * adv_t
                    g_surr2 = torch.clamp(
                        gifting_ratio,
                        1.0 - self.clip_ratio,
                        1.0 + self.clip_ratio,
                    ) * adv_t

                    gifting_loss = -torch.min(g_surr1, g_surr2).mean()
                    gifting_entropy = gifting_ent

                    with torch.no_grad():
                        g_clip_frac = (
                            (torch.abs(gifting_ratio - 1.0) > self.clip_ratio)
                            .float()
                            .mean()
                        )
                    gifting_losses.append(gifting_loss.detach())
                    gifting_entropy_losses.append(gifting_entropy.detach())
                    gifting_clip_fracs.append(g_clip_frac.detach())

                # ── Total loss ────────────────────────────────────────────────
                total_loss = (
                    actor_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                    + gifting_loss
                    - self.gifting_entropy_coef * gifting_entropy
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

        # ── Aggregate gifting stats from rollout ──────────────────────────────
        mean_gift_fraction = 0.0
        gift_rate = 0.0
        mean_gift_amount = 0.0

        if is_gifting and rollout_batch.gifting_stats:
            mean_gift_fraction = float(
                sum(s["mean_gift_fraction"] for s in rollout_batch.gifting_stats)
                / len(rollout_batch.gifting_stats)
            )
            gift_rate = float(
                sum(s["gift_rate"] for s in rollout_batch.gifting_stats)
                / len(rollout_batch.gifting_stats)
            )
            mean_gift_amount = float(
                sum(s["mean_gift_amount"] for s in rollout_batch.gifting_stats)
                / len(rollout_batch.gifting_stats)
            )

        stats = {
            "actor_loss": float(torch.stack(actor_losses).mean()),
            "critic_loss": float(torch.stack(value_losses).mean()),
            "entropy_loss": float(torch.stack(entropy_losses).mean()),
            "total_loss": float(torch.stack(actor_losses).mean() + 
                          self.value_coef * torch.stack(value_losses).mean()),
            "policy_clip_fraction": float(torch.stack(policy_clip_fracs).mean()),
            "mean_training_reward": float(rollout_batch.rewards.mean()),
            "rollout_length": len(rollout_batch.observations),
            "update_epochs": self.update_epochs,
        }

        if is_gifting:
            stats.update({
                "gifting_loss": float(torch.stack(gifting_losses).mean()),
                "gifting_entropy_loss": float(torch.stack(gifting_entropy_losses).mean()),
                "gifting_clip_fraction": float(torch.stack(gifting_clip_fracs).mean()),
                "mean_gift_fraction": mean_gift_fraction,
                "gift_rate": gift_rate,
                "mean_gift_amount": mean_gift_amount,
            })

        return stats