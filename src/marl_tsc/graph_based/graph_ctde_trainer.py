from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import Counter

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
      entropy_coef = 5e-2,
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
        self.entropy_coef = entropy_coef

    def update(
        self,
        rollout_batch,
        advantage_batch,
    ):

        advantages = advantage_batch.advantages
        returns = advantage_batch.returns

        #
        # -------------------------------------------------
        # Value normalisation
        # -------------------------------------------------
        #

        # Uncomment if using value normalisation.
        #
        # normalized_returns = (
        #     returns
        #     - self.value_normalizer.mean
        # ) / (
        #     self.value_normalizer.std
        #     + 1e-8
        # )

        normalized_returns = returns

        self.value_normalizer.update(
            returns.flatten().cpu().numpy()
        )

        print(
            f"ValueNorm "
            f"mean={self.value_normalizer.mean:.3f} "
            f"std={self.value_normalizer.std:.3f}"
        )

        #
        # -------------------------------------------------
        # Advantage normalisation
        # -------------------------------------------------
        #

        advantages = (
            advantages
            - advantages.mean()
        ) / (
            advantages.std()
            + 1e-8
        )

        print(
            "ADV:",
            f"mean={advantages.mean().item():.4f}",
            f"std={advantages.std().item():.4f}",
            f"min={advantages.min().item():.4f}",
            f"max={advantages.max().item():.4f}",
        )

        #
        # -------------------------------------------------
        # Storage
        # -------------------------------------------------
        #

        actor_losses = []
        critic_losses = []
        entropy_losses = []

        #
        # Diagnostics
        #

        explained_variances = []

        value_means = []
        value_stds = []

        return_means = []
        return_stds = []

        td_error_means = []
        td_error_maxes = []

        adv_scale = 10.0

        #
        # -------------------------------------------------
        # Replay rollout
        # -------------------------------------------------
        #

        for t, graph_obs in enumerate(
            rollout_batch.observations
        ):

            output = self.policy(
                graph_obs
            )

            if t == 0:

                emb = (
                    output
                    .encoder_output
                    .node_embeddings
                )

                print(
                    f"emb_mean={emb.mean().item():.4f} "
                    f"emb_std={emb.std().item():.4f}"
                )

                print(
                    "first_node_emb:",
                    emb[0][:5]
                    .detach()
                    .cpu()
                    .numpy(),
                )

            dist = Categorical(
                logits=output.logits
            )

            entropy = dist.entropy().sum()

            probs = torch.softmax(
                output.logits,
                dim=-1,
            )

            logit_gap = (
                output.logits.max(dim=-1).values
                - output.logits.min(dim=-1).values
            ).mean()

            max_prob = (
                probs.max(dim=-1)
                .values
                .mean()
            )

            print(
                f"entropy={entropy.item():.4f} | "
                f"logit_gap={logit_gap.item():.4f} | "
                f"max_prob={max_prob.item():.4f}"
            )

            actions = rollout_batch.actions[t]

            if t == 0:
                print(
                    Counter(
                        rollout_batch.actions
                        .flatten()
                        .tolist()
                    )
                )

            log_probs = dist.log_prob(
                actions
            )

            actor_loss = -(
                log_probs
                * advantages[t].detach()
                * adv_scale
            ).sum()

            entropy_loss = entropy

            values = (
                output.value
                .squeeze(-1)
            )

            targets = (
                normalized_returns[t]
                .detach()
            )

            critic_loss = F.mse_loss(
                values,
                targets,
            )

            #
            # ------------------------
            # Critic diagnostics
            # ------------------------
            #

            with torch.no_grad():

                td_error = (
                    targets
                    - values
                )

                value_means.append(
                    values.mean().item()
                )

                value_stds.append(
                    values.std().item()
                )

                return_means.append(
                    targets.mean().item()
                )

                return_stds.append(
                    targets.std().item()
                )

                td_error_means.append(
                    td_error.abs().mean().item()
                )

                td_error_maxes.append(
                    td_error.abs().max().item()
                )

                var_returns = torch.var(
                    targets
                )

                if var_returns > 1e-8:

                    ev = (
                        1.0
                        - torch.var(
                            targets - values
                        )
                        / var_returns
                    )

                    explained_variances.append(
                        ev.item()
                    )

            print(
                f"value={values.mean().item():.3f} "
                f"target={targets.mean().item():.3f}"
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

        #
        # -------------------------------------------------
        # Critic updates
        # -------------------------------------------------
        #

        critic_update_epochs = 5

        for epoch in range(
            critic_update_epochs
        ):

            critic_losses_epoch = []

            for t, graph_obs in enumerate(
                rollout_batch.observations
            ):

                output = self.policy(
                    graph_obs
                )

                values = output.value.squeeze(-1)

                targets = (
                    normalized_returns[t]
                    .detach()
                )

                critic_loss_t = F.mse_loss(
                    values,
                    targets,
                )

                critic_losses_epoch.append(
                    critic_loss_t
                )

            critic_loss_epoch = torch.stack(
                critic_losses_epoch
            ).mean()

            self.critic_optimizer.zero_grad()

            critic_loss_epoch.backward()

            critic_grad = get_grad_norm(
                self.policy.critic_head.parameters()
            )

            torch.nn.utils.clip_grad_norm_(
                self.policy.critic_head.parameters(),
                max_norm=0.5,
            )

            self.critic_optimizer.step()

            if epoch == (
                critic_update_epochs - 1
            ):

                print(
                    f"critic_epoch={epoch+1} "
                    f"loss={critic_loss_epoch.item():.4f} "
                    f"grad={critic_grad:.4f}"
                )
        #
        # -------------------------------------------------
        # Actor update
        # -------------------------------------------------
        #

        actor_losses = []
        entropy_losses = []

        for t, graph_obs in enumerate(
            rollout_batch.observations
        ):

            output = self.policy(
                graph_obs
            )

            dist = Categorical(
                logits=output.logits
            )

            log_probs = dist.log_prob(
                rollout_batch.actions[t]
            )

            actor_loss_t = -(
                log_probs
                * advantages[t].detach()
                * adv_scale
            ).sum()

            actor_losses.append(
                actor_loss_t
            )

            entropy_losses.append(
                dist.entropy().sum()
            )

        actor_loss = torch.stack(
            actor_losses
        ).mean()

        entropy_loss = torch.stack(
            entropy_losses
        ).mean()

        actor_objective = (
            actor_loss
            - self.entropy_coef
            * entropy_loss
        )

        self.actor_encoder_optimizer.zero_grad()

        actor_objective.backward()

        encoder_grad = get_grad_norm(
            self.policy.encoder.parameters()
        )

        actor_grad = get_grad_norm(
            self.policy.actor_head.parameters()
        )

        print()

        print(
            f"encoder_grad : {encoder_grad:.4f}"
        )

        print(
            f"actor_grad   : {actor_grad:.4f}"
        )

        print(
            f"critic_grad  : {critic_grad:.4f}"
        )

        if explained_variances:

            print(
                f"ExplainedVar : "
                f"{np.mean(explained_variances):.3f}"
            )

        print(
            f"Value mean   : "
            f"{np.mean(value_means):.3f}"
        )

        print(
            f"Value std    : "
            f"{np.mean(value_stds):.3f}"
        )

        print(
            f"Return mean  : "
            f"{np.mean(return_means):.3f}"
        )

        print(
            f"Return std   : "
            f"{np.mean(return_stds):.3f}"
        )

        print(
            f"TD mean      : "
            f"{np.mean(td_error_means):.3f}"
        )

        print(
            f"TD max       : "
            f"{np.max(td_error_maxes):.3f}"
        )

        print()

        torch.nn.utils.clip_grad_norm_(
            list(
                self.policy.encoder.parameters()
            )
            + list(
                self.policy.actor_head.parameters()
            ),
            max_norm=0.5,
        )

        self.actor_encoder_optimizer.step()

        #
        # -------------------------------------------------
        # Final losses
        # -------------------------------------------------
        #

        total_loss = (
            actor_loss
            + critic_loss_epoch.detach()
            - self.entropy_coef * entropy_loss
        )

        result = {
            "actor_loss": actor_loss.detach().item(),
            "critic_loss": critic_loss_epoch.detach().item(),
            "entropy_loss": entropy_loss.detach().item(),
            "total_loss": total_loss.detach().item(),

            #
            # Rollout statistics
            #
            "rollout_length": len(
                rollout_batch.observations
            ),

            "mean_training_reward": (
                rollout_batch.rewards
                .mean()
                .item()
            ),

            #
            # Diagnostics
            #
            "explained_variance": (
                float(np.mean(explained_variances))
                if explained_variances
                else 0.0
            ),

            "value_mean": float(
                np.mean(value_means)
            ),

            "value_std": float(
                np.mean(value_stds)
            ),

            "return_mean": float(
                np.mean(return_means)
            ),

            "return_std": float(
                np.mean(return_stds)
            ),

            "td_error_mean": float(
                np.mean(td_error_means)
            ),

            "td_error_max": float(
                np.max(td_error_maxes)
            ),

            "encoder_grad": encoder_grad,

            "actor_grad": actor_grad,

            "critic_grad": critic_grad,
        }

        print()

        print(
            "=" * 70
        )

        print(
            f"Reward      : "
            f"{result['mean_training_reward']:.4f}"
        )

        print(
            f"Actor Loss  : "
            f"{result['actor_loss']:.4f}"
        )

        print(
            f"Critic Loss : "
            f"{result['critic_loss']:.4f}"
        )

        print(
            f"Entropy     : "
            f"{result['entropy_loss']:.4f}"
        )

        print(
            f"ExplainedVar: "
            f"{result['explained_variance']:.3f}"
        )

        print(
            f"TD mean/max : "
            f"{result['td_error_mean']:.3f}"
            f" / "
            f"{result['td_error_max']:.3f}"
        )

        print(
            "=" * 70
        )

        print()

        return result
