"""
base_trainer.py

Shared training infrastructure for graph-based MARL methods.

Why this exists
---------------
Most graph-based MARL algorithms share the same high-level workflow:

    Collect rollout
        ↓
    Build rollout batch
        ↓
    Estimate returns / advantages
        ↓
    Update policy
        ↓
    Report training statistics

Only the optimisation step differs significantly between algorithms.

This base class implements the common workflow while allowing subclasses
to provide their own update rules.

Examples
--------
GraphCTDETrainer(BaseGraphTrainer)
GraphPPOTrainer(BaseGraphTrainer)
GraphCOMATrainer(BaseGraphTrainer)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from marl_tsc.graph_based.graph_runner import GraphRunner
from marl_tsc.graph_based.graph_rollout import GraphRollout
from marl_tsc.graph_based.advantage_estimator import (
    AdvantageEstimator,
)


class BaseGraphTrainer(ABC):

    def __init__(
        self,
        env,
        policy,
        optimizer,
        rollout_steps: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):

        self.env = env
        self.policy = policy
        self.optimizer = optimizer

        self.rollout_steps = rollout_steps

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.runner = GraphRunner(
            env=env,
            policy=policy,
        )

    def collect_batch(self):

        transitions = self.runner.collect_rollout(
            num_steps=self.rollout_steps,
        )

        rollout_batch = (
            GraphRollout.from_transitions(
                transitions,
                self.env.agent_ids,
            )
        )

        # -----------------------------
        # CHANGED: fetch the critic's value estimate for the
        # observation immediately following the rollout, instead of
        # letting compute_gae assume a terminal state (zero future
        # value).
        #
        # Episodes in this env run far longer than a single rollout
        # (~600 env steps vs. 64-step rollouts), so the final
        # transition in almost every rollout is a TRUNCATION, not a
        # genuine episode end. Bootstrapping with zero there was
        # telling GAE "there is no future reward after this point",
        # which is false and was the source of the compounding
        # negative-value drift seen in training.
        #
        # self.runner is expected to expose the most recent
        # observation after collect_rollout() returns (i.e. the
        # observation the next rollout will start from). If your
        # GraphRunner stores this under a different attribute name,
        # adjust the line below accordingly.
        # -----------------------------
        last_observation = self.runner.last_observation

        with torch.no_grad():
            bootstrap_output = self.policy(last_observation)
            # bootstrap_output.value is (N, 1) from the per-node
            # critic; squeeze to (N,) to match what compute_gae
            # expects for bootstrap_value.
            bootstrap_value = bootstrap_output.value.squeeze(-1)

        advantage_batch = (
            AdvantageEstimator.compute_gae(
                rollout_batch,
                bootstrap_value=bootstrap_value,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
        )

        return (
            rollout_batch,
            advantage_batch,
        )

    def train_step(self):

        rollout_batch, advantage_batch = (
            self.collect_batch()
        )

        return self.update(
            rollout_batch,
            advantage_batch,
        )

    @abstractmethod
    def update(
        self,
        rollout_batch,
        advantage_batch,
    ):
        """
        Perform one optimisation step.

        Must be implemented by subclasses.
        """
        raise NotImplementedError
