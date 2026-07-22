# Graph-Based Training Framework

## Overview

The graph-based framework provides a standard pipeline for training graph neural network policies for traffic signal control.

A typical workflow is:

```text
Traffic Environment
        ↓
Graph Representation
        ↓
Graph Policy
        ↓
Rollout Collection
        ↓
Advantage Estimation
        ↓
Policy Update
        ↓
Training History / Saved Model
```

The framework separates these stages so that new graph architectures and training algorithms can be added without rewriting the rest of the system.

---

# Typical Usage

## Training an Existing Policy

The simplest way to train a graph policy is through a training entry point such as:

```python
model, history, model_path = train_graph_ctde(
    config_file=paths.config_file,
    network_file=paths.network_file,
    traffic_light_ids=traffic_light_ids,
    output_dir=OUTPUT_DIR,
    total_timesteps=10000,
)
```

This function:

1. Creates the graph environment
2. Builds the graph policy
3. Creates the trainer
4. Runs training
5. Saves the model
6. Returns training history

---

## Evaluating a Trained Policy

Graph policies are evaluated through a `GraphPolicyAdapter`.

```python
from marl_tsc.graph_based.graph_builder import GraphBuilder
from marl_tsc.graph_based.graph_policy_adapter import GraphPolicyAdapter

topology = GraphBuilder(
    paths.network_file
).build()

graph_policy = GraphPolicyAdapter(
    model,
    topology,
)

results = evaluate_policy(
    config_file=paths.config_file,
    traffic_light_ids=traffic_light_ids,
    policy=graph_policy,
)

print(results)
```

The adapter converts standard environment observations into graph observations and forwards them through the graph policy.

---

# Creating a New Graph Algorithm

Most new algorithms require only three components:

1. A policy
2. A trainer
3. A training entry point

---

## Step 1: Create a Policy

Create a new policy architecture inside:

```text
models/
```

A policy typically consists of:

```text
Graph Encoder
        ↓
Actor Head
        ↓
Action Logits

Graph Encoder
        ↓
Critic Head
        ↓
Value Estimate
```

The policy should accept a `GraphObservation` and produce:

* action logits
* value estimates

### Example

```python
from marl_tsc.graph_based.encoders.gat_encoder import GATEncoder

from marl_tsc.graph_based.models.actor import ActorHead
from marl_tsc.graph_based.models.critic import CriticHead
from marl_tsc.graph_based.models.graph_policy import GraphPolicy


encoder = GATEncoder(
    obs_dim=10,
    hidden_dim=64,
)

actor_head = ActorHead(
    embedding_dim=64,
    action_dim=4,
)

critic_head = CriticHead(
    embedding_dim=64,
)

policy = GraphPolicy(
    encoder=encoder,
    actor_head=actor_head,
    critic_head=critic_head,
)
```

The resulting policy can immediately be used by any graph trainer.

### Policy Factory

Reusable policy construction logic should be placed in:

```text
models/policy_factory.py
```

Example:

```python
policy = build_default_graph_policy(
    obs_dim=10,
    action_dim=4,
)
```

This keeps training scripts small and ensures policies are built consistently.

---

## Step 2: Create a Trainer

Create a trainer that inherits from:

```python
BaseGraphTrainer
```

Example:

```python
class GraphPPOTrainer(BaseGraphTrainer):
    ...
```

The only required method is:

```python
update(
    rollout_batch,
    advantage_batch,
)
```

This method contains the algorithm-specific learning logic.

Examples:

* Actor-Critic
* PPO
* A2C
* Custom methods

The remainder of the training pipeline is inherited automatically.

### Minimal Example

```python
class MyTrainer(BaseGraphTrainer):

    def update(
        self,
        rollout_batch,
        advantage_batch,
    ):

        loss = ...

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.detach())
        }
```

---

## Step 3: Create a Training Entry Point

Create a launcher function such as:

```python
train_graph_ppo(...)
```

Responsibilities:

* Create environment
* Create policy
* Create optimizer
* Create trainer
* Call `run_training(...)`

This becomes the public entry point for the new algorithm.

### Example

```python
def train_graph_ppo(
    config_file,
    network_file,
    traffic_light_ids,
    output_dir,
    total_timesteps,
):

    env = GraphTrafficEnv(
        config_file=config_file,
        network_file=network_file,
        possible_agents=traffic_light_ids,
    )

    policy = build_default_graph_policy(
        obs_dim=env.obs_dim,
        action_dim=4,
    )

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=3e-4,
    )

    trainer = GraphPPOTrainer(
        env=env,
        policy=policy,
        optimizer=optimizer,
    )

    return run_training(
        trainer=trainer,
        total_timesteps=total_timesteps,
        rollout_steps=trainer.rollout_steps,
        algorithm_name="graph_ppo",
        model_path=f"{output_dir}/graph_ppo.pt",
    )
```

The training entry point should contain only experiment setup and orchestration. Learning logic belongs in the trainer.

### Typical Usage

```python
model, history, model_path = train_graph_ppo(
    config_file=paths.config_file,
    network_file=paths.network_file,
    traffic_light_ids=traffic_light_ids,
    output_dir=OUTPUT_DIR,
    total_timesteps=10000,
)
```

---

# How Everything Connects

A contributor implementing a new algorithm should generally touch only:

```text
encoders/
    ↓
models/
    ↓
new_trainer.py
    ↓
train_new_algorithm.py
```

The remainder of the framework already provides:

```text
Graph Environment
Rollout Collection
Advantage Computation
Training Loop
Model Saving
History Tracking
Evaluation Integration
```

and can usually be reused unchanged.

---

# Training Pipeline

The complete training flow is:

```text
GraphTrafficEnv
        ↓
GraphRunner
        ↓
GraphRollout
        ↓
AdvantageEstimator
        ↓
Trainer.update()
        ↓
Optimizer Step
```

### GraphTrafficEnv

Produces graph observations from the traffic environment.

### GraphRunner

Interacts with the environment and collects experience.

### GraphRollout

Converts transitions into training batches.

### AdvantageEstimator

Computes returns and advantages.

### Trainer

Uses those batches to update the policy.

---

# Model Pipeline

The complete inference flow is:

```text
GraphObservation
        ↓
Encoder
        ↓
Node Embeddings
        ↓
Actor Head
        ↓
Actions

Node Embeddings
        ↓
Critic Head
        ↓
Value Estimate
```

The encoder can be replaced without changing the rest of the framework.

---

# File Reference

### graph_types.py

Defines shared graph data structures.

Primary type:

```python
GraphObservation
```

Used throughout the framework.

---

### graph_builder.py

Builds graph topology and agent ordering from a SUMO network.

Provides:

* edge indices
* agent ordering
* graph connectivity

---

### graph_env.py

Graph wrapper around the traffic environment.

Converts standard observations into graph observations.

---

### graph_runner.py

Collects environment interactions and records transitions.

---

### graph_rollout.py

Converts transitions into a `RolloutBatch`.

Stores:

* observations
* actions
* rewards
* values
* dones

---

### advantage_estimator.py

Computes:

* returns
* advantages

using Generalized Advantage Estimation (GAE).

---

### base_trainer.py

Shared training workflow.

Provides:

```text
Collect Rollout
      ↓
Build Batch
      ↓
Compute Advantages
      ↓
Update Policy
```

All graph trainers inherit from this class.

---

### graph_ctde_trainer.py

Reference trainer implementation.

Contains the optimization logic for Graph CTDE.

Useful as the template for new algorithms.

---

### run_training.py

Generic training loop.

Handles:

* repeated training iterations
* history collection
* logging
* model saving

Reusable across all graph algorithms.

---

### train_graph_ctde.py

Graph CTDE entry point.

Creates:

* environment
* policy
* optimizer
* trainer

and launches training.

---

### graph_policy_adapter.py

Compatibility layer used during evaluation.

Converts:

```python
{
    agent_id: observation
}
```

into a `GraphObservation` and returns graph-policy actions.

---

### encoders/

Graph representation learning modules.

Current implementation:

```text
gat_encoder.py
```

Encoders transform graph observations into node embeddings.

---

### models/

Contains policy-network components.

Current components include:

```text
actor.py
critic.py
graph_policy.py
policy_factory.py
```

Responsibilities:

* actor networks
* critic networks
* policy composition
* policy construction helpers

---

# Design Principle

Most experimentation should occur in one of three places:

1. Encoders
2. Policies
3. Trainers

The remainder of the framework provides the common infrastructure needed to train, save, load, and evaluate graph-based policies.
