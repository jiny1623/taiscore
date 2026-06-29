# Training Pipeline

This document describes how TAIScore is used in critic-actor training.

## Single-Round Training

1. Generate an initial response `y0` for each training prompt with the current actor.
2. Train a critic with GRPO. For each prompt, sample multiple critiques from the critic.
3. For each critique, call the refiner endpoint to produce a revised response `y1`.
4. Call the judge endpoint to score `(prompt, y0, critique, y1)` with TAIScore.
5. Use the within-prompt normalized TAIScore values as GRPO rewards.
6. After critic training, generate critique-guided revisions and convert them into DPO preference pairs.
7. Train the actor with DPO using `(chosen=y1, rejected=y0)` pairs or another experiment-specific filtering rule.

## Co-Evolution

Co-evolution alternates critic and actor updates across disjoint prompt subsets:

```text
round 1: train critic against actor_0 -> build DPO pairs -> train actor_1
round 2: train critic against actor_1 -> build DPO pairs -> train actor_2
round 3: train critic against actor_2 -> build DPO pairs -> train actor_3
```

The important property is actor-conditioned critic training: the refiner used inside the reward loop should match the actor whose behavior the critic is learning to improve.

In script form, each round has three stages:

```text
1. critic GRPO
   inputs: current actor/refiner endpoint, critic initialization, round prompt shard
   output: round-specific trained critic checkpoint

2. preference pair generation
   inputs: trained critic, current actor/refiner endpoint, initial responses
   output: DPO pairs with chosen revised responses and rejected initial responses

3. actor DPO
   inputs: current actor checkpoint, preference pairs, reference model
   output: next actor checkpoint
```

The next round uses the new actor as the refiner policy for critic training and as the initialization for DPO. `scripts/run_coevolution_template.sh` shows the corresponding round-level wiring.

The paper provides the experimental settings and reported results. This repository exposes the reward implementation and the associated training interface.
