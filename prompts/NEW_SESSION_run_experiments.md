# New-session prompt (updated: reverse-engineering edition)

> The old version framed the task as "doing multiverse analysis". The new version makes it **reverse engineering**:
> given the paper's reported results, infer its analysis path with as few BO queries as possible.
> The multiverse is only the search space; BO's sample efficiency is the main metric.

## Just use this

Copy the entire contents of the file below into a new conversation:

**`agent_prompts/REVERSE_ENGINEERING_MASTER.md`**

It already contains:

- The three-layer structure (paper-agnostic menu / paper-specific subspaces / generic engine);
- The information-isolation and space-building rules for Experiments 1/2/3;
- Objective function, success threshold, convergence metrics;
- Run commands for BOCS / COMBO / the random baseline;
- Deliverables and constraints.

## Sub-agents for the three experiments

To run BO in parallel, hand the three files below to three sub-agents:

- `agent_prompts/exp1_blind_agent.md`
- `agent_prompts/exp2_reproduction_agent.md`
- `agent_prompts/exp3_perturbation_agent.md`

Each sub-agent only touches its own `exp*/results/reverse_bo/`; the shared engine is read-only.

## Key differences from the old version

| Old version | New version |
|---|---|
| Goal: enumerate/present the multiverse | Goal: reverse engineer, find the paper's path ASAP |
| Main metric: space size, specification curve | Main metric: `queries_to_target`, `best_so_far` vs query, success rate |
| Bigger space = more complete analysis | Bigger space = stronger proof of BO's sample efficiency |
| Menu may retain paper examples | Menu is purely paper-agnostic; all choices are AI-instantiated, need not be in the menu |
| Run the whole space first, then optimize | Use BO directly to find the target in the space, no full enumeration |
