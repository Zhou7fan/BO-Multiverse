# -*- coding: utf-8 -*-
"""Generic multiverse engine (paper-agnostic, v0.2).

Module overview:
- space_loader       : loads multi_space.yaml, menu backtracking + role validation, DFS enumeration, encoding.
- choice_runtime     : executes forks in assignment order (inline code / ops modules), prefix memoization.
- gpu_stats          : batched closed-form estimation on torch+CUDA; CPU fallback for logit/quantile/custom.
- effect_size        : multi100-standardized effect size (Cohen's d) with sign alignment.
- full_space         : fully enumerates one space and shards it to disk (used by exp2).
- runner             : executes sampled assignment lists, support/objective judgment, parquet sharding to disk.
- optimizer_adapter  : evaluate(assignment)->objective, BOCS/COMBO encoding, warm-up.
- run_optimizer      : generic BO driver loop, per-query CSV logging, convergence metrics.
- benchmark          : throughput / GPU utilization / VRAM / batch scan / CPU fallback report.
"""

__version__ = "0.2.0"
ENGINE_SEED = 20261020
