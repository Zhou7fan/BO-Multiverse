# Third-party optimizer libraries

The adapters in this directory wrap the following third-party libraries.
The libraries are not vendored here; install or clone them from the URLs
below and place them so that `sys.path` finds them (the adapters resolve
their imports through the configured vendor path).

| Optimizer | Library | URL | License |
|---|---|---|---|
| BOCS | engine/bo/bocs (included) | --- | --- |
| COMBO | engine/bo/combo (included) | --- | --- |
| CASMOPOLITAN | casmopolitan | https://github.com/facebookresearch/casmopolitan | MIT |
| Bounce | bounce | https://github.com/huawei-noah/HEBO/tree/master/BOUNCE | MIT |
| BODi | bodi | https://github.com/huawei-noah/HEBO/tree/master/BODi | MIT |
| MerCBO | mercbo | https://github.com/huawei-noah/HEBO/tree/master/MerCBO | MIT |
| HEBO | HEBO | https://github.com/huawei-noah/HEBO | MIT |
| bo_pr | bo_pr (probabilistic reparameterization) | https://github.com/facebookresearch/bo_pr | MIT |
| GP-EI/PI/TS/UCB | scikit-learn Gaussian processes | https://scikit-learn.org | BSD-3-Clause |
| BoTorch | botorch | https://github.com/pytorch/botorch | MIT |
| Ax | ax | https://github.com/facebook/Ax | MIT |
| BayBE | baybe | https://github.com/emdgroup/baybe | Apache-2.0 |
| Summit | summit | https://github.com/sustainable-processes/summit | MIT |
| bayes_optim | bayes-optim | https://github.com/wangronin/Bayesian-Optimization | MIT |
| BOCS (external variant) | bocs | https://github.com/baptistar/BOCS | MIT |
| Random search | engine/bo/random_search.py (included) | --- | --- |

Versions are pinned in the experiment environment manifest that accompanies
the search logs (`core/write_algo_status.py` records, per run, the library
versions used).
