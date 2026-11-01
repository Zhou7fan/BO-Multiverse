# De-identification of the operation menu

The menu is released in its final form without per-paper provenance. In
addition, identifiers that would let a reader recognise one specific source
paper among the menu papers were renamed. Mapping applied (all menu text
fields; source/evidence fields were removed entirely):

| original | released as |
|---|---|
| source_ref (P*:D*:O*), evidence quotes (pdf:*/code:*) | removed |
| CAREER | career-sequence foundation model |
| coarse_grained_lasso / wage_pred_methods | coarse-grained lasso / wage prediction methods |
| dragonnet | joint debiasing network |
| round-2-wage | alternating debiasing fine-tune |
| famwgt | survey weight |
| expf, expfsq, expp, exppsq, edyrs, colldeg, advdeg | experience (with squares), education years and degree indicators |
| 21 / 330 occupation categories | coarse / fine occupation categories (counts dropped) |
| PSID | survey panel |
| Table 1 / Table 2 / Table S4 | predictive-accuracy table / main-results table / appendix table |
| clipping_threshold | propensity clipping threshold |

Also scrubbed outside the menu, in the wage-gap space itself:

- `multi_space.yaml` meta.source and instantiation notes no longer name the
  menu-building project or the paper index; fork labels lost their
  "(D*)" index suffixes.
- `ops/wage_ops.py` docstring and module name no longer carry the paper index.

Nothing in the spaces' fork/choice definitions was changed: column names and
choice values of the wage-gap space keep the real dataset columns, because the
wage-gap paper is openly one of the four study papers.
