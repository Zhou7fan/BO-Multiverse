# operation_menu.json

Final operation menu used to instantiate the four spaces.

- `families` (115): coarse families of analytic operations, each with
  definition / scientific function / operation object / mechanism scope.
- `methods` (262): specific methods (id, family_id, name, definition,
  scientific_function, operation_object, mechanism, semantic_signature,
  aliases, parameter_slots).
- `rows` (269): menu entries — one per decision-option, with `row_id`,
  `entity_kind` (method / strategy / metric / parameter_setting /
  workflow_bundle / uncertain), `composition`, `components` (references to
  method ids, with parameters and applicability conditions),
  `option_parameters`, `retained_conditions`, `uncertainties`.
- `limitations`: known coverage limitations of the menu.

Per-paper provenance (which paper contributed which row) is intentionally not
included; see DEIDENTIFICATION.md.
