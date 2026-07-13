# S2-15 Flip Spike

## Spec Matrix

- Controls: none; configured control set
- Weighting: unweighted; configured weight column
- Estimators: OLS, TWFE
- Sample axis: not included
- Specs per dataset: 8

## Decision Rules

- GO: successful comparable matched-pair results show at least one positive-negative sign flip with readable axis attribution.
- GO validates readable axis fragility in the current axis set; it does not by itself validate a canonical estimator-only or staggered-estimator story.
- NO-GO: successful comparable results exist for both datasets, but neither has a readable sign flip; backstop axis expansion required.
- INCONCLUSIVE: estimator/spec failures or no sign-comparable matched pairs prevent a reliable product decision.
- Descriptive dominant sign axis ranks axes by sign_flip_count, then sign_flip_rate, then mean_abs_delta; this is not formal statistical dominance.

## Limitations

- sample axis excluded because config and current estimators do not define a canonical applied filter
- pre_period_window and include_never_treated excluded because current estimators do not apply them
- canonical staggered estimators are not supported in the committed core; only OLS and TWFE are run

## Overall Decision

**GO**: all analyzed datasets (divorce and castle) are conclusive and at least one has readable sign fragility

## Divorce

- Decision: **GO** - readable axis fragility; control_set is dominant, with estimator flips also observed
- Data: `data/divorce/raw/divorce.csv`
- Specs: 8
- Successful specs: 8
- Failed specs: 0
- Band: fragile
- Sign agreement: 0.625
- Coefficient range: -3.827094 to 3.732567
- Descriptive dominant sign axis: control_set
- Dominant partial-R2 axis: control_set

### Matched-Pair Axis Attribution

| Axis | Pairs | Sign-comparable | Sign flips | Sign flip rate | Significance-comparable | Significance flips | Significance flip rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control_set | 4 | 4 | 3 | 0.750 | 4 | 0 | 0.000 |
| sample | 0 | 0 | 0 | - | 0 | 0 | - |
| pre_period | 0 | 0 | 0 | - | 0 | 0 | - |
| clustering | 0 | 0 | 0 | - | 0 | 0 | - |
| never_treated | 0 | 0 | 0 | - | 0 | 0 | - |
| estimator | 4 | 4 | 1 | 0.250 | 4 | 0 | 0.000 |
| weighting | 4 | 4 | 1 | 0.250 | 4 | 0 | 0.000 |

### ANOVA Partial-R2

- control_set: 0.616
- sample: -
- pre_period: -
- clustering: -
- never_treated: -
- estimator: 0.287
- weighting: 0.534

### Failed Specs

- None

### Spec List

- divorce_00: estimator=OLS, controls=[], weight=None
- divorce_01: estimator=TWFE, controls=[], weight=None
- divorce_02: estimator=OLS, controls=[], weight=weight
- divorce_03: estimator=TWFE, controls=[], weight=weight
- divorce_04: estimator=OLS, controls=['pcinc', 'asmrh', 'cases'], weight=None
- divorce_05: estimator=TWFE, controls=['pcinc', 'asmrh', 'cases'], weight=None
- divorce_06: estimator=OLS, controls=['pcinc', 'asmrh', 'cases'], weight=weight
- divorce_07: estimator=TWFE, controls=['pcinc', 'asmrh', 'cases'], weight=weight

## Castle

- Decision: **GO** - readable axis fragility; control_set is dominant, with estimator flips also observed
- Data: `data/castle/raw/castle.csv`
- Specs: 8
- Successful specs: 8
- Failed specs: 0
- Band: fragile
- Sign agreement: 0.750
- Coefficient range: -0.038779 to 0.371595
- Descriptive dominant sign axis: control_set
- Dominant partial-R2 axis: control_set

### Matched-Pair Axis Attribution

| Axis | Pairs | Sign-comparable | Sign flips | Sign flip rate | Significance-comparable | Significance flips | Significance flip rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control_set | 4 | 4 | 2 | 0.500 | 4 | 2 | 0.500 |
| sample | 0 | 0 | 0 | - | 0 | 0 | - |
| pre_period | 0 | 0 | 0 | - | 0 | 0 | - |
| clustering | 0 | 0 | 0 | - | 0 | 0 | - |
| never_treated | 0 | 0 | 0 | - | 0 | 0 | - |
| estimator | 4 | 4 | 2 | 0.500 | 4 | 2 | 0.500 |
| weighting | 4 | 4 | 0 | 0.000 | 4 | 2 | 0.500 |

### ANOVA Partial-R2

- control_set: 0.435
- sample: -
- pre_period: -
- clustering: -
- never_treated: -
- estimator: 0.058
- weighting: 0.101

### Failed Specs

- None

### Spec List

- castle_00: estimator=OLS, controls=[], weight=None
- castle_01: estimator=TWFE, controls=[], weight=None
- castle_02: estimator=OLS, controls=[], weight=population
- castle_03: estimator=TWFE, controls=[], weight=population
- castle_04: estimator=OLS, controls=['unemployrt', 'income', 'poverty', 'police', 'prisoner'], weight=None
- castle_05: estimator=TWFE, controls=['unemployrt', 'income', 'poverty', 'police', 'prisoner'], weight=None
- castle_06: estimator=OLS, controls=['unemployrt', 'income', 'poverty', 'police', 'prisoner'], weight=population
- castle_07: estimator=TWFE, controls=['unemployrt', 'income', 'poverty', 'police', 'prisoner'], weight=population
