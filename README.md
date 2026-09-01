# GP Transient Steady-State Prediction

This repository contains the code, simulated data and experiment-level results accompanying the dissertation **Steady-State Prediction from Partial Transient Measurements Using Gaussian Process Regression**.

The study investigates whether early measurements from a nonlinear continuous stirred-tank reactor (CSTR), together with relevant historical transitions, can be used to predict the unobserved trajectory and final steady state before the process has physically settled.

## Repository structure

```text
.
├── code/          Python source code for data generation, experiments and the final model
├── data/          Simulated dataset containing 1,000 process transitions
├── evaluation/    Fixed configuration-test and additional-holdout identifiers
├── results/       Published CSV results, configurations and figures
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Dataset

`data/transition_batches_long.csv` is the primary dataset. Each row represents one sampled point from a steady-state-to-steady-state transition and contains:

- batch and sample identifiers;
- elapsed time;
- initial and target manipulated variables;
- simulated trajectories of `Ca`, `Cb`, `T`, `Cc` and `Cd`.

`data/transition_summary.csv` contains transition-level settling times, observation counts, input-space distances and state-space distances.

The dataset contains 1,000 noise-free simulated transitions. It is intended for reproducibility and methodological evaluation rather than as experimental plant data.

## Installation

Python 3.10 or later is recommended. Create and activate a virtual environment, then install the required packages:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Final model

The final configuration uses:

- the first eight observed anchors;
- anchor index as the temporal coordinate;
- 30 nearest preceding historical transitions;
- similarity based on the initial operating condition and signed input changes;
- separate scalar radial-basis-function Gaussian Process models for product concentration and reactor temperature;
- a fixed prediction range covering anchor indices 8--13;
- joint endpoint identification when the normalised changes in both outputs remain below 2% for two consecutive intervals.

The true settling endpoint is not supplied during prediction. If the convergence
criterion is not satisfied, the prediction at anchor index 13 is returned and
the transition is flagged as unresolved.

Reproduce the 50-transition additional holdout evaluation from the repository
root without replacing the published outputs:

```powershell
python code/final_gp_model.py `
  --additional-holdout-size 50 `
  --additional-test-seed 20260901 `
  --exclude-batches 84 93 `
  --output-dir reproduced_results/additional_holdout_50_outputs
```

To perform a short smoke test, add `--max-tests 2` while retaining a separate
output directory.

The fixed holdout identifiers and published objective performance summary are
available at:

```text
evaluation/selected_additional_holdout_batches.csv
results/additional_holdout_50_outputs/final_model_objective_summary.csv
```

Across the 50 additional holdout transitions, final-value mean absolute errors
were 0.0102 for product concentration and 1.20 K for reactor temperature. These
corresponded to 0.761% and 0.197% of their respective holdout ranges. The endpoint
criterion was satisfied for 32 of 50 transitions (64%), and the selected endpoint
differed from the simulated settling endpoint by a mean of 0.86 anchor positions.
The first eight observations represented 23.2% of the physical settling period
on average.

## Comparative experiments

Run individual experiments from the repository root as required:

```bash
python code/experiment_anchor_count.py
python code/experiment_temporal_coordinate.py
python code/experiment_distance_feature.py
python code/experiment_feature_ablation.py
python code/experiment_history_size.py
python code/experiment_history_selection.py
python code/experiment_selection_disturbance_features.py
python code/experiment_transition_direction.py
python code/experiment_categorical_direction_encoding.py
python code/experiment_state_space_size.py
```

Each experiment writes its CSV files and figures to the corresponding directory under `results/`. Rerunning an experiment may replace the published files in that directory. Copy the repository or commit the published results before rerunning the full experiments if they need to be retained unchanged.

## Regenerating the simulated dataset

The supplied dataset can be regenerated with:

```bash
python code/generate_cstr_transition_dataset.py --output-dir reproduced_data
```

Dataset generation simulates all 1,000 transitions and may take substantially longer than loading the supplied CSV files.

## Reproducibility notes

- The 29 fixed configuration-comparison batches are stored in `evaluation/selected_test_batches.csv`.
- The 50 final additional-holdout batches are stored in `evaluation/selected_additional_holdout_batches.csv`.
- The comparative experiments used the simulated endpoint as a common retrospective evaluation point; the final holdout evaluation used a fixed future range and did not use the true endpoint during prediction.
- Random seeds and experiment settings are recorded in the source files and the JSON configuration files under `results/`.
- Published numerical results are retained as CSV files so that reported figures and summary statistics can be inspected without rerunning every model.
- Runtime measurements can vary with hardware and software environment.

## Citation

If this repository supports your work, please cite the accompanying dissertation and this software repository. Machine-readable citation metadata are provided in `CITATION.cff`.

## Licence

The source code and repository contents are released under the [MIT License](LICENSE). Please retain attribution when reusing or adapting the material.
