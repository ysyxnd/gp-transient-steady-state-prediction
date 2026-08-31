# GP Transient Steady-State Prediction

This repository contains the code, simulated data and experiment-level results accompanying the dissertation **Steady-State Prediction from Partial Transient Measurements Using Gaussian Process Regression**.

The study investigates whether early measurements from a nonlinear continuous stirred-tank reactor (CSTR), together with relevant historical transitions, can be used to predict the unobserved trajectory and final steady state before the process has physically settled.

## Repository structure

```text
.
├── code/          Python source code for data generation, experiments and the final model
├── data/          Simulated dataset containing 1,000 process transitions
├── evaluation/    Fixed test-batch identifiers used for model comparison
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
- separate scalar radial-basis-function Gaussian Process models for product concentration and reactor temperature.

Run the final evaluation from the repository root:

```bash
python code/final_gp_model.py
```

To perform a short smoke test without replacing the published results, specify a separate output directory:

```bash
python code/final_gp_model.py --max-tests 2 --output-dir reproduced_results/final_model_smoke_test
```

The published objective performance summary is available at:

```text
results/final_model_outputs/final_model_objective_summary.csv
```

Across 29 eligible fixed test transitions, the final steady-state mean absolute error was 0.00331 for product concentration and 0.645 K for reactor temperature. These corresponded to 0.247% and 0.119% of their respective test ranges. The mean observation period was approximately 19.8% of the physical settling time, representing an average waiting-time reduction of approximately 80.2%.

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

- The fixed evaluation batches are stored in `evaluation/selected_test_batches.csv`.
- Random seeds and experiment settings are recorded in the source files and the JSON configuration files under `results/`.
- Published numerical results are retained as CSV files so that reported figures and summary statistics can be inspected without rerunning every model.
- Runtime measurements can vary with hardware and software environment.

## Citation

If this repository supports your work, please cite the accompanying dissertation and this software repository. Machine-readable citation metadata are provided in `CITATION.cff`.

## Licence

The source code and repository contents are released under the [MIT License](LICENSE). Please retain attribution when reusing or adapting the material.

