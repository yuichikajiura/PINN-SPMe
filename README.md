# PINN-SPMe

Physics-informed and supervised neural-network training for lithium-ion battery state estimation with SPMe models.

This repository contains the electrolyte-aware SPMe training workflow from generated SPMe CSV datasets.

The single entry point is:

```bash
python main.py <config.yaml>
```

`main.py` dispatches from the config keys `model_type` and `method`.

---

## Quick Start: SPMe

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch build that matches your CUDA setup.

### 2. Validate supervised NN mode first

This mode trains the SPMe neural network directly against normalized state labels from the generated CSV files. It is useful for checking whether the network size and input history length can generalize before moving to PINN losses.

```bash
python main.py config_spme_nn.yaml
```

Outputs are written to `save_dir`, by default:

- `results/spme_nn/spme_nn_n.pth`
- `results/spme_nn/spme_nn_e.pth`
- `results/spme_nn/spme_nn_p.pth`
- `results/spme_nn/spme_nn_history.csv`

### 3. Train SPMe PINN mode

After NN capacity looks reasonable, run the SPMe PINN path:

```bash
python main.py config_spme_pinn.yaml
```

This trains using SPMe rollout losses, voltage loss, separate solid/electrolyte lithium conservation losses, and SPM-style integration consistency.

---

## Weights & Biases Logging

W&B is optional and controlled by config. For SPMe NN mode, edit `config_spme_nn.yaml`:

```yaml
use_wandb: true
wandb_project: pinn-spme
wandb_run_name: spme-nn-state-supervision
wandb_mode: online
wandb_plot_every: 50
wandb_plot_samples: 2
```

Then run:

```bash
wandb login
python main.py config_spme_nn.yaml
```

NN mode logs:

- train and validation grouped normalized losses
- learning rate
- periodic full-profile actual-vs-predicted plots for `c_s_n(r)`, reconstructed `c_e(x)`, `c_s_p(r)`, and voltage
- plotted-sample MAE and max absolute state error in raw concentration units

For a quick local W&B test without cloud sync:

```yaml
use_wandb: true
wandb_mode: offline
epochs: 5
max_train_windows: 100
max_val_windows: 50
print_every: 1
wandb_plot_every: 1
```

---

## Config Guide

### Dispatch Options

| Option | Values | Meaning |
|---|---|---|
| `model_type` | `spme` | Selects the SPMe code path. |
| `method` | `nn`, `pinn` | `nn` trains direct neural state prediction; `pinn` trains through model rollout and physics/data losses. |

Current SPMe configs:

| File | Purpose |
|---|---|
| `config_spme_nn.yaml` | Supervised SPMe NN training against actual state labels. Use this first to test network capacity. |
| `config_spme_pinn.yaml` | SPMe PINN training with rollout, voltage, conservation, and integration-consistency losses. |

### Save And Logging Options

| Option | Example | Meaning |
|---|---|---|
| `save_dir` | `results/spme_nn` | Directory for checkpoints, CSV history, and local W&B files. |
| `network_save_name` | `spme_nn.pth` | Base filename for three model files: `*_n.pth`, `*_e.pth`, `*_p.pth`. |
| `history_save_name` | `spme_nn_history.csv` | Per-epoch training history filename. |
| `print_every` | `50` | Console logging interval in epochs. |
| `eval_every` | `50` | PINN diagnostic interval for a fixed small validation subset. Training still runs every epoch. |
| `full_eval_every` | `500` | PINN diagnostic interval for full train/validation rollouts and full-profile W&B plots. |
| `checkpoint_every` | `1000` | Epoch interval for full training checkpoints. |
| `keep_last_checkpoint` | `true` | Also update `checkpoint_latest.pth` when periodic checkpoints are saved. |
| `save_optimizer_state` | `true` | Store and restore optimizer states in checkpoints. |
| `resume_from_checkpoint` | empty or path | Resume all three networks, optimizers, best validation value, and history from a checkpoint. |
| `use_wandb` | `false` | Enables W&B logging when true. |
| `wandb_project` | `pinn-spme` | W&B project name. |
| `wandb_run_name` | `spme-nn-state-supervision` | W&B run name. |
| `wandb_mode` | `online`, `offline`, `disabled` | W&B mode. Use `offline` to log locally. |
| `wandb_plot_every` | `50` | Epoch interval for W&B plots. In PINN mode this controls full-profile plots independently from `full_eval_every`. |
| `wandb_plot_samples` | `2` | Number of train/val samples to plot. |
| `plot_samples` | `20` | Optional alias/override for per-dataset W&B spatial samples. |
| `profile_timing` | `true` | Enable detailed timing of PINN training sections. |
| `profile_timing_every` | `20` | Display/log timing summaries every N displayed epochs when profiling is enabled. |
| `profile_timing_warmup_epochs` | `5` | Skip timing summaries for the first N displayed epochs. |
| `profile_timing_log_wandb` | `true` | Log detailed timing fields to W&B with `timing/..._s` keys. |
| `debug_delta_sei_grad` | `false` | Print/log `delta_sei_scaled.grad` and derived SEI quantities after backward and before optimizer step. |
| `debug_delta_sei_grad_every` | `20` | Epoch interval for the SEI gradient debug check. |

### Network Options

| Option | Example | Meaning |
|---|---|---|
| `batch_size` | `256` | Training batch size. |
| `layer_lstm` | `1` | Number of LSTM layers in each state block. |
| `hidden_lstm` | `64` | LSTM hidden dimension. |
| `layer_fc` | `1` | Number of fully connected hidden layers after the LSTM. |
| `hidden_fc` | `256` | Width of each fully connected hidden layer. |
| `activation` | `tanh` | Activation used by the custom LSTM/FC block. |
| `network_output` | `raw` | SPMe uses SPM-style `CustomLSTM` output scaling to raw physical concentrations. |
| `epochs` | `1000` | Number of training epochs. |
| `lrate` | `0.001` | Initial learning rate. |
| `decay_step` | `500` | Epoch after which multiplicative LR decay starts. |
| `decay_rate` | `0.999` | Per-epoch LR multiplier after `decay_step`. |
| `seed` | `1234` | NumPy and PyTorch random seed. |

SPMe uses the same architecture principle as the original SPM implementation:

- `nn_models[0]`: anode solid concentration network
- `nn_models[1]`: electrolyte concentration network
- `nn_models[2]`: cathode solid concentration network

Each model receives the same normalized current/voltage history and predicts one raw concentration block. The packed state order is:

```text
[c_s_n, c_e, c_s_p]
```

The output ranges are subsystem-specific:

- anode: `p.c_n_soc0` to `p.c_n_soc100`
- electrolyte: `p.ce_min` to `p.ce_max`
- cathode: `p.c_p_soc100` to `p.c_p_soc0`

NN mode trains with grouped normalized losses for scale balance:

```text
loss = loss_c_s_n + loss_c_e + loss_c_s_p
```

### Sequence And Integrator Options

| Option | Example | Meaning |
|---|---|---|
| `k` | `90` | LSTM input history length only: number of historical current/voltage samples fed to the network. |
| `h` | `1` | Model integration step in seconds for SPMe rollout. |
| `pinn_rollout_steps` | `90` | PINN rollout length: number of SPMe integration steps used for voltage, lithium, and integration-consistency losses. |
| `integ_type` | `1` | Solid-state integrator option: `0` = RK4, `1` = Euler. |
| `electrolyte_integrator` | `implicit` | SPMe electrolyte integrator: `implicit` for training, `rk4` for debugging/reference. |
| `electrolyte_substeps` | `1` | RK4 electrolyte substeps when `electrolyte_integrator: rk4`. Keep `1` for implicit training. |

For PINN-SPMe training with one-second data, use:

```yaml
h: 1
electrolyte_integrator: implicit
electrolyte_substeps: 1
```

The explicit RK4 electrolyte path is retained as a reference/debug option.

### Data Options

| Option | Example | Meaning |
|---|---|---|
| `data_source` | `csv` | Loads generated SPMe CSV datasets. |
| `data_dir` | `data` | Base directory for relative data paths. |
| `train_data_files` | `spme_train_BJDST_soh90.csv` | Training CSV files. Multiple files are concatenated while preserving sequence lengths for windowing. |
| `val_data_files` | `spme_val_DST_soh90.csv` | Validation CSV files. |
| `test_data_files` | `spme_debug_CC_1C_discharge.csv` | Reserved for later test/evaluation scripts. |
| `train_window_stride` | `30` | Spacing between NN windows. In PINN integration consistency, adjacent NN window `j+1` corresponds to this many rollout steps later. |
| `val_window_stride` | `30` | Spacing between validation NN windows. |
| `max_train_windows` | empty or integer | Optional cap on sampled train windows for smoke tests. |
| `max_val_windows` | empty or integer | Optional cap on sampled validation windows for smoke tests. |
| `shuffle_pinn_profiles` | `true` | In PINN mode, shuffle profile order only; windows inside each batch remain temporally ordered. |

SPMe CSV files must include:

- `current_A`
- `voltage_V`
- `c_s_n_1 ... c_s_n_N`
- `c_e_1 ... c_e_N`
- `c_s_p_1 ... c_s_p_N`

The current SPMe configs expect:

```yaml
n_r: 15
n_x_n: 10
n_x_s: 5
n_x_p: 10
```

This corresponds to 14 negative solid states, 22 electrolyte states, and 14 positive solid states.

### Input Scaling Options

| Option | Example | Meaning |
|---|---|---|
| `OneC` | `5.0` | One-C current magnitude used for input scaling. |
| `max_Crate` | `4` | Current scaling range is `[-OneC * max_Crate, OneC * max_Crate]`. |
| `V_upper` | `4.2` | Upper voltage bound used for input scaling. |
| `V_lower` | `2.5` | Lower voltage bound used for input scaling. |

### SPMe Model Options

| Option | Example | Meaning |
|---|---|---|
| `n_r` | `15` | Radial solid discretization count. The state excludes one boundary value, so each solid state block has `n_r - 1` outputs. |
| `n_x_n` | `10` | Negative-electrode electrolyte/control-volume count. |
| `n_x_s` | `5` | Separator electrolyte/control-volume count. |
| `n_x_p` | `10` | Positive-electrode electrolyte/control-volume count. |
| `ce_min` | `200.0` | Lower concentration bound used by SPMe bounds losses. |
| `ce_max` | `2000.0` | Upper concentration bound used by SPMe bounds losses. |
| `temperature_K` | `298.15` | Cell temperature for SPMe calculations. |
| `use_electrolyte_activity` | `false` | Enables electrolyte activity correction if implemented/enabled in the model. |
| `disable_electrolyte_ohmic` | `false` | Debug switch to remove electrolyte ohmic voltage term. |
| `disable_electrolyte_concentration` | `false` | Debug switch to remove electrolyte concentration overpotential term. |

### SPMe PINN Loss Options

Used by `config_spme_pinn.yaml` with `method: pinn`.

| Option | Example | Meaning |
|---|---|---|
| `w_v` | `1.0` | Voltage loss weight. |
| `w_li_s` | `1.0` | Solid lithium conservation loss weight. |
| `w_li_e` | `0.0` | Electrolyte lithium conservation loss weight. Keep `0.0` initially for SEI estimation. |
| `w_integ` | `1.0` | SPM-style integration consistency loss weight across anode, electrolyte, and cathode blocks. |
| `estimate_delta_sei` | `true` | Train one scaled SEI thickness parameter, `delta_sei_scaled`. |
| `delta_sei_init` | `1.0e-12` | Initial SEI thickness estimate in meters. Fresh-cell SEI is zero. |
| `delta_sei_lower` / `delta_sei_upper` | `0.0` / `2.0e-6` | Physical clamp range for estimated SEI thickness. |
| `delta_sei_true` | `5.05188e-07` | Optional diagnostic target for W&B/history logging. |
| `epsilon_e_n_true` | `0.0560283` | Optional diagnostic target for W&B/history logging. |
| `nLi_s_fresh` | `0.2840` | Fresh solid lithium inventory. Degraded `nLi_s_est` is derived from SEI lithium loss. |
| `R_f_n_base` | `0.0` | Fresh-cell anode film resistance base. `R_f_n_est = R_f_n_base + delta_sei_est / kappa_sei`. |
| `rho_sei`, `M_sei`, `kappa_sei` | `1690.0`, `0.162`, `1.0e-6` | SEI material constants used to map thickness to lost lithium and film resistance. |
| `supervise_states` | `false` | Kept as a config marker; SPMe PINN follows SPM style and does not use true-state supervision loss. |

SPMe PINN loss is:

```text
loss = w_v * loss_voltage
     + w_li_s * loss_solid_lithium
     + w_li_e * loss_electrolyte_lithium
     + w_integ * loss_integ
```

Boundary conditions are handled as hard constraints through solid state reconstruction, SPMe integration, conservative electrolyte FVM, and MATLAB-aligned `C_ce` electrolyte reconstruction. There are no extra soft BC loss terms.

SPMe PINN estimates degradation through one scalar SEI thickness. The derived quantities are:

```text
nLi_sei_est = (2 * rho_sei / M_sei) * a_s_n * Area_n * L_n * delta_sei_est
nLi_s_est = nLi_s_fresh - nLi_sei_est
epsilon_e_n_est = epsilon_e_n_fresh - a_s_n * delta_sei_est
R_f_n_est = R_f_n_base + delta_sei_est / kappa_sei
```

Lithium inventory losses use bare NN predictions only and compare them to these derived targets. They do not use rolled SPMe states:

```text
loss_li_s = RMSE((nLi_s_nn - nLi_s_est) / nLi_s_init)
loss_li_e = RMSE((nLi_e_nn - nLi_e_est) / nLi_e_init)
```

The current default keeps `w_li_e: 0.0`; electrolyte inventory is derived from porosity for diagnostics but is not used as a training target unless that weight is enabled.

For PINN integration consistency, SPMe follows the original SPM batching strategy: batches contain consecutive windows from a single profile, each NN is evaluated once for that batch, SPMe rolls the predicted initial states forward for `pinn_rollout_steps`, and `loss_integ` compares rolled future states with already-computed neighboring NN predictions by tensor slicing. Shifted windows are not sent through the LSTMs again. With `train_window_stride: 30` and `pinn_rollout_steps: 90`, the integration loss compares rolled `t+30`, `t+60`, and `t+90` states to NN windows `+1`, `+2`, and `+3`.

In `method: nn`, the network is trained directly against state labels using grouped normalized losses for scale balance. The outputs and plots remain raw physical concentrations.

---

## Data

The `data/` directory contains generated SPMe datasets.

### SPMe Datasets

Current generated SPMe files:

- `spme_train_BJDST_soh90.csv`
- `spme_train_FUDS_soh90.csv`
- `spme_train_US06_soh90.csv`
- `spme_val_DST_soh90.csv`
- `spme_debug_CC_1C_discharge_soh90.csv`

These are used by `config_spme_nn.yaml` and `config_spme_pinn.yaml`.

## Tests And Verification

Run the SPMe unit tests:

```bash
python -m unittest tests.test_spme -v
python tests/test_spme_consistency.py --strict
```

Run the MATLAB cross-check when MATLAB is available:

```bash
export PATH="/Applications/MATLAB_R2025b.app/bin:$PATH"
python -m unittest tests.test_spme_matlab_cross_run -v
```

The MATLAB cross-run test intentionally requires MATLAB. It should not be skipped silently because the comparison is only meaningful when both Python SPMe and MATLAB SPMe execute.

---

## Project Structure

```text
PINN-SPMe/
|-- main.py                         # SPMe entry point; dispatches by method
|-- config_spme_nn.yaml             # SPMe supervised NN state training
|-- config_spme_pinn.yaml           # SPMe PINN training
|-- train_pinn_spme.py              # Shared SPMe NN/PINN training code
|-- pinn_spme_network.py            # SPMe neural state estimator
|-- spme_model.py                   # Differentiable SPMe rollout model
|-- electrolyte_fvm.py              # Electrolyte FVM RHS and implicit/RK4 steps
|-- losses_spme.py                  # SPMe voltage, conservation, bounds losses
|-- init_params.py                  # SPMe parameters
|-- data/                           # SPMe CSV data
|-- tests/                          # SPMe verification tests
`-- results/                        # Saved models, histories, plots, W&B files
```

---

## References

- PINN methodology: [Physics-Informed Neural Networks for Battery Parameter Identification and State Estimation](https://ieeexplore.ieee.org/document/10644822)
- SPM/SPMe model background and parameters: [SPMe Observability Paper](https://ecal.studentorg.berkeley.edu/pubs/SPMe-Obs-Journal-Final.pdf) and [SPMeT repository](https://github.com/scott-moura/SPMeT)

---

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0), in line with the license of SPMeT-derived model components. See [LICENSE](LICENSE) for the full text.
