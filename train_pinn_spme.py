import os
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

import losses_spme
from electrolyte_fvm import electrolyte_lithium_inventory, reconstruct_full_concentration
from pinn_spme_network import create_spme_nn_models, predict_spme_state_raw
from spme_model import SPMeModel


def _create_windows(array, window, lengths, stride=1, max_windows=None, seed=0):
    array = np.asarray(array)
    original_is_vector = array.ndim == 1
    if original_is_vector:
        array = array[:, None]

    windows = []
    start = 0
    for length in lengths:
        if length < window:
            raise ValueError(f"Profile length {length} is shorter than required window length {window}.")
        for idx in range(0, length - window + 1, stride):
            windows.append(array[start + idx:start + idx + window])
        start += length
    windows = np.stack(windows, axis=0)
    if max_windows is not None and max_windows > 0 and windows.shape[0] > max_windows:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(windows.shape[0], size=max_windows, replace=False))
        windows = windows[keep]
    if original_is_vector:
        return windows[:, :, 0]
    return windows


def _create_window_metadata(lengths, window, stride=1, max_windows=None, seed=0):
    rows = []
    offset = 0
    for profile_idx, length in enumerate(lengths):
        if length < window:
            raise ValueError(f"Profile length {length} is shorter than required window length {window}.")
        for start_idx in range(0, length - window + 1, stride):
            rows.append((profile_idx, offset + start_idx, start_idx, start_idx + window - 1))
        offset += length
    rows = np.asarray(rows, dtype=np.int64)
    if max_windows is not None and max_windows > 0 and rows.shape[0] > max_windows:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(rows.shape[0], size=max_windows, replace=False))
        rows = rows[keep]
    return {
        'profile_idx': rows[:, 0],
        'global_start_idx': rows[:, 1],
        'window_start_idx': rows[:, 2],
        'target_idx': rows[:, 3],
    }


def _configured_file_weights(cfg, split_name, profile_names):
    value = cfg.get(f'{split_name}_data_file_weights', cfg.get('data_file_weights', None))
    if value is None:
        return None
    if isinstance(value, dict):
        weights = []
        for name in profile_names:
            if name in value:
                weights.append(float(value[name]))
            elif os.path.basename(name) in value:
                weights.append(float(value[os.path.basename(name)]))
            else:
                raise ValueError(f"Missing {split_name}_data_file_weights entry for {name}")
        return np.asarray(weights, dtype=np.float64)
    if len(value) != len(profile_names):
        raise ValueError(
            f"{split_name}_data_file_weights has {len(value)} entries, "
            f"but {len(profile_names)} {split_name} files were loaded."
        )
    return np.asarray([float(item) for item in value], dtype=np.float64)


def _window_weights_from_metadata(metadata, split_data, cfg):
    profile_names = [profile['name'] for profile in split_data.get('profiles', [])]
    split_name = split_data.get('split', 'train')
    file_weights = _configured_file_weights(cfg, split_name, profile_names)
    n_windows = len(metadata['profile_idx'])
    if file_weights is None:
        return np.ones(n_windows, dtype=np.float32), None
    if np.any(file_weights < 0):
        raise ValueError(f"{split_name}_data_file_weights must be non-negative.")
    if not np.any(file_weights > 0):
        raise ValueError(f"At least one {split_name}_data_file_weights entry must be positive.")

    profile_idx = metadata['profile_idx']
    weights = np.zeros(n_windows, dtype=np.float64)
    summary = []
    for idx, name in enumerate(profile_names):
        mask = profile_idx == idx
        count = int(mask.sum())
        if count == 0:
            per_window = 0.0
        else:
            per_window = float(file_weights[idx]) / count
            weights[mask] = per_window
        summary.append({
            'name': name,
            'file_weight': float(file_weights[idx]),
            'windows': count,
            'per_window_weight': per_window,
        })

    mean_weight = weights.mean()
    if mean_weight > 0:
        weights = weights / mean_weight
        for item in summary:
            item['per_window_weight_normalized'] = item['per_window_weight'] / mean_weight
    return weights.astype(np.float32), summary


def _pinn_rollout_steps(cfg):
    return int(cfg.get('pinn_rollout_steps', cfg.get('rollout_steps', cfg['k'])))


def _profile_slices_from_metadata(metadata):
    profile_ids = metadata['profile_idx']
    if len(profile_ids) == 0:
        return []
    slices = []
    start = 0
    current_profile = profile_ids[0]
    for idx in range(1, len(profile_ids)):
        if profile_ids[idx] != current_profile:
            slices.append((start, idx))
            start = idx
            current_profile = profile_ids[idx]
    slices.append((start, len(profile_ids)))
    return slices


def _iter_ordered_profile_batches(dataset, batch_size, shuffle_profiles=False, seed=0, epoch=0):
    profile_slices = list(dataset.get('profile_slices', [(0, dataset['u'].shape[0])]))
    if shuffle_profiles and len(profile_slices) > 1:
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(profile_slices))
        profile_slices = [profile_slices[idx] for idx in order]

    for profile_start, profile_end in profile_slices:
        if profile_end <= profile_start:
            continue
        if batch_size is None or batch_size <= 0:
            yield slice(profile_start, profile_end)
            continue
        for batch_start in range(profile_start, profile_end, batch_size):
            yield slice(batch_start, min(batch_start + batch_size, profile_end))


def _slice_pinn_dataset(dataset, row_slice):
    subset = {}
    tensor_keys = {
        'u',
        'i_future_ave',
        'i_future_sampled',
        'v_future_sampled',
        'x_future_sampled',
        'window_weights',
    }
    for key, value in dataset.items():
        if key in tensor_keys:
            subset[key] = value[row_slice]
        elif key == 'metadata':
            subset[key] = {name: values[row_slice] for name, values in value.items()}
        elif key == 'profile_slices':
            subset[key] = [(0, subset['u'].shape[0])]
        else:
            subset[key] = value
    return subset


def _validation_eval_subset(dataset, cfg):
    n_subset = int(cfg.get('eval_subset_windows', cfg.get('batch_size', 256)))
    if n_subset <= 0 or n_subset >= dataset['u'].shape[0]:
        return dataset
    profile_start, profile_end = dataset.get('profile_slices', [(0, dataset['u'].shape[0])])[0]
    end = min(profile_end, profile_start + n_subset)
    return _slice_pinn_dataset(dataset, slice(profile_start, end))


def _is_due_epoch(epoch_display, every):
    return every is not None and every > 0 and epoch_display % every == 0


def sync_device(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps' and hasattr(torch, 'mps'):
        torch.mps.synchronize()


def _timer_now(device):
    sync_device(device)
    return time.perf_counter()


def _timer_add(timing, key, start, device):
    sync_device(device)
    timing[key] = timing.get(key, 0.0) + (time.perf_counter() - start)


class TimingAccumulator:
    def __init__(self, device, enabled=False):
        self.device = torch.device(device)
        self.enabled = bool(enabled)
        self.times = {}

    def _sync(self):
        if self.enabled:
            sync_device(self.device)

    def section(self, key):
        if not self.enabled:
            return nullcontext()
        return _TimingSection(self, key)

    def add(self, key, elapsed):
        if self.enabled:
            self.times[key] = self.times.get(key, 0.0) + elapsed

    def update(self, values):
        if self.enabled:
            for key, value in values.items():
                if key.endswith('_sec'):
                    key = key[:-4]
                self.times[key] = self.times.get(key, 0.0) + float(value)

    def payload(self):
        keys = [
            'data_loading',
            'nn_forward',
            'rollout',
            'solid_update',
            'electrolyte_update',
            'loss_voltage',
            'loss_li_s',
            'loss_li_e',
            'loss_integ',
            'loss_assembly',
            'backward',
            'optimizer',
            'batch_total',
            'logging_eval',
            'epoch_total',
        ]
        return {f'timing/{key}_s': self.times.get(key, 0.0) for key in keys}

    def history_fields(self):
        return {f'timing_{key}_s': value for key, value in self.times.items()}

    def print_report(self, epoch_display):
        total = self.times.get('epoch_total', 0.0)
        print(f"Timing profile epoch {epoch_display}")
        print(f"  total epoch: {total:.2f} s")
        for key in [
            'data_loading',
            'nn_forward',
            'rollout',
            'solid_update',
            'electrolyte_update',
            'loss_voltage',
            'loss_li_s',
            'loss_li_e',
            'loss_integ',
            'loss_assembly',
            'backward',
            'optimizer',
            'batch_total',
            'logging_eval',
        ]:
            elapsed = self.times.get(key, 0.0)
            pct = 100.0 * elapsed / total if total > 0 else 0.0
            print(f"  {key:18s}{elapsed:9.2f} s {pct:6.1f}%")


class _TimingSection:
    def __init__(self, timer, key):
        self.timer = timer
        self.key = key
        self.start = None

    def __enter__(self):
        self.timer._sync()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.timer._sync()
        self.timer.add(self.key, time.perf_counter() - self.start)


def _average_chunks(sequence, h):
    batch_size, length = sequence.shape
    sequence = sequence[:, :length - (length % h)]
    return sequence.reshape(batch_size, -1, h).mean(dim=-1)


def _sample_chunks(sequence, h):
    idx = torch.arange(h - 1, sequence.shape[1], h, device=sequence.device)
    return sequence.index_select(1, idx)


def _sample_state_chunks(sequence, h):
    idx = torch.arange(h - 1, sequence.shape[1], h, device=sequence.device)
    return sequence.index_select(1, idx)


def _resolve_data_files(files, data_dir):
    if not files:
        return []
    resolved = []
    for item in files:
        path = item if os.path.isabs(item) else os.path.join(data_dir, item)
        if not os.path.exists(path):
            raise FileNotFoundError(f"SPMe dataset not found: {path}")
        resolved.append(path)
    return resolved


def _spme_state_columns(p):
    c_s_n_cols = [f"c_s_n_{idx}" for idx in range(1, p.n_cs_n + 1)]
    c_e_cols = [f"c_e_{idx}" for idx in range(1, p.n_ce + 1)]
    c_s_p_cols = [f"c_s_p_{idx}" for idx in range(1, p.n_cs_p + 1)]
    return c_s_n_cols, c_e_cols, c_s_p_cols


def _load_spme_csv_dataset(files, p):
    frames = []
    lengths = []
    c_s_n_cols, c_e_cols, c_s_p_cols = _spme_state_columns(p)
    required = ["current_A", "voltage_V", *c_s_n_cols, *c_e_cols, *c_s_p_cols]
    profiles = []

    for path in files:
        df = pd.read_csv(path)
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required SPMe columns: {missing}")
        frames.append(df)
        lengths.append(len(df))
        profile_state = df[c_s_n_cols + c_e_cols + c_s_p_cols].to_numpy(dtype=np.float32)
        if "time_s" in df.columns:
            time_values = df["time_s"].to_numpy(dtype=np.float32)
        elif "Test_Time(s)" in df.columns:
            time_values = df["Test_Time(s)"].to_numpy(dtype=np.float32)
        else:
            time_values = np.arange(len(df), dtype=np.float32)
        profile = {
            'name': os.path.basename(path),
            'time': time_values,
            'current': df["current_A"].to_numpy(dtype=np.float32),
            'voltage': df["voltage_V"].to_numpy(dtype=np.float32),
            'state': profile_state,
        }
        profiles.append(profile)

    if not frames:
        raise ValueError("No SPMe CSV datasets were provided.")

    data = pd.concat(frames, ignore_index=True)
    state = data[c_s_n_cols + c_e_cols + c_s_p_cols].to_numpy(dtype=np.float32)
    return {
        'current': data["current_A"].to_numpy(dtype=np.float32),
        'voltage': data["voltage_V"].to_numpy(dtype=np.float32),
        'state': state,
        'lengths': lengths,
        'profiles': profiles,
    }


def _load_training_dataset(cfg, device, p):
    if cfg.get("data_source", "csv") != "csv" and not cfg.get("train_data_files"):
        raise ValueError("Only CSV-backed SPMe datasets are supported.")
    data_dir = cfg.get("data_dir", "data")
    train_files = _resolve_data_files(cfg.get("train_data_files", []), data_dir)
    val_files = _resolve_data_files(cfg.get("val_data_files", []), data_dir)
    train = _load_spme_csv_dataset(train_files, p)
    val = _load_spme_csv_dataset(val_files, p)
    print("Loaded SPMe CSV datasets:")
    print(f"  train files: {len(train_files)}, samples: {sum(train['lengths'])}")
    print(f"  val files:   {len(val_files)}, samples: {sum(val['lengths'])}")
    return {'train': train, 'val': val}


def _prepare_dataset(split_data, cfg, p, device):
    k = int(cfg['k'])
    h = int(cfg['h'])
    rollout_steps = _pinn_rollout_steps(cfg)
    window_length = k + rollout_steps
    lengths = split_data['lengths']
    split_name = split_data.get('split', 'train')
    stride = cfg.get(f'{split_name}_window_stride', cfg.get('window_stride', 1))
    max_windows = cfg.get(f'max_{split_name}_windows', cfg.get('max_windows', None))

    i_windows = _create_windows(split_data['current'], window_length, lengths, stride, max_windows, cfg.get('seed', 0))
    v_windows = _create_windows(split_data['voltage'], window_length, lengths, stride, max_windows, cfg.get('seed', 0))
    x_windows = _create_windows(split_data['state'], window_length, lengths, stride, max_windows, cfg.get('seed', 0))
    metadata = _create_window_metadata(lengths, window_length, stride, max_windows, cfg.get('seed', 0))
    window_weights_np, weight_summary = _window_weights_from_metadata(metadata, split_data, cfg)

    i_windows = torch.tensor(i_windows, dtype=torch.float32, device=device)
    v_windows = torch.tensor(v_windows, dtype=torch.float32, device=device)
    x_windows = torch.tensor(x_windows, dtype=torch.float32, device=device)
    window_weights = torch.tensor(window_weights_np, dtype=torch.float32, device=device)

    u = torch.zeros(i_windows.shape[0], k, p.input_size, device=device)
    u[:, :, 0] = i_windows[:, :k]
    u[:, :, 1] = v_windows[:, :k]

    ub = torch.tensor([
        cfg['OneC'] * cfg['max_Crate'],
        cfg['V_upper'],
    ], dtype=torch.float32, device=device)
    lb = torch.tensor([
        -cfg['OneC'] * cfg['max_Crate'],
        cfg['V_lower'],
    ], dtype=torch.float32, device=device)
    u = (u - lb) / (ub - lb)

    i_future = i_windows[:, k:]
    i_future_ave = _average_chunks(i_future, h)
    i_future_sampled = torch.cat([i_windows[:, k - 1:k], _sample_chunks(i_future, h)], dim=1)
    v_future_sampled = torch.cat([v_windows[:, k - 1:k], _sample_chunks(v_windows[:, k:], h)], dim=1)

    x_future_sampled = torch.cat([
        x_windows[:, k - 1:k, :],
        _sample_state_chunks(x_windows[:, k:, :], h),
    ], dim=1)

    return {
        'u': u,
        'i_future_ave': i_future_ave,
        'i_future_sampled': i_future_sampled,
        'v_future_sampled': v_future_sampled,
        'x_future_sampled': x_future_sampled,
        'window_weights': window_weights,
        'weight_summary': weight_summary,
        'metadata': metadata,
        'profile_slices': _profile_slices_from_metadata(metadata),
        'window_stride': stride,
        'pinn_rollout_steps': i_future_ave.shape[1],
        'profiles': split_data.get('profiles', []),
    }


def _prepare_state_dataset(split_data, cfg, p, device):
    k = int(cfg['k'])
    lengths = split_data['lengths']
    split_name = split_data.get('split', 'train')
    stride = cfg.get(f'{split_name}_window_stride', cfg.get('window_stride', 1))
    max_windows = cfg.get(f'max_{split_name}_windows', cfg.get('max_windows', None))

    i_windows = _create_windows(split_data['current'], k, lengths, stride, max_windows, cfg.get('seed', 0))
    v_windows = _create_windows(split_data['voltage'], k, lengths, stride, max_windows, cfg.get('seed', 0))
    x_windows = _create_windows(split_data['state'], k, lengths, stride, max_windows, cfg.get('seed', 0))
    metadata = _create_window_metadata(lengths, k, stride, max_windows, cfg.get('seed', 0))

    i_windows = torch.tensor(i_windows, dtype=torch.float32, device=device)
    v_windows = torch.tensor(v_windows, dtype=torch.float32, device=device)
    x_windows = torch.tensor(x_windows, dtype=torch.float32, device=device)

    u = torch.zeros(i_windows.shape[0], k, p.input_size, device=device)
    u[:, :, 0] = i_windows
    u[:, :, 1] = v_windows

    ub = torch.tensor([cfg['OneC'] * cfg['max_Crate'], cfg['V_upper']], dtype=torch.float32, device=device)
    lb = torch.tensor([-cfg['OneC'] * cfg['max_Crate'], cfg['V_lower']], dtype=torch.float32, device=device)
    u = (u - lb) / (ub - lb)
    return {
        'u': u,
        'x_target_raw': x_windows[:, -1, :],
        'i_target': i_windows[:, -1:].contiguous(),
        'v_target': v_windows[:, -1:].contiguous(),
        'metadata': metadata,
        'profiles': split_data.get('profiles', []),
    }


def _ce0(p):
    return getattr(p, 'ce0', p.c_e0)


def normalize_state(x_raw, p):
    n_end = p.n_cs_n
    e_end = n_end + p.n_ce
    c_n_scale = p.c_n_max - p.c_n_min
    c_p_scale = p.c_p_max - p.c_p_min
    if x_raw.dim() == 2:
        c_s_n_scaled = (x_raw[:, :n_end] - p.c_n_min) / c_n_scale
        ce_norm = (x_raw[:, n_end:e_end] - _ce0(p)) / _ce0(p)
        c_s_p_scaled = (x_raw[:, e_end:] - p.c_p_min) / c_p_scale
        return torch.cat([c_s_n_scaled, ce_norm, c_s_p_scaled], dim=-1)
    if x_raw.dim() == 3:
        c_s_n_scaled = (x_raw[:, :n_end, :] - p.c_n_min) / c_n_scale
        ce_norm = (x_raw[:, n_end:e_end, :] - _ce0(p)) / _ce0(p)
        c_s_p_scaled = (x_raw[:, e_end:, :] - p.c_p_min) / c_p_scale
        return torch.cat([c_s_n_scaled, ce_norm, c_s_p_scaled], dim=1)
    raise ValueError('State tensor must be 2D or 3D.')


def _spme_network_save_paths(cfg):
    save_dir = cfg.get('save_dir', 'results')
    filename = cfg.get('network_save_name', 'spme_network.pth')
    stem, ext = os.path.splitext(filename)
    if not ext:
        ext = '.pth'
    return [
        os.path.join(save_dir, f'{stem}_n{ext}'),
        os.path.join(save_dir, f'{stem}_e{ext}'),
        os.path.join(save_dir, f'{stem}_p{ext}'),
    ]


def _save_spme_nn_models(nn_models, cfg):
    for nn_model, path in zip(nn_models, _spme_network_save_paths(cfg)):
        torch.save(nn_model.state_dict(), path)


def _save_best_spme_nn_models(nn_models, cfg):
    save_dir = cfg.get('save_dir', 'results')
    names = ['best_n.pth', 'best_e.pth', 'best_p.pth']
    for nn_model, name in zip(nn_models, names):
        torch.save(nn_model.state_dict(), os.path.join(save_dir, name))


def _checkpoint_payload(epoch, nn_models, optimizers, best_val, history, cfg, model=None):
    payload = {
        'epoch': epoch,
        'nn_models': [nn_model.state_dict() for nn_model in nn_models],
        'optimizers': [optimizer.state_dict() for optimizer in optimizers],
        'schedulers': [],
        'best_val': best_val,
        'history': history,
        'config': cfg,
    }
    if model is not None:
        payload['model_state'] = model.state_dict()
    return payload


def _save_checkpoint(path, epoch, nn_models, optimizers, best_val, history, cfg, model=None):
    torch.save(_checkpoint_payload(epoch, nn_models, optimizers, best_val, history, cfg, model), path)


def _save_periodic_checkpoint(epoch, nn_models, optimizers, best_val, history, cfg, model=None):
    checkpoint_every = cfg.get('checkpoint_every', None)
    if not checkpoint_every or checkpoint_every <= 0:
        return
    if (epoch + 1) % checkpoint_every != 0:
        return

    save_dir = cfg.get('save_dir', 'results')
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1:07d}.pth')
    _save_checkpoint(checkpoint_path, epoch, nn_models, optimizers, best_val, history, cfg, model)
    if cfg.get('keep_last_checkpoint', True):
        _save_checkpoint(os.path.join(save_dir, 'checkpoint_latest.pth'), epoch, nn_models, optimizers, best_val, history, cfg, model)


def _save_interrupt_checkpoint(epoch, nn_models, optimizers, best_val, history, cfg, model=None):
    save_dir = cfg.get('save_dir', 'results')
    _save_checkpoint(
        os.path.join(save_dir, f'checkpoint_interrupt_epoch_{epoch + 1:07d}.pth'),
        epoch,
        nn_models,
        optimizers,
        best_val,
        history,
        cfg,
        model,
    )
    _save_checkpoint(os.path.join(save_dir, 'checkpoint_latest.pth'), epoch, nn_models, optimizers, best_val, history, cfg, model)
    pd.DataFrame(history).to_csv(os.path.join(save_dir, cfg['history_save_name']), index=False)


def _resume_from_checkpoint(cfg, nn_models, optimizers, device, model=None):
    checkpoint_path = cfg.get('resume_from_checkpoint', None)
    if not checkpoint_path:
        return 0, float('inf'), []
    checkpoint = torch.load(checkpoint_path, map_location=device)
    for nn_model, state_dict in zip(nn_models, checkpoint['nn_models']):
        nn_model.load_state_dict(state_dict)
    if model is not None and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=False)
    if cfg.get('save_optimizer_state', True) and 'optimizers' in checkpoint:
        for optimizer, state_dict in zip(optimizers, checkpoint['optimizers']):
            optimizer.load_state_dict(state_dict)
    start_epoch = int(checkpoint.get('epoch', -1)) + 1
    best_val = checkpoint.get('best_val', float('inf'))
    history = checkpoint.get('history', [])
    print(f"Resumed SPMe training from {checkpoint_path} at epoch {start_epoch}.")
    return start_epoch, best_val, history


def _load_pretrained_nn_models(cfg, nn_models, device):
    checkpoint_path = cfg.get('pretrained_nn_checkpoint', None)
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'nn_models' in checkpoint:
        state_dicts = checkpoint['nn_models']
    else:
        raise ValueError(
            "pretrained_nn_checkpoint must point to a checkpoint containing the 'nn_models' list."
        )
    for nn_model, state_dict in zip(nn_models, state_dicts):
        nn_model.load_state_dict(state_dict)
    print(f"Loaded pre-trained SPMe NN weights from {checkpoint_path}.")


def _evaluate(nn_models, model, dataset, cfg, p):
    with torch.no_grad():
        sample_weights = dataset.get('window_weights', None)
        x0_raw = predict_spme_state_raw(nn_models, dataset['u'])
        # SPMe physics expects raw concentration units.
        x_roll_future = model(x0_raw, dataset['i_future_ave'])
        x_roll_raw = torch.cat([x0_raw.unsqueeze(-1), x_roll_future], dim=-1)
        c_s_n_seq, c_e_seq, c_s_p_seq = model.unpack_state(x_roll_raw)
        c_n_seq, c_p_seq = model.calc_css_and_cs0_seq(x_roll_raw, dataset['i_future_sampled'])
        vt_pred = model.calc_voltage(c_n_seq[:, -1, :], c_p_seq[:, -1, :], c_e_seq, dataset['i_future_sampled'],
                                     disable_electrolyte_ohmic=cfg.get('disable_electrolyte_ohmic', False),
                                     disable_electrolyte_concentration=cfg.get('disable_electrolyte_concentration', False))

        loss_v = losses_spme.voltage_loss(vt_pred, dataset['v_future_sampled'], sample_weights=sample_weights)
        loss_li_s, loss_li_e, li_stats = _bare_nn_lithium_inventory_losses(
            model,
            x0_raw,
            p,
            sample_weights=sample_weights,
        )
        w_integ = cfg.get('w_integ', 1.0)
        if w_integ:
            loss_integ, integ_stats = _profile_integration_consistency_loss(
                model,
                x_roll_raw,
                x0_raw,
                dataset,
                p,
                sample_weights=sample_weights,
            )
        else:
            loss_integ = torch.zeros((), dtype=x_roll_raw.dtype, device=x_roll_raw.device)
            integ_stats = {'loss_integ_n': loss_integ, 'loss_integ_e': loss_integ, 'loss_integ_p': loss_integ}
        loss = (cfg['w_v'] * loss_v + cfg['w_li_s'] * loss_li_s + cfg['w_li_e'] * loss_li_e
                + w_integ * loss_integ)

    return {
        'loss': loss.item(),
        'loss_v': loss_v.item(),
        'loss_li_s': loss_li_s.item(),
        'loss_li_e': loss_li_e.item(),
        'loss_integ': loss_integ.item(),
        'loss_integ_n': integ_stats['loss_integ_n'].item(),
        'loss_integ_e': integ_stats['loss_integ_e'].item(),
        'loss_integ_p': integ_stats['loss_integ_p'].item(),
        'nLi_s_mean': li_stats['nLi_s_mean'].item(),
        'mean_pred_nLi_s': li_stats['mean_pred_nLi_s'].item(),
        'std_pred_nLi_s': li_stats['std_pred_nLi_s'].item(),
        'nLi_s_est': li_stats['nLi_s_est'].item(),
        'nLi_s_rel_std': li_stats['nLi_s_rel_std'].item(),
        'nLi_e_mean': li_stats['nLi_e_mean'].item(),
        'mean_pred_nLi_e': li_stats['mean_pred_nLi_e'].item(),
        'std_pred_nLi_e': li_stats['std_pred_nLi_e'].item(),
        'nLi_e_est': li_stats['nLi_e_est'].item(),
        'nLi_e_rel_std': li_stats['nLi_e_rel_std'].item(),
        **_float_stats(_degradation_stats(model, cfg)),
    }


def _warm_start_diagnostics(nn_models, model, dataset, cfg, p, prefix):
    max_windows = int(cfg.get('warm_start_diagnostic_windows', 2048))
    n_samples = min(dataset['u'].shape[0], max_windows)
    if n_samples == 0:
        return {}

    with torch.no_grad():
        u = dataset['u'][:n_samples]
        i_future_ave = dataset['i_future_ave'][:n_samples]
        i_future_sampled = dataset['i_future_sampled'][:n_samples]
        v_future_sampled = dataset['v_future_sampled'][:n_samples]
        target_seq_raw = dataset['x_future_sampled'][:n_samples]

        x_nn_raw = predict_spme_state_raw(nn_models, u)
        target0_raw = target_seq_raw[:, 0, :]
        loss_bare_state_norm = _nn_grouped_normalized_losses(x_nn_raw, target0_raw, model, p)[0]

        # SPMe physics expects raw concentration units.
        c_n_bare, c_p_bare = model.calc_css_and_cs0(x_nn_raw, i_future_sampled[:, 0:1])
        _, c_e_bare, _ = model.unpack_state(x_nn_raw)
        v_bare = model.calc_voltage(
            c_n_bare[:, -1],
            c_p_bare[:, -1],
            c_e_bare,
            i_future_sampled[:, 0:1],
            disable_electrolyte_ohmic=cfg.get('disable_electrolyte_ohmic', False),
            disable_electrolyte_concentration=cfg.get('disable_electrolyte_concentration', False),
        ).reshape_as(v_future_sampled[:, 0:1])
        loss_bare_v = losses_spme.voltage_loss(v_bare, v_future_sampled[:, 0:1])

        # SPMe physics expects raw concentration units.
        x_roll_future = model(x_nn_raw, i_future_ave)
        x_roll_raw = torch.cat([x_nn_raw.unsqueeze(-1), x_roll_future], dim=-1)
        target_roll_raw = target_seq_raw.permute(0, 2, 1)
        loss_roll_state_norm = losses_spme.rmse_loss(normalize_state(x_roll_raw, p), normalize_state(target_roll_raw, p))

        c_n_roll_seq, c_p_roll_seq = model.calc_css_and_cs0_seq(x_roll_raw, i_future_sampled)
        _, c_e_roll_seq, _ = model.unpack_state(x_roll_raw)
        v_roll = model.calc_voltage(
            c_n_roll_seq[:, -1, :],
            c_p_roll_seq[:, -1, :],
            c_e_roll_seq,
            i_future_sampled,
            disable_electrolyte_ohmic=cfg.get('disable_electrolyte_ohmic', False),
            disable_electrolyte_concentration=cfg.get('disable_electrolyte_concentration', False),
        )
        loss_roll_v = losses_spme.voltage_loss(v_roll, v_future_sampled)

    return {
        f'{prefix}/epoch0_bare_state_rmse_norm': loss_bare_state_norm.item(),
        f'{prefix}/epoch0_bare_voltage_rmse': loss_bare_v.item(),
        f'{prefix}/epoch0_rolled_state_rmse_norm': loss_roll_state_norm.item(),
        f'{prefix}/epoch0_rolled_voltage_rmse': loss_roll_v.item(),
    }


def _load_prepared_pinn_data(cfg, device, p):
    dataset = _load_training_dataset(cfg, device, p)
    dataset['train']['split'] = 'train'
    dataset['val']['split'] = 'val'
    train_data = _prepare_dataset(dataset['train'], cfg, p, device)
    val_data = _prepare_dataset(dataset['val'], cfg, p, device)
    print(f"Prepared windows: train={train_data['u'].shape[0]}, val={val_data['u'].shape[0]}")
    for split_name, split in [('train', train_data), ('val', val_data)]:
        if split.get('weight_summary'):
            print(f"  {split_name} file weights:")
            for item in split['weight_summary']:
                print(
                    f"    {item['name']}: file_weight={item['file_weight']:.6g}, "
                    f"windows={item['windows']}, "
                    f"per_window_weight={item['per_window_weight_normalized']:.6g}"
                )
    return train_data, val_data


def _load_prepared_nn_data(cfg, device, p):
    dataset = _load_training_dataset(cfg, device, p)
    dataset['train']['split'] = 'train'
    dataset['val']['split'] = 'val'
    train_data = _prepare_state_dataset(dataset['train'], cfg, p, device)
    val_data = _prepare_state_dataset(dataset['val'], cfg, p, device)
    print(f"Prepared NN windows: train={train_data['u'].shape[0]}, val={val_data['u'].shape[0]}")
    return train_data, val_data


def _nn_grouped_normalized_losses(pred_raw, target_raw, model, p):
    pred_scaled = normalize_state(pred_raw, p)
    target_scaled = normalize_state(target_raw, p)
    pred_c_s_n_scaled, pred_ce_norm, pred_c_s_p_scaled = model.unpack_state(pred_scaled)
    true_c_s_n_scaled, true_ce_norm, true_c_s_p_scaled = model.unpack_state(target_scaled)

    loss_c_s_n = losses_spme.rmse_loss(pred_c_s_n_scaled, true_c_s_n_scaled)
    loss_c_e = losses_spme.rmse_loss(pred_ce_norm, true_ce_norm)
    loss_c_s_p = losses_spme.rmse_loss(pred_c_s_p_scaled, true_c_s_p_scaled)
    loss = loss_c_s_n + loss_c_e + loss_c_s_p
    return loss, loss_c_s_n, loss_c_e, loss_c_s_p


def _relative_inventory_estimation_loss(inventory, estimate, init_scale, eps=1e-12, sample_weights=None):
    estimate = estimate.to(dtype=inventory.dtype, device=inventory.device).reshape(())
    scale = init_scale.to(dtype=inventory.dtype, device=inventory.device).reshape(()).abs() + eps
    rel = (inventory - estimate) / scale
    target = torch.zeros_like(rel)
    loss = losses_spme.rmse_loss(rel.reshape(-1, 1), target.reshape(-1, 1), sample_weights=sample_weights)
    if sample_weights is None:
        mean_pred = inventory.mean()
        std_pred = inventory.std(unbiased=False)
        rel_std = torch.sqrt(torch.mean(((inventory - mean_pred) / scale) ** 2))
    else:
        flat_inventory = inventory.reshape(-1)
        weights = sample_weights.to(dtype=inventory.dtype, device=inventory.device).reshape(-1)
        denom = weights.sum().clamp_min(torch.finfo(inventory.dtype).eps)
        mean_pred = (flat_inventory * weights).sum() / denom
        variance = (((flat_inventory - mean_pred) ** 2) * weights).sum() / denom
        std_pred = torch.sqrt(variance)
        rel_std = torch.sqrt(((((flat_inventory - mean_pred) / scale) ** 2) * weights).sum() / denom)
    return loss, mean_pred, std_pred, estimate, rel_std


def _degradation_stats(model, cfg):
    delta = model.delta_sei_est.reshape(())
    epsilon = model.epsilon_e_n_est.reshape(())
    nli_sei = model.nLi_sei_est.reshape(())
    nli_s = model.nLi_s_est.reshape(())
    nli_e = model.nLi_e_est.reshape(())
    r_f_n = model.R_f_n_est.reshape(())
    delta_true = torch.as_tensor(model.p.delta_sei_true, dtype=delta.dtype, device=delta.device)
    epsilon_true = torch.as_tensor(model.p.epsilon_e_n_true, dtype=epsilon.dtype, device=epsilon.device)
    nli_sei_true = model.C_li_sei.to(dtype=delta.dtype, device=delta.device) * delta_true
    return {
        'delta_sei_est': delta,
        'delta_sei_scaled': model.delta_sei_scaled.reshape(()).to(dtype=delta.dtype, device=delta.device),
        'delta_sei_true': delta_true,
        'delta_sei_error': delta - delta_true,
        'epsilon_e_n_est': epsilon,
        'epsilon_e_n_true': epsilon_true,
        'epsilon_e_n_error': epsilon - epsilon_true,
        'nLi_sei_est': nli_sei,
        'nLi_sei_true': nli_sei_true,
        'nLi_s_est': nli_s,
        'nLi_e_est': nli_e,
        'R_f_n_est': r_f_n,
    }


def _true_degradation_for_plot(model, n_samples, ref):
    dtype = ref.dtype
    device = ref.device
    delta = torch.as_tensor(model.p.delta_sei_true, dtype=dtype, device=device)
    nli_sei = model.nLi_sei_from_delta(delta).to(dtype=dtype, device=device)
    nli_s = model.nLi_s_from_sei_inventory(nli_sei).to(dtype=dtype, device=device)
    nli_e = model.nLi_e_from_delta(delta).to(dtype=dtype, device=device)
    return {
        'dynamic': model.electrolyte_dynamic_from_delta(delta, ref),
        'delta_sei': delta,
        'nLi_sei': nli_sei.expand(n_samples),
        'nLi_s': nli_s.expand(n_samples),
        'nLi_e': nli_e.expand(n_samples),
    }


def _float_stats(stats):
    return {key: float(value.detach().cpu().reshape(())) for key, value in stats.items()}


def _delta_sei_grad_diagnostics(model):
    grad = model.delta_sei_scaled.grad
    stats = _float_stats(_degradation_stats(model, model.cfg))
    if grad is None:
        stats['delta_sei_scaled_grad'] = float('nan')
        warning = "WARNING: delta_sei_scaled.grad is None; SEI path may be detached."
    else:
        grad_detached = grad.detach()
        stats['delta_sei_scaled_grad'] = float(grad_detached.reshape(-1)[0].cpu())
        if bool(torch.all(grad_detached == 0).detach().cpu()):
            warning = "WARNING: delta_sei_scaled.grad is None/zero; SEI path may be detached."
        else:
            warning = None

    print("[delta_sei_grad_debug]")
    print(f"  delta_sei_scaled.grad: {stats['delta_sei_scaled_grad']:.6g}")
    print(f"  delta_sei_est: {stats['delta_sei_est']:.6g}")
    print(f"  epsilon_e_n_est: {stats['epsilon_e_n_est']:.6g}")
    print(f"  nLi_sei_est: {stats['nLi_sei_est']:.6g}")
    print(f"  R_f_n_est: {stats['R_f_n_est']:.6g}")
    if warning:
        print(warning)
        stats['delta_sei_grad_warning'] = 1.0
    else:
        stats['delta_sei_grad_warning'] = 0.0
    return stats


def _bare_nn_lithium_inventory_losses(model, x_nn_raw, p, eps=1e-12, timer=None, sample_weights=None):
    if x_nn_raw.shape[0] == 0:
        zero = torch.zeros((), dtype=x_nn_raw.dtype, device=x_nn_raw.device)
        return zero, zero, {
            'nLi_s_mean': zero,
            'mean_pred_nLi_s': zero,
            'std_pred_nLi_s': zero,
            'nLi_s_est': zero,
            'nLi_s_rel_std': zero,
            'nLi_e_mean': zero,
            'mean_pred_nLi_e': zero,
            'std_pred_nLi_e': zero,
            'nLi_e_est': zero,
            'nLi_e_rel_std': zero,
        }

    timer = timer if timer is not None else TimingAccumulator(x_nn_raw.device, enabled=False)
    with timer.section('loss_li_s'):
        c_s_n_nn, c_e_nn, c_s_p_nn = model.unpack_state(x_nn_raw)
        c_n_nn, c_p_nn = model.calc_css_and_cs0(
            x_nn_raw,
            torch.zeros(x_nn_raw.shape[0], 1, dtype=x_nn_raw.dtype, device=x_nn_raw.device),
        )
        n_li_s_nn = model.calc_solid_lithium(c_n_nn, c_p_nn)
        loss_li_s, mean_s, std_s, est_s, rel_std_s = _relative_inventory_estimation_loss(
            n_li_s_nn, model.nLi_s_est, model.nLi_s_init, eps, sample_weights=sample_weights
        )
    with timer.section('loss_li_e'):
        n_li_e_nn = electrolyte_lithium_inventory(c_e_nn, p, model.electrolyte_dynamic(c_e_nn))
        loss_li_e, mean_e, std_e, est_e, rel_std_e = _relative_inventory_estimation_loss(
            n_li_e_nn, model.nLi_e_est, model.nLi_e_init, eps, sample_weights=sample_weights
        )
    stats = {
        'nLi_s_mean': mean_s,
        'mean_pred_nLi_s': mean_s,
        'std_pred_nLi_s': std_s,
        'nLi_s_est': est_s,
        'nLi_s_rel_std': rel_std_s,
        'nLi_e_mean': mean_e,
        'mean_pred_nLi_e': mean_e,
        'std_pred_nLi_e': std_e,
        'nLi_e_est': est_e,
        'nLi_e_rel_std': rel_std_e,
    }
    stats.update(_degradation_stats(model, model.cfg))
    return loss_li_s, loss_li_e, stats


def _integration_consistency_loss_from_predictions(
    model,
    x_roll_raw,
    x_nn_raw,
    p,
    window_stride,
    rollout_steps,
    sample_weights=None,
):
    pred_blocks = []
    roll_blocks = []
    weight_blocks = []
    n_samples = x_nn_raw.shape[0]
    max_offset = int(rollout_steps) // int(window_stride)
    for offset in range(1, max_offset + 1):
        step = offset * int(window_stride)
        if step >= x_roll_raw.shape[-1] or offset >= n_samples:
            break
        pred_blocks.append(x_nn_raw[offset:])
        roll_blocks.append(x_roll_raw[:-offset, :, step])
        if sample_weights is not None:
            weight_blocks.append(sample_weights[offset:])

    if not pred_blocks:
        zero = torch.zeros((), dtype=x_roll_raw.dtype, device=x_roll_raw.device)
        return zero, {'loss_integ_n': zero, 'loss_integ_e': zero, 'loss_integ_p': zero}

    shifted_pred_raw = torch.cat(pred_blocks, dim=0)
    integrated_raw = torch.cat(roll_blocks, dim=0)
    shifted_weights = torch.cat(weight_blocks, dim=0) if weight_blocks else None
    shifted_pred_norm = normalize_state(shifted_pred_raw, p)
    integrated_norm = normalize_state(integrated_raw, p)

    pred_n, pred_e, pred_p = model.unpack_state(shifted_pred_norm)
    integ_n, integ_e, integ_p = model.unpack_state(integrated_norm)
    loss_n = losses_spme.rmse_loss(pred_n, integ_n, sample_weights=shifted_weights)
    loss_e = losses_spme.rmse_loss(pred_e, integ_e, sample_weights=shifted_weights)
    loss_p = losses_spme.rmse_loss(pred_p, integ_p, sample_weights=shifted_weights)
    loss = (loss_n + loss_e + loss_p) / 3.0
    return loss, {'loss_integ_n': loss_n, 'loss_integ_e': loss_e, 'loss_integ_p': loss_p}


def _profile_integration_consistency_loss(model, x_roll_raw, x_nn_raw, dataset, p, sample_weights=None):
    losses = []
    losses_n = []
    losses_e = []
    losses_p = []
    weights = []
    window_stride = int(dataset.get('window_stride', 1))
    rollout_steps = int(dataset.get('pinn_rollout_steps', x_roll_raw.shape[-1] - 1))
    for profile_start, profile_end in dataset.get('profile_slices', [(0, x_nn_raw.shape[0])]):
        loss, stats = _integration_consistency_loss_from_predictions(
            model,
            x_roll_raw[profile_start:profile_end],
            x_nn_raw[profile_start:profile_end],
            p,
            window_stride,
            rollout_steps,
            sample_weights=(
                sample_weights[profile_start:profile_end]
                if sample_weights is not None else None
            ),
        )
        if torch.isfinite(loss):
            losses.append(loss)
            losses_n.append(stats['loss_integ_n'])
            losses_e.append(stats['loss_integ_e'])
            losses_p.append(stats['loss_integ_p'])
            if sample_weights is None:
                weights.append(profile_end - profile_start)
            else:
                weights.append(sample_weights[profile_start:profile_end].sum().detach())
    if not losses:
        zero = torch.zeros((), dtype=x_roll_raw.dtype, device=x_roll_raw.device)
        return zero, {'loss_integ_n': zero, 'loss_integ_e': zero, 'loss_integ_p': zero}
    if sample_weights is None:
        weight_tensor = torch.as_tensor(weights, dtype=x_roll_raw.dtype, device=x_roll_raw.device)
    else:
        weight_tensor = torch.stack(weights).to(dtype=x_roll_raw.dtype, device=x_roll_raw.device)

    def weighted(values):
        return (torch.stack(values) * weight_tensor).sum() / weight_tensor.sum()

    loss = weighted(losses)
    stats = {
        'loss_integ_n': weighted(losses_n),
        'loss_integ_e': weighted(losses_e),
        'loss_integ_p': weighted(losses_p),
    }
    return loss, stats


def _evaluate_nn(nn_models, model, dataset, cfg, p):
    with torch.no_grad():
        pred_raw = predict_spme_state_raw(nn_models, dataset['u'])
        target_raw = dataset['x_target_raw']
        loss_state, loss_c_s_n_norm, loss_c_e_norm, loss_c_s_p_norm = _nn_grouped_normalized_losses(
            pred_raw, target_raw, model, p
        )
        target_c_s_n_raw, target_c_e_raw, target_c_s_p_raw = model.unpack_state(target_raw)
        pred_c_s_n_raw, pred_c_e_raw, pred_c_s_p_raw = model.unpack_state(pred_raw)
        loss_c_s_n_raw = losses_spme.rmse_loss(pred_c_s_n_raw, target_c_s_n_raw)
        loss_c_e_raw = losses_spme.rmse_loss(pred_c_e_raw, target_c_e_raw)
        loss_c_s_p_raw = losses_spme.rmse_loss(pred_c_s_p_raw, target_c_s_p_raw)

        # SPMe physics expects raw concentration units.
        c_n, c_p = model.calc_css_and_cs0(pred_raw, dataset['i_target'])
        vt_pred = model.calc_voltage(
            c_n[:, -1],
            c_p[:, -1],
            pred_c_e_raw,
            dataset['i_target'],
            disable_electrolyte_ohmic=cfg.get('disable_electrolyte_ohmic', False),
            disable_electrolyte_concentration=cfg.get('disable_electrolyte_concentration', False),
        )
        loss_v = losses_spme.voltage_loss(vt_pred.reshape_as(dataset['v_target']), dataset['v_target'])

    return {
        'loss_state': loss_state.item(),
        'loss_v': loss_v.item(),
        'loss_c_s_n': loss_c_s_n_norm.item(),
        'loss_c_e': loss_c_e_norm.item(),
        'loss_c_s_p': loss_c_s_p_norm.item(),
        'loss_c_s_n_raw': loss_c_s_n_raw.item(),
        'loss_c_e_raw': loss_c_e_raw.item(),
        'loss_c_s_p_raw': loss_c_s_p_raw.item(),
    }


def _init_wandb(cfg, run_name):
    if not cfg.get('use_wandb', False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("use_wandb is true, but wandb is not installed. Run `pip install wandb`.") from exc

    wandb_dir = cfg.get('wandb_dir', os.path.join(cfg.get('save_dir', 'results'), 'wandb'))
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ.setdefault('WANDB_DIR', wandb_dir)
    os.environ.setdefault('WANDB_CACHE_DIR', os.path.join(wandb_dir, 'cache'))
    os.environ.setdefault('WANDB_CONFIG_DIR', os.path.join(wandb_dir, 'config'))

    return wandb.init(
        project=cfg.get('wandb_project', 'pinn-spme'),
        entity=cfg.get('wandb_entity', None),
        name=cfg.get('wandb_run_name', run_name),
        mode=cfg.get('wandb_mode', 'online'),
        dir=wandb_dir,
        config=cfg,
    )


def _predict_profile_windows(nn_models, model, profile, cfg, device):
    k = int(cfg['k'])
    h = int(cfg.get('h', 1))
    rollout_steps = _pinn_rollout_steps(cfg)
    length = len(profile['current'])
    if length < k:
        return None

    i_windows = _create_windows(profile['current'], k, [length], stride=1)
    v_windows = _create_windows(profile['voltage'], k, [length], stride=1)
    i_windows_t = torch.tensor(i_windows, dtype=torch.float32, device=device)
    v_windows_t = torch.tensor(v_windows, dtype=torch.float32, device=device)

    u = torch.zeros(i_windows_t.shape[0], k, model.p.input_size, device=device)
    u[:, :, 0] = i_windows_t
    u[:, :, 1] = v_windows_t
    ub = torch.tensor([cfg['OneC'] * cfg['max_Crate'], cfg['V_upper']], dtype=torch.float32, device=device)
    lb = torch.tensor([-cfg['OneC'] * cfg['max_Crate'], cfg['V_lower']], dtype=torch.float32, device=device)
    u = (u - lb) / (ub - lb)

    preds = []
    batch_size = int(cfg.get('plot_batch_size', 1024))
    with torch.no_grad():
        for start_idx in range(0, u.shape[0], batch_size):
            preds.append(predict_spme_state_raw(nn_models, u[start_idx:start_idx + batch_size]))
        x_nn_raw = torch.cat(preds, dim=0)
        target_idx = np.arange(k - 1, length)
        target_raw = torch.tensor(profile['state'][target_idx], dtype=torch.float32, device=device)
        i_target = torch.tensor(profile['current'][target_idx], dtype=torch.float32, device=device).unsqueeze(1)
        v_target = torch.tensor(profile['voltage'][target_idx], dtype=torch.float32, device=device).unsqueeze(1)

        # SPMe physics expects raw concentration units.
        pred_c_n, pred_c_p = model.calc_css_and_cs0(x_nn_raw, i_target)
        true_c_n, true_c_p = model.calc_css_and_cs0(target_raw, i_target)
        _, pred_c_e, _ = model.unpack_state(x_nn_raw)
        _, true_c_e, _ = model.unpack_state(target_raw)
        pred_dynamic = model.electrolyte_dynamic(pred_c_e)
        true_degradation = _true_degradation_for_plot(model, len(target_idx), true_c_e)
        true_dynamic = true_degradation['dynamic']
        pred_c_ex = reconstruct_full_concentration(pred_c_e, model.p, pred_dynamic)
        true_c_ex = reconstruct_full_concentration(true_c_e, model.p, true_dynamic)
        n_li_s_true = true_degradation['nLi_s']
        n_li_s_pred = model.calc_solid_lithium(pred_c_n, pred_c_p)
        n_li_e_true = true_degradation['nLi_e']
        n_li_e_pred = electrolyte_lithium_inventory(pred_c_e, model.p, pred_dynamic)
        n_li_sei_true = true_degradation['nLi_sei']
        n_li_sei_pred = model.nLi_sei_est.to(dtype=true_c_e.dtype, device=true_c_e.device).reshape(()).expand(len(target_idx))
        components_pred = model.calc_voltage_components(
            pred_c_n[:, -1],
            pred_c_p[:, -1],
            pred_c_e,
            i_target,
            disable_electrolyte_ohmic=cfg.get('disable_electrolyte_ohmic', False),
            disable_electrolyte_concentration=cfg.get('disable_electrolyte_concentration', False),
        )
        v_pred = components_pred['voltage'].reshape_as(v_target)
        components_true = model.calc_voltage_components(
            true_c_n[:, -1],
            true_c_p[:, -1],
            true_c_e,
            i_target,
            disable_electrolyte_ohmic=cfg.get('disable_electrolyte_ohmic', False),
            disable_electrolyte_concentration=cfg.get('disable_electrolyte_concentration', False),
            delta_sei_override=model.p.delta_sei_true,
        )

        x_roll_raw = torch.full_like(x_nn_raw, float('nan'))
        v_roll = torch.full_like(v_target, float('nan'))
        rollout_window = k + rollout_steps
        if cfg.get('method', '').lower() == 'pinn' and length >= rollout_window:
            i_windows_roll = _create_windows(profile['current'], rollout_window, [length], stride=1)
            v_windows_roll = _create_windows(profile['voltage'], rollout_window, [length], stride=1)
            i_windows_roll_t = torch.tensor(i_windows_roll, dtype=torch.float32, device=device)
            v_windows_roll_t = torch.tensor(v_windows_roll, dtype=torch.float32, device=device)

            u_roll = torch.zeros(i_windows_roll_t.shape[0], k, model.p.input_size, device=device)
            u_roll[:, :, 0] = i_windows_roll_t[:, :k]
            u_roll[:, :, 1] = v_windows_roll_t[:, :k]
            u_roll = (u_roll - lb) / (ub - lb)
            i_future_ave = _average_chunks(i_windows_roll_t[:, k:], h)
            i_future_sampled = torch.cat([
                i_windows_roll_t[:, k - 1:k],
                _sample_chunks(i_windows_roll_t[:, k:], h),
            ], dim=1)

            roll_sum = torch.zeros_like(x_nn_raw)
            roll_count = torch.zeros(x_nn_raw.shape[0], 1, dtype=x_nn_raw.dtype, device=device)
            v_roll_sum = torch.zeros_like(v_target)
            v_roll_count = torch.zeros_like(v_target)

            for start_idx in range(0, u_roll.shape[0], batch_size):
                end_idx = start_idx + batch_size
                u_batch = u_roll[start_idx:end_idx]
                i_future_ave_batch = i_future_ave[start_idx:end_idx]
                i_future_sampled_batch = i_future_sampled[start_idx:end_idx]
                x0_roll_raw = predict_spme_state_raw(nn_models, u_batch)
                # SPMe physics expects raw concentration units.
                x_roll_future_batch = model(x0_roll_raw, i_future_ave_batch)
                x_roll_seq_batch = torch.cat([x0_roll_raw.unsqueeze(-1), x_roll_future_batch], dim=-1)
                c_n_roll_seq, c_p_roll_seq = model.calc_css_and_cs0_seq(x_roll_seq_batch, i_future_sampled_batch)
                _, c_e_roll_seq, _ = model.unpack_state(x_roll_seq_batch)
                v_roll_seq_batch = model.calc_voltage(
                    c_n_roll_seq[:, -1, :],
                    c_p_roll_seq[:, -1, :],
                    c_e_roll_seq,
                    i_future_sampled_batch,
                    disable_electrolyte_ohmic=cfg.get('disable_electrolyte_ohmic', False),
                    disable_electrolyte_concentration=cfg.get('disable_electrolyte_concentration', False),
                )

                n_batch = x_roll_seq_batch.shape[0]
                for local_idx in range(n_batch):
                    global_start = start_idx + local_idx
                    for seq_idx in range(1, x_roll_seq_batch.shape[-1]):
                        absolute_idx = global_start + k - 1 + seq_idx * h
                        if absolute_idx >= length:
                            continue
                        aligned_idx = absolute_idx - (k - 1)
                        roll_sum[aligned_idx] += x_roll_seq_batch[local_idx, :, seq_idx]
                        roll_count[aligned_idx] += 1.0
                        v_roll_sum[aligned_idx] += v_roll_seq_batch[local_idx, seq_idx]
                        v_roll_count[aligned_idx] += 1.0

            valid_roll = roll_count.squeeze(1) > 0
            x_roll_raw[valid_roll] = roll_sum[valid_roll] / roll_count[valid_roll]
            valid_v_roll = v_roll_count.squeeze(1) > 0
            v_roll[valid_v_roll] = v_roll_sum[valid_v_roll] / v_roll_count[valid_v_roll]

        roll_c_n = roll_c_p = roll_c_ex = components_roll = None
        n_li_s_roll = n_li_e_roll = n_li_sei_roll = None
        valid_roll_state = torch.isfinite(x_roll_raw).all(dim=1)
        if valid_roll_state.any():
            i_roll_target = i_target[valid_roll_state]
            roll_c_n_valid, roll_c_p_valid = model.calc_css_and_cs0(x_roll_raw[valid_roll_state], i_roll_target)
            _, roll_c_e_valid, _ = model.unpack_state(x_roll_raw[valid_roll_state])
            roll_dynamic = model.electrolyte_dynamic(roll_c_e_valid)
            roll_c_ex_valid = reconstruct_full_concentration(roll_c_e_valid, model.p, roll_dynamic)
            n_li_s_roll_valid = model.calc_solid_lithium(roll_c_n_valid, roll_c_p_valid)
            n_li_e_roll_valid = electrolyte_lithium_inventory(roll_c_e_valid, model.p, roll_dynamic)
            components_roll_valid = model.calc_voltage_components(
                roll_c_n_valid[:, -1],
                roll_c_p_valid[:, -1],
                roll_c_e_valid,
                i_roll_target,
                disable_electrolyte_ohmic=cfg.get('disable_electrolyte_ohmic', False),
                disable_electrolyte_concentration=cfg.get('disable_electrolyte_concentration', False),
            )
            roll_c_n = torch.full_like(pred_c_n, float('nan'))
            roll_c_p = torch.full_like(pred_c_p, float('nan'))
            roll_c_ex = torch.full_like(pred_c_ex, float('nan'))
            n_li_s_roll = torch.full_like(n_li_s_pred, float('nan'))
            n_li_e_roll = torch.full_like(n_li_e_pred, float('nan'))
            n_li_sei_roll = torch.full_like(n_li_sei_pred, float('nan'))
            roll_c_n[valid_roll_state] = roll_c_n_valid
            roll_c_p[valid_roll_state] = roll_c_p_valid
            roll_c_ex[valid_roll_state] = roll_c_ex_valid
            n_li_s_roll[valid_roll_state] = n_li_s_roll_valid
            n_li_e_roll[valid_roll_state] = n_li_e_roll_valid
            n_li_sei_roll[valid_roll_state] = n_li_sei_pred[valid_roll_state]
            components_roll = {}
            for name, value in components_pred.items():
                filled = torch.full_like(value.reshape_as(v_target), float('nan'))
                filled[valid_roll_state] = components_roll_valid[name].reshape(-1, 1)
                components_roll[name] = filled

    return {
        'time': profile['time'][target_idx],
        'target_idx': target_idx,
        'v_true': v_target.detach().cpu().numpy().squeeze(1),
        'v_pred': v_pred.detach().cpu().numpy().squeeze(1),
        'v_roll': v_roll.detach().cpu().numpy().squeeze(1),
        'n_li_s_true': n_li_s_true.detach().cpu().numpy().reshape(-1),
        'n_li_s_pred': n_li_s_pred.detach().cpu().numpy().reshape(-1),
        'n_li_s_roll': None if n_li_s_roll is None else n_li_s_roll.detach().cpu().numpy().reshape(-1),
        'n_li_e_true': n_li_e_true.detach().cpu().numpy().reshape(-1),
        'n_li_e_pred': n_li_e_pred.detach().cpu().numpy().reshape(-1),
        'n_li_e_roll': None if n_li_e_roll is None else n_li_e_roll.detach().cpu().numpy().reshape(-1),
        'n_li_sei_true': n_li_sei_true.detach().cpu().numpy().reshape(-1),
        'n_li_sei_pred': n_li_sei_pred.detach().cpu().numpy().reshape(-1),
        'n_li_sei_roll': None if n_li_sei_roll is None else n_li_sei_roll.detach().cpu().numpy().reshape(-1),
        'components_true': {name: value.detach().cpu().numpy().reshape(-1) for name, value in components_true.items()},
        'components_pred': {name: value.detach().cpu().numpy().reshape(-1) for name, value in components_pred.items()},
        'components_roll': None if components_roll is None else {
            name: value.detach().cpu().numpy().reshape(-1) for name, value in components_roll.items()
        },
        'c_n_true': true_c_n.detach().cpu().numpy(),
        'c_n_pred': pred_c_n.detach().cpu().numpy(),
        'c_n_roll': None if roll_c_n is None else roll_c_n.detach().cpu().numpy(),
        'c_p_true': true_c_p.detach().cpu().numpy(),
        'c_p_pred': pred_c_p.detach().cpu().numpy(),
        'c_p_roll': None if roll_c_p is None else roll_c_p.detach().cpu().numpy(),
        'c_ex_true': true_c_ex.detach().cpu().numpy(),
        'c_ex_pred': pred_c_ex.detach().cpu().numpy(),
        'c_ex_roll': None if roll_c_ex is None else roll_c_ex.detach().cpu().numpy(),
    }


def _log_profile_predictions(run, nn_models, model, datasets, epoch, cfg):
    if run is None:
        return

    import matplotlib.pyplot as plt
    import wandb

    n_samples = int(cfg.get('plot_samples', cfg.get('wandb_plot_samples', 2)))
    log_data = {}
    pred_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    actual_colors = [
        '#111111', '#8c564b', '#7f7f7f', '#bcbd22', '#17becf',
        '#9467bd', '#e377c2', '#393b79', '#637939', '#8c6d31',
    ]

    def plot_pair(ax, x, true_y, pred_y, label, color_idx):
        actual_color = actual_colors[color_idx % len(actual_colors)]
        pred_color = pred_colors[color_idx % len(pred_colors)]
        ax.plot(x, true_y, linestyle=':', linewidth=1.8, color=actual_color, label=f'{label} actual')
        ax.plot(x, pred_y, linestyle='--', linewidth=1.5, color=pred_color, label=f'predicted {label}')

    def plot_pair_with_colors(ax, x, true_y, pred_y, label, actual_color, pred_color):
        ax.plot(x, true_y, linestyle=':', linewidth=1.8, color=actual_color, label=f'{label} actual')
        ax.plot(x, pred_y, linestyle='--', linewidth=1.5, color=pred_color, label=f'predicted {label}')

    for split_name, dataset in datasets.items():
        for profile in dataset.get('profiles', []):
            result = _predict_profile_windows(nn_models, model, profile, cfg, dataset['u'].device)
            if result is None:
                continue

            profile_key = os.path.splitext(profile['name'])[0]
            count = min(n_samples, len(result['time']))
            sample_idx = np.linspace(0, len(result['time']) - 1, count, dtype=int)

            r_axis = np.linspace(0.0, 1.0, model.n_r + 1)
            x_n = np.arange(0, model.p.Nxn + 1)
            x_s = np.arange(model.p.Nxn, model.p.Nxn + model.p.Nxs + 1)
            x_p = np.arange(model.p.Nxn + model.p.Nxs, model.p.Nx + 1)

            t = result['time']
            solid_locs = {
                'center': 0,
                'mid': model.n_r // 2,
                'surface': -1,
            }
            e_locs_n = {
                'cc_n': 0,
                'mid_n': model.p.Nxn // 2,
                'n_s': model.p.Nxn,
            }
            e_locs_s = {
                'n_s': model.p.Nxn,
                'mid_s': model.p.Nxn + model.p.Nxs // 2,
                's_p': model.p.Nxn + model.p.Nxs,
            }
            e_locs_p = {
                's_p': model.p.Nxn + model.p.Nxs,
                'mid_p': model.p.Nxn + model.p.Nxs + model.p.Nxp // 2,
                'cc_p': -1,
            }

            def plot_spatial(kind, c_n_key, c_ex_key, c_p_key, indices):
                if len(indices) == 0:
                    return
                local_count = len(indices)
                fig, axes = plt.subplots(5, local_count, figsize=(3.8 * local_count, 12.0), squeeze=False)
                for col, idx in enumerate(indices):
                    plot_pair(axes[0, col], r_axis, result['c_n_true'][idx], result[c_n_key][idx], kind, 0)
                    axes[0, col].set_title(f't={result["time"][idx]:.0f}s c_s_n')
                    plot_pair(
                        axes[1, col], x_n, result['c_ex_true'][idx, :model.p.Nxn + 1],
                        result[c_ex_key][idx, :model.p.Nxn + 1], kind, 1
                    )
                    axes[1, col].set_title('c_e negative')
                    plot_pair(
                        axes[2, col], x_s,
                        result['c_ex_true'][idx, model.p.Nxn:model.p.Nxn + model.p.Nxs + 1],
                        result[c_ex_key][idx, model.p.Nxn:model.p.Nxn + model.p.Nxs + 1],
                        kind, 2
                    )
                    axes[2, col].set_title('c_e separator')
                    plot_pair(
                        axes[3, col], x_p, result['c_ex_true'][idx, model.p.Nxn + model.p.Nxs:],
                        result[c_ex_key][idx, model.p.Nxn + model.p.Nxs:], kind, 3
                    )
                    axes[3, col].set_title('c_e positive')
                    plot_pair(axes[4, col], r_axis, result['c_p_true'][idx], result[c_p_key][idx], kind, 4)
                    axes[4, col].set_title('c_s_p')
                    for row in range(5):
                        axes[row, col].grid(True, alpha=0.25)
                        if col == local_count - 1:
                            axes[row, col].legend(loc='best')
                fig.tight_layout()
                log_data[f'{split_name}/{profile_key}/spatial_profiles_{kind}'] = wandb.Image(fig)
                plt.close(fig)

            def plot_time_series(kind, c_n_key, c_ex_key, c_p_key, v_key, n_li_s_key, n_li_e_key, n_li_sei_key, mask):
                if not np.any(mask):
                    return
                def draw_time_series(axes, local_mask, x_label=False):
                    t_plot = t[local_mask]
                    for color_idx, (label, idx) in enumerate(solid_locs.items()):
                        plot_pair(axes[0], t_plot, result['c_n_true'][local_mask, idx], result[c_n_key][local_mask, idx], label, color_idx)
                        plot_pair(axes[1], t_plot, result['c_p_true'][local_mask, idx], result[c_p_key][local_mask, idx], label, color_idx)
                    for color_idx, (label, idx) in enumerate(e_locs_n.items()):
                        plot_pair(axes[2], t_plot, result['c_ex_true'][local_mask, idx], result[c_ex_key][local_mask, idx], label, color_idx)
                    for color_idx, (label, idx) in enumerate(e_locs_s.items()):
                        plot_pair(axes[3], t_plot, result['c_ex_true'][local_mask, idx], result[c_ex_key][local_mask, idx], label, color_idx)
                    for color_idx, (label, idx) in enumerate(e_locs_p.items()):
                        plot_pair(axes[4], t_plot, result['c_ex_true'][local_mask, idx], result[c_ex_key][local_mask, idx], label, color_idx)
                    plot_pair(axes[5], t_plot, result['v_true'][local_mask], result[v_key][local_mask], 'voltage', 0)
                    axes[6].plot(
                        t_plot, result[v_key][local_mask] - result['v_true'][local_mask],
                        linestyle='--', linewidth=1.5, color=pred_colors[1], label=f'predicted {kind} error'
                    )
                    plot_pair(axes[7], t_plot, result['n_li_s_true'][local_mask], result[n_li_s_key][local_mask], 'solid Li inventory', 2)
                    plot_pair(axes[8], t_plot, result['n_li_e_true'][local_mask], result[n_li_e_key][local_mask], 'electrolyte Li inventory', 3)
                    plot_pair(axes[9], t_plot, result['n_li_sei_true'][local_mask], result[n_li_sei_key][local_mask], 'SEI Li inventory', 4)
                    labels = ['c_s_n', 'c_s_p', 'c_e neg', 'c_e sep', 'c_e pos', 'V', 'V error', 'nLi solid', 'nLi electrolyte', 'nLi SEI']
                    for row_idx, ax in enumerate(axes):
                        ax.set_ylabel(labels[row_idx])
                        ax.grid(True, alpha=0.25)
                        ax.legend(loc='best', ncol=2)
                    if x_label:
                        axes[-1].set_xlabel('time [s]')

                t_plot = t[mask]
                fig, axes = plt.subplots(10, 1, figsize=(12, 24), sharex=True)
                draw_time_series(axes, mask, x_label=True)
                fig.tight_layout()
                log_data[f'{split_name}/{profile_key}/time_series_{kind}'] = wandb.Image(fig)
                plt.close(fig)

                first_mask = mask & (t <= t_plot[0] + 500.0)
                last_mask = mask & (t >= t_plot[-1] - 500.0)
                fig, axes = plt.subplots(10, 2, figsize=(18, 24), sharex='col', squeeze=False)
                if np.any(first_mask):
                    draw_time_series(axes[:, 0], first_mask, x_label=True)
                    axes[0, 0].set_title('first 500 s')
                if np.any(last_mask):
                    draw_time_series(axes[:, 1], last_mask, x_label=True)
                    axes[0, 1].set_title('last 500 s')
                fig.tight_layout()
                log_data[f'{split_name}/{profile_key}/time_series_edges_{kind}'] = wandb.Image(fig)
                plt.close(fig)

            def plot_voltage_components(kind, components_key, mask):
                components = result[components_key]
                if components is None or not np.any(mask):
                    return

                def draw_voltage_components(axes, local_mask, x_label=False):
                    t_plot = t[local_mask]
                    for row_idx, (ax, (key, label)) in enumerate(zip(axes, rows)):
                        if key == 'film_drop':
                            plot_pair_with_colors(
                                ax,
                                t_plot,
                                true_components[key][local_mask],
                                components[key][local_mask],
                                label,
                                actual_color='#000000',
                                pred_color='#d62728',
                            )
                        else:
                            plot_pair(ax, t_plot, true_components[key][local_mask], components[key][local_mask], label, row_idx)
                        ax.set_ylabel(label)
                        ax.grid(True, alpha=0.25)
                        ax.legend(loc='best')
                    if x_label:
                        axes[-1].set_xlabel('time [s]')

                t_plot = t[mask]
                true_components = result['components_true']
                rows = [
                    ('U_p', 'U_p'),
                    ('U_n', 'U_n'),
                    ('eta_p', 'eta_p'),
                    ('eta_n', 'eta_n'),
                    ('electrolyte_ohmic', 'elyte ohmic'),
                    ('electrolyte_concentration', 'elyte conc'),
                    ('film_drop', 'film drop'),
                ]
                fig, axes = plt.subplots(len(rows), 1, figsize=(12, 18), sharex=True)
                draw_voltage_components(axes, mask, x_label=True)
                fig.tight_layout()
                log_data[f'{split_name}/{profile_key}/voltage_components_{kind}'] = wandb.Image(fig)
                plt.close(fig)

                first_mask = mask & (t <= t_plot[0] + 500.0)
                last_mask = mask & (t >= t_plot[-1] - 500.0)
                fig, axes = plt.subplots(len(rows), 2, figsize=(18, 18), sharex='col', squeeze=False)
                if np.any(first_mask):
                    draw_voltage_components(axes[:, 0], first_mask, x_label=True)
                    axes[0, 0].set_title('first 500 s')
                if np.any(last_mask):
                    draw_voltage_components(axes[:, 1], last_mask, x_label=True)
                    axes[0, 1].set_title('last 500 s')
                fig.tight_layout()
                log_data[f'{split_name}/{profile_key}/voltage_components_edges_{kind}'] = wandb.Image(fig)
                plt.close(fig)

            bare_mask = np.ones(len(result['time']), dtype=bool)
            plot_spatial('bare_nn', 'c_n_pred', 'c_ex_pred', 'c_p_pred', sample_idx)
            plot_time_series('bare_nn', 'c_n_pred', 'c_ex_pred', 'c_p_pred', 'v_pred', 'n_li_s_pred', 'n_li_e_pred', 'n_li_sei_pred', bare_mask)
            plot_voltage_components('bare_nn', 'components_pred', bare_mask)

            if result['c_n_roll'] is not None:
                roll_mask = (
                    np.isfinite(result['c_n_roll']).all(axis=1)
                    & np.isfinite(result['c_ex_roll']).all(axis=1)
                    & np.isfinite(result['c_p_roll']).all(axis=1)
                    & np.isfinite(result['v_roll'])
                    & np.isfinite(result['n_li_s_roll'])
                    & np.isfinite(result['n_li_e_roll'])
                    & np.isfinite(result['n_li_sei_roll'])
                )
            else:
                roll_mask = np.zeros(len(result['time']), dtype=bool)
            if np.any(roll_mask):
                roll_indices_all = np.flatnonzero(roll_mask)
                roll_count = min(n_samples, len(roll_indices_all))
                roll_sample_idx = roll_indices_all[np.linspace(0, len(roll_indices_all) - 1, roll_count, dtype=int)]
                plot_spatial('rolled_spme', 'c_n_roll', 'c_ex_roll', 'c_p_roll', roll_sample_idx)
                plot_time_series('rolled_spme', 'c_n_roll', 'c_ex_roll', 'c_p_roll', 'v_roll', 'n_li_s_roll', 'n_li_e_roll', 'n_li_sei_roll', roll_mask)
                plot_voltage_components('rolled_spme', 'components_roll', roll_mask)

    if log_data:
        run.log(log_data, step=epoch)


def _log_degradation_history_plot(run, history, epoch, cfg):
    if not history:
        return
    required = ('delta_sei_est_epoch', 'epsilon_e_n_est_epoch', 'nLi_sei_est_epoch')
    if not all(required_key in history[-1] for required_key in required):
        return

    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    if run is not None:
        import wandb

    epochs = np.asarray([row['epoch'] for row in history], dtype=float)
    delta_est = np.asarray([row.get('delta_sei_est_epoch', np.nan) for row in history], dtype=float)
    delta_true = np.asarray([row.get('delta_sei_true_epoch', np.nan) for row in history], dtype=float)
    eps_est = np.asarray([row.get('epsilon_e_n_est_epoch', np.nan) for row in history], dtype=float)
    eps_true = np.asarray([row.get('epsilon_e_n_true_epoch', np.nan) for row in history], dtype=float)
    nli_sei_est = np.asarray([row.get('nLi_sei_est_epoch', np.nan) for row in history], dtype=float)
    nli_sei_true = np.asarray([row.get('nLi_sei_true_epoch', np.nan) for row in history], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(epochs, delta_true, linestyle=':', linewidth=1.8, color='#111111', label='delta_sei actual')
    axes[0].plot(epochs, delta_est, linestyle='--', linewidth=1.5, color='#1f77b4', label='predicted delta_sei')
    axes[0].set_ylabel('delta_sei [m]')
    axes[1].plot(epochs, eps_true, linestyle=':', linewidth=1.8, color='#111111', label='epsilon_e_n actual')
    axes[1].plot(epochs, eps_est, linestyle='--', linewidth=1.5, color='#ff7f0e', label='predicted epsilon_e_n')
    axes[1].set_ylabel('epsilon_e_n')
    axes[2].plot(epochs, nli_sei_true, linestyle=':', linewidth=1.8, color='#111111', label='nLi_sei actual')
    axes[2].plot(epochs, nli_sei_est, linestyle='--', linewidth=1.5, color='#2ca02c', label='predicted nLi_sei')
    axes[2].set_ylabel('nLi_sei [mol]')
    axes[2].set_xlabel('epoch')
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best')
    fig.tight_layout()
    save_dir = cfg.get('save_dir', 'results')
    os.makedirs(save_dir, exist_ok=True)
    latest_path = os.path.join(save_dir, 'degradation_history_latest.png')
    epoch_path = os.path.join(save_dir, f'degradation_history_epoch_{int(epoch):06d}.png')
    fig.savefig(latest_path, dpi=150)
    fig.savefig(epoch_path, dpi=150)
    if run is not None:
        run.log({'degradation/epoch_history': wandb.Image(fig)}, step=epoch)
    plt.close(fig)


def train_nn(cfg, device, p):
    os.makedirs(cfg.get('save_dir', 'results'), exist_ok=True)
    start_time = time.time()

    train_data, val_data = _load_prepared_nn_data(cfg, device, p)
    nn_models = create_spme_nn_models(cfg, p).to(device)
    model = SPMeModel(p, cfg).to(device)
    optimizers = [optim.Adam(nn_model.parameters(), lr=cfg['lrate']) for nn_model in nn_models]

    n_samples = train_data['u'].shape[0]
    batch_size = cfg.get('batch_size', n_samples)
    start_epoch, best_val, history = _resume_from_checkpoint(cfg, nn_models, optimizers, device)
    if start_epoch == 0:
        _load_pretrained_nn_models(cfg, nn_models, device)
    run = _init_wandb(cfg, 'spme-nn')
    plot_every = int(cfg.get('wandb_plot_every', cfg.get('print_every', 50)))
    if run is not None and start_epoch == 0:
        _log_profile_predictions(
            run,
            nn_models,
            model,
            {'train': train_data, 'val': val_data},
            0,
            cfg,
        )

    print('Start SPMe NN state training')
    try:
        for epoch in range(start_epoch, cfg['epochs']):
            epoch_display = epoch + 1
            permutation = torch.randperm(n_samples, device=device)
            epoch_loss = 0.0

            for batch_start in range(0, n_samples, batch_size):
                idx = permutation[batch_start:batch_start + batch_size]
                pred_raw = predict_spme_state_raw(nn_models, train_data['u'][idx])
                target_raw = train_data['x_target_raw'][idx]
                loss, _, _, _ = _nn_grouped_normalized_losses(pred_raw, target_raw, model, p)

                for optimizer in optimizers:
                    optimizer.zero_grad()
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                epoch_loss += loss.item() * len(idx)

            if epoch >= cfg['decay_step']:
                for optimizer in optimizers:
                    for group in optimizer.param_groups:
                        group['lr'] *= cfg['decay_rate']

            epoch_loss /= n_samples
            metrics_train = _evaluate_nn(nn_models, model, train_data, cfg, p)
            metrics_val = _evaluate_nn(nn_models, model, val_data, cfg, p)
            history.append({
                'epoch': epoch_display,
                'loss_total': epoch_loss,
                'train_loss_total_eval': metrics_train['loss_state'],
                'train_loss_v': metrics_train['loss_v'],
                'train_loss_c_s_n': metrics_train['loss_c_s_n'],
                'train_loss_c_e': metrics_train['loss_c_e'],
                'train_loss_c_s_p': metrics_train['loss_c_s_p'],
                'train_loss_c_s_n_raw': metrics_train['loss_c_s_n_raw'],
                'train_loss_c_e_raw': metrics_train['loss_c_e_raw'],
                'train_loss_c_s_p_raw': metrics_train['loss_c_s_p_raw'],
                'val_loss_state': metrics_val['loss_state'],
                'val_loss_v': metrics_val['loss_v'],
                'val_loss_c_s_n': metrics_val['loss_c_s_n'],
                'val_loss_c_e': metrics_val['loss_c_e'],
                'val_loss_c_s_p': metrics_val['loss_c_s_p'],
                'val_loss_c_s_n_raw': metrics_val['loss_c_s_n_raw'],
                'val_loss_c_e_raw': metrics_val['loss_c_e_raw'],
                'val_loss_c_s_p_raw': metrics_val['loss_c_s_p_raw'],
            })

            log_payload = {
                'epoch': epoch_display,
                'train/loss_total': epoch_loss,
                'train/loss_total_eval': metrics_train['loss_state'],
                'train/voltage_rmse': metrics_train['loss_v'],
                'train/loss_c_s_n': metrics_train['loss_c_s_n'],
                'train/loss_c_e': metrics_train['loss_c_e'],
                'train/loss_c_s_p': metrics_train['loss_c_s_p'],
                'train/c_s_n_rmse_raw': metrics_train['loss_c_s_n_raw'],
                'train/c_e_rmse_raw': metrics_train['loss_c_e_raw'],
                'train/c_s_p_rmse_raw': metrics_train['loss_c_s_p_raw'],
                'val/loss_total': metrics_val['loss_state'],
                'val/voltage_rmse': metrics_val['loss_v'],
                'val/loss_c_s_n': metrics_val['loss_c_s_n'],
                'val/loss_c_e': metrics_val['loss_c_e'],
                'val/loss_c_s_p': metrics_val['loss_c_s_p'],
                'val/c_s_n_rmse_raw': metrics_val['loss_c_s_n_raw'],
                'val/c_e_rmse_raw': metrics_val['loss_c_e_raw'],
                'val/c_s_p_rmse_raw': metrics_val['loss_c_s_p_raw'],
                'learning_rate_n': optimizers[0].param_groups[0]['lr'],
                'learning_rate_e': optimizers[1].param_groups[0]['lr'],
                'learning_rate_p': optimizers[2].param_groups[0]['lr'],
            }
            if run is not None:
                run.log(log_payload, step=epoch + 1)

            if epoch_display == 1 or _is_due_epoch(epoch_display, int(cfg.get('print_every', 50))):
                print(f"Epoch {epoch_display:5d} | loss {epoch_loss:.6f} | "
                      f"val {metrics_val['loss_state']:.6f} | val V {metrics_val['loss_v']:.6f} | "
                      f"n {metrics_val['loss_c_s_n']:.6f} | e {metrics_val['loss_c_e']:.6f} | "
                      f"p {metrics_val['loss_c_s_p']:.6f}")

            if run is not None and (epoch_display == 1 or _is_due_epoch(epoch_display, plot_every)):
                _log_profile_predictions(
                    run,
                    nn_models,
                    model,
                    {'train': train_data, 'val': val_data},
                    epoch_display,
                    cfg,
                )

            if metrics_val['loss_state'] < best_val:
                best_val = metrics_val['loss_state']
                _save_spme_nn_models(nn_models, cfg)
                _save_best_spme_nn_models(nn_models, cfg)

            _save_periodic_checkpoint(epoch, nn_models, optimizers, best_val, history, cfg)

        pd.DataFrame(history).to_csv(os.path.join(cfg.get('save_dir', 'results'), cfg['history_save_name']), index=False)
        elapsed = time.time() - start_time
        print(f'NN training time: {elapsed:.2f} sec')
    except KeyboardInterrupt:
        interrupted_epoch = epoch if 'epoch' in locals() else start_epoch
        _save_interrupt_checkpoint(interrupted_epoch, nn_models, optimizers, best_val, history, cfg)
        print(f"Interrupted. Saved checkpoint at epoch {interrupted_epoch + 1}.")
    finally:
        if run is not None:
            run.finish()



def train(cfg, device, p):
    os.makedirs(cfg.get('save_dir', 'results'), exist_ok=True)
    start_time = time.time()

    train_data, val_data = _load_prepared_pinn_data(cfg, device, p)

    nn_models = create_spme_nn_models(cfg, p).to(device)
    model = SPMeModel(p, cfg).to(device)
    optimizers = [optim.Adam(nn_model.parameters(), lr=cfg['lrate']) for nn_model in nn_models]
    degradation_params = [model.delta_sei_scaled] if model.delta_sei_scaled.requires_grad else []
    if degradation_params:
        optimizers.append(optim.Adam(degradation_params, lr=cfg.get('lrate_sei', cfg.get('lrate_degradation', cfg['lrate']))))

    n_samples = train_data['u'].shape[0]
    batch_size = cfg.get('batch_size', n_samples)
    eval_every = int(cfg.get('eval_every', 50))
    full_eval_every = int(cfg.get('full_eval_every', 500))
    plot_every = int(cfg.get('wandb_plot_every', full_eval_every))
    profile_timing = bool(cfg.get('profile_timing', False))
    profile_timing_every = int(cfg.get('profile_timing_every', 20))
    profile_timing_warmup_epochs = int(cfg.get('profile_timing_warmup_epochs', 5))
    profile_timing_log_wandb = bool(cfg.get('profile_timing_log_wandb', True))
    val_eval_subset = _validation_eval_subset(val_data, cfg)
    start_epoch, best_val, history = _resume_from_checkpoint(cfg, nn_models, optimizers, device, model=model)
    if start_epoch == 0:
        _load_pretrained_nn_models(cfg, nn_models, device)
    run = _init_wandb(cfg, 'spme-pinn')

    if start_epoch == 0:
        warm_train = _warm_start_diagnostics(nn_models, model, train_data, cfg, p, 'train')
        warm_val = _warm_start_diagnostics(nn_models, model, val_data, cfg, p, 'val')
        warm_metrics = {**warm_train, **warm_val}
        if warm_metrics:
            print("Epoch 0 warm-start diagnostics:")
            for key, value in warm_metrics.items():
                print(f"  {key}: {value:.6g}")
            if run is not None:
                run.log(warm_metrics, step=0)
        if run is not None:
            _log_profile_predictions(run, nn_models, model, {'train': train_data, 'val': val_data}, 0, cfg)

    print('Start SPMe PINN training')
    try:
        for epoch in range(start_epoch, cfg['epochs']):
            epoch_display = epoch + 1
            timing_enabled = profile_timing and epoch_display > profile_timing_warmup_epochs
            timer = TimingAccumulator(device, enabled=timing_enabled)
            model.reset_timing(enabled=timing_enabled)
            with timer.section('epoch_total'):
                epoch_loss = 0.0
                epoch_loss_v = 0.0
                epoch_loss_li_s = 0.0
                epoch_loss_li_e = 0.0
                epoch_loss_integ = 0.0
                epoch_loss_integ_n = 0.0
                epoch_loss_integ_e = 0.0
                epoch_loss_integ_p = 0.0
                epoch_nli_s_mean = 0.0
                epoch_nli_s_std = 0.0
                epoch_nli_s_rel_std = 0.0
                epoch_nli_e_mean = 0.0
                epoch_nli_e_std = 0.0
                epoch_nli_e_rel_std = 0.0
                epoch_weight = 0.0
                delta_sei_grad_debug_stats = None
                delta_sei_grad_debug_done = False

                for batch_slice in _iter_ordered_profile_batches(
                    train_data,
                    batch_size,
                    shuffle_profiles=cfg.get('shuffle_pinn_profiles', True),
                    seed=cfg.get('seed', 0),
                    epoch=epoch,
                ):
                    with timer.section('batch_total'):
                        with timer.section('data_loading'):
                            u_batch = train_data['u'][batch_slice]
                            i_future_ave_batch = train_data['i_future_ave'][batch_slice]
                            i_future_sampled_batch = train_data['i_future_sampled'][batch_slice]
                            v_target_batch = train_data['v_future_sampled'][batch_slice]
                            weights_batch = train_data['window_weights'][batch_slice]

                        with timer.section('nn_forward'):
                            x0_raw = predict_spme_state_raw(nn_models, u_batch)

                        # SPMe physics expects raw concentration units.
                        with timer.section('rollout'):
                            x_roll_future = model(x0_raw, i_future_ave_batch)
                            x_roll_raw = torch.cat([x0_raw.unsqueeze(-1), x_roll_future], dim=-1)

                        with timer.section('loss_voltage'):
                            c_s_n_seq, c_e_seq, c_s_p_seq = model.unpack_state(x_roll_raw)
                            c_n_seq, c_p_seq = model.calc_css_and_cs0_seq(x_roll_raw, i_future_sampled_batch)
                            vt_pred = model.calc_voltage(
                                c_n_seq[:, -1, :],
                                c_p_seq[:, -1, :],
                                c_e_seq,
                                i_future_sampled_batch,
                                disable_electrolyte_ohmic=cfg.get('disable_electrolyte_ohmic', False),
                                disable_electrolyte_concentration=cfg.get('disable_electrolyte_concentration', False),
                            )
                            loss_v = losses_spme.voltage_loss(
                                vt_pred,
                                v_target_batch,
                                sample_weights=weights_batch,
                            )

                        loss_li_s, loss_li_e, li_stats = _bare_nn_lithium_inventory_losses(
                            model,
                            x0_raw,
                            p,
                            timer=timer,
                            sample_weights=weights_batch,
                        )
                        w_integ = cfg.get('w_integ', 1.0)
                        with timer.section('loss_integ'):
                            if w_integ:
                                loss_integ, integ_stats = _integration_consistency_loss_from_predictions(
                                    model,
                                    x_roll_raw,
                                    x0_raw,
                                    p,
                                    int(train_data.get('window_stride', cfg.get('train_window_stride', 1))),
                                    int(train_data.get('pinn_rollout_steps', x_roll_raw.shape[-1] - 1)),
                                    sample_weights=weights_batch,
                                )
                            else:
                                loss_integ = torch.zeros((), dtype=x_roll_raw.dtype, device=x_roll_raw.device)
                                integ_stats = {'loss_integ_n': loss_integ, 'loss_integ_e': loss_integ, 'loss_integ_p': loss_integ}

                        with timer.section('loss_assembly'):
                            loss = (cfg['w_v'] * loss_v + cfg['w_li_s'] * loss_li_s + cfg['w_li_e'] * loss_li_e
                                    + w_integ * loss_integ)

                        for optimizer in optimizers:
                            optimizer.zero_grad()
                        with timer.section('backward'):
                            loss.backward()
                        if (cfg.get('estimate_delta_sei', False)
                                and cfg.get('debug_delta_sei_grad', False)
                                and _is_due_epoch(epoch_display, int(cfg.get('debug_delta_sei_grad_every', 20)))
                                and not delta_sei_grad_debug_done):
                            delta_sei_grad_debug_stats = _delta_sei_grad_diagnostics(model)
                            delta_sei_grad_debug_done = True
                        with timer.section('optimizer'):
                            for optimizer in optimizers:
                                optimizer.step()
                            model.clamp_inventory_parameters()

                        batch_weight = float(weights_batch.sum().detach().cpu())
                        epoch_weight += batch_weight
                        epoch_loss += loss.item() * batch_weight
                        epoch_loss_v += loss_v.item() * batch_weight
                        epoch_loss_li_s += loss_li_s.item() * batch_weight
                        epoch_loss_li_e += loss_li_e.item() * batch_weight
                        epoch_loss_integ += loss_integ.item() * batch_weight
                        epoch_loss_integ_n += integ_stats['loss_integ_n'].item() * batch_weight
                        epoch_loss_integ_e += integ_stats['loss_integ_e'].item() * batch_weight
                        epoch_loss_integ_p += integ_stats['loss_integ_p'].item() * batch_weight
                        epoch_nli_s_mean += li_stats['nLi_s_mean'].item() * batch_weight
                        epoch_nli_s_std += li_stats['std_pred_nLi_s'].item() * batch_weight
                        epoch_nli_s_rel_std += li_stats['nLi_s_rel_std'].item() * batch_weight
                        epoch_nli_e_mean += li_stats['nLi_e_mean'].item() * batch_weight
                        epoch_nli_e_std += li_stats['std_pred_nLi_e'].item() * batch_weight
                        epoch_nli_e_rel_std += li_stats['nLi_e_rel_std'].item() * batch_weight

                if epoch >= cfg['decay_step']:
                    for optimizer in optimizers:
                        for group in optimizer.param_groups:
                            group['lr'] *= cfg['decay_rate']

            epoch_denominator = epoch_weight if epoch_weight > 0.0 else float(n_samples)
            epoch_loss /= epoch_denominator
            epoch_loss_v /= epoch_denominator
            epoch_loss_li_s /= epoch_denominator
            epoch_loss_li_e /= epoch_denominator
            epoch_loss_integ /= epoch_denominator
            epoch_loss_integ_n /= epoch_denominator
            epoch_loss_integ_e /= epoch_denominator
            epoch_loss_integ_p /= epoch_denominator
            epoch_nli_s_mean /= epoch_denominator
            epoch_nli_s_std /= epoch_denominator
            epoch_nli_s_rel_std /= epoch_denominator
            epoch_nli_e_mean /= epoch_denominator
            epoch_nli_e_std /= epoch_denominator
            epoch_nli_e_rel_std /= epoch_denominator
            degradation_epoch = _float_stats(_degradation_stats(model, cfg))
            nli_s_est_epoch = degradation_epoch['nLi_s_est']
            nli_e_est_epoch = degradation_epoch['nLi_e_est']

            history_row = {
                'epoch': epoch_display,
                'loss': epoch_loss,
                'train_batch_loss_v': epoch_loss_v,
                'train_batch_loss_li_s': epoch_loss_li_s,
                'train_batch_loss_li_e': epoch_loss_li_e,
                'train_batch_loss_integ': epoch_loss_integ,
                'train_batch_loss_integ_n': epoch_loss_integ_n,
                'train_batch_loss_integ_e': epoch_loss_integ_e,
                'train_batch_loss_integ_p': epoch_loss_integ_p,
                'train_batch_nLi_s_mean': epoch_nli_s_mean,
                'train_batch_mean_pred_nLi_s': epoch_nli_s_mean,
                'train_batch_std_pred_nLi_s': epoch_nli_s_std,
                'nLi_s_est_epoch': nli_s_est_epoch,
                'train_batch_nLi_s_rel_std': epoch_nli_s_rel_std,
                'train_batch_nLi_e_mean': epoch_nli_e_mean,
                'train_batch_mean_pred_nLi_e': epoch_nli_e_mean,
                'train_batch_std_pred_nLi_e': epoch_nli_e_std,
                'nLi_e_est_epoch': nli_e_est_epoch,
                'train_batch_nLi_e_rel_std': epoch_nli_e_rel_std,
            }
            history_row.update({f'{key}_epoch': value for key, value in degradation_epoch.items()})
            if delta_sei_grad_debug_stats is not None:
                history_row.update({
                    f'debug_{key}_epoch': value for key, value in delta_sei_grad_debug_stats.items()
                })

            log_payload = {
                'epoch': epoch_display,
                'train/loss': epoch_loss,
                'train/batch_loss_voltage': epoch_loss_v,
                'train/batch_loss_li_s': epoch_loss_li_s,
                'train/batch_loss_li_e': epoch_loss_li_e,
                'train/batch_loss_integ': epoch_loss_integ,
                'train/batch_loss_integ_n': epoch_loss_integ_n,
                'train/batch_loss_integ_e': epoch_loss_integ_e,
                'train/batch_loss_integ_p': epoch_loss_integ_p,
                'train/nLi_s_mean': epoch_nli_s_mean,
                'train/mean_pred_nLi_s': epoch_nli_s_mean,
                'train/std_pred_nLi_s': epoch_nli_s_std,
                'train/nLi_s_est': nli_s_est_epoch,
                'train/nLi_s_rel_std': epoch_nli_s_rel_std,
                'train/nLi_e_mean': epoch_nli_e_mean,
                'train/mean_pred_nLi_e': epoch_nli_e_mean,
                'train/std_pred_nLi_e': epoch_nli_e_std,
                'train/nLi_e_est': nli_e_est_epoch,
                'train/nLi_e_rel_std': epoch_nli_e_rel_std,
                'learning_rate_n': optimizers[0].param_groups[0]['lr'],
                'learning_rate_e': optimizers[1].param_groups[0]['lr'],
                'learning_rate_p': optimizers[2].param_groups[0]['lr'],
            }
            log_payload.update({f'train/{key}': value for key, value in degradation_epoch.items()})
            if delta_sei_grad_debug_stats is not None:
                log_payload.update({
                    f'debug/{key}': value for key, value in delta_sei_grad_debug_stats.items()
                })
            if len(optimizers) > 3:
                log_payload['learning_rate_sei'] = optimizers[3].param_groups[0]['lr']

            metrics_val_subset = None
            with timer.section('logging_eval'):
                if _is_due_epoch(epoch_display, eval_every):
                    metrics_val_subset = _evaluate(nn_models, model, val_eval_subset, cfg, p)
            if metrics_val_subset is not None:
                history_row.update({
                    'val_subset_loss': metrics_val_subset['loss'],
                    'val_subset_loss_v': metrics_val_subset['loss_v'],
                    'val_subset_loss_li_s': metrics_val_subset['loss_li_s'],
                    'val_subset_loss_li_e': metrics_val_subset['loss_li_e'],
                    'val_subset_loss_integ': metrics_val_subset['loss_integ'],
                    'val_subset_loss_integ_n': metrics_val_subset['loss_integ_n'],
                    'val_subset_loss_integ_e': metrics_val_subset['loss_integ_e'],
                    'val_subset_loss_integ_p': metrics_val_subset['loss_integ_p'],
                    'val_subset_nLi_s_mean': metrics_val_subset['nLi_s_mean'],
                    'val_subset_mean_pred_nLi_s': metrics_val_subset['mean_pred_nLi_s'],
                    'val_subset_std_pred_nLi_s': metrics_val_subset['std_pred_nLi_s'],
                    'val_subset_nLi_s_est': metrics_val_subset['nLi_s_est'],
                    'val_subset_nLi_s_rel_std': metrics_val_subset['nLi_s_rel_std'],
                    'val_subset_nLi_e_mean': metrics_val_subset['nLi_e_mean'],
                    'val_subset_mean_pred_nLi_e': metrics_val_subset['mean_pred_nLi_e'],
                    'val_subset_std_pred_nLi_e': metrics_val_subset['std_pred_nLi_e'],
                    'val_subset_nLi_e_est': metrics_val_subset['nLi_e_est'],
                    'val_subset_nLi_e_rel_std': metrics_val_subset['nLi_e_rel_std'],
                })
                history_row.update({
                    f'val_subset_{key}': metrics_val_subset[key]
                    for key in ('delta_sei_est', 'delta_sei_true', 'delta_sei_error',
                                'epsilon_e_n_est', 'epsilon_e_n_true', 'epsilon_e_n_error',
                                'nLi_sei_est', 'nLi_sei_true', 'R_f_n_est')
                })
                log_payload.update({
                    'val_subset/loss': metrics_val_subset['loss'],
                    'val_subset/loss_voltage': metrics_val_subset['loss_v'],
                    'val_subset/loss_li_s': metrics_val_subset['loss_li_s'],
                    'val_subset/loss_li_e': metrics_val_subset['loss_li_e'],
                    'val_subset/loss_integ': metrics_val_subset['loss_integ'],
                    'val_subset/loss_integ_n': metrics_val_subset['loss_integ_n'],
                    'val_subset/loss_integ_e': metrics_val_subset['loss_integ_e'],
                    'val_subset/loss_integ_p': metrics_val_subset['loss_integ_p'],
                    'val_subset/nLi_s_mean': metrics_val_subset['nLi_s_mean'],
                    'val_subset/mean_pred_nLi_s': metrics_val_subset['mean_pred_nLi_s'],
                    'val_subset/std_pred_nLi_s': metrics_val_subset['std_pred_nLi_s'],
                    'val_subset/nLi_s_est': metrics_val_subset['nLi_s_est'],
                    'val_subset/nLi_s_rel_std': metrics_val_subset['nLi_s_rel_std'],
                    'val_subset/nLi_e_mean': metrics_val_subset['nLi_e_mean'],
                    'val_subset/mean_pred_nLi_e': metrics_val_subset['mean_pred_nLi_e'],
                    'val_subset/std_pred_nLi_e': metrics_val_subset['std_pred_nLi_e'],
                    'val_subset/nLi_e_est': metrics_val_subset['nLi_e_est'],
                    'val_subset/nLi_e_rel_std': metrics_val_subset['nLi_e_rel_std'],
                })
                log_payload.update({
                    f'val_subset/{key}': metrics_val_subset[key]
                    for key in ('delta_sei_est', 'delta_sei_true', 'delta_sei_error',
                                'epsilon_e_n_est', 'epsilon_e_n_true', 'epsilon_e_n_error',
                                'nLi_sei_est', 'nLi_sei_true', 'R_f_n_est')
                })

            metrics_train_full = metrics_val_full = None
            with timer.section('logging_eval'):
                if _is_due_epoch(epoch_display, full_eval_every):
                    metrics_train_full = _evaluate(nn_models, model, train_data, cfg, p)
                    metrics_val_full = _evaluate(nn_models, model, val_data, cfg, p)
            if metrics_train_full is not None:
                history_row.update({
                    'train_full_loss': metrics_train_full['loss'],
                    'train_full_loss_v': metrics_train_full['loss_v'],
                    'train_full_loss_li_s': metrics_train_full['loss_li_s'],
                    'train_full_loss_li_e': metrics_train_full['loss_li_e'],
                    'train_full_loss_integ': metrics_train_full['loss_integ'],
                    'train_full_loss_integ_n': metrics_train_full['loss_integ_n'],
                    'train_full_loss_integ_e': metrics_train_full['loss_integ_e'],
                    'train_full_loss_integ_p': metrics_train_full['loss_integ_p'],
                    'train_full_nLi_s_mean': metrics_train_full['nLi_s_mean'],
                    'train_full_mean_pred_nLi_s': metrics_train_full['mean_pred_nLi_s'],
                    'train_full_std_pred_nLi_s': metrics_train_full['std_pred_nLi_s'],
                    'train_full_nLi_s_est': metrics_train_full['nLi_s_est'],
                    'train_full_nLi_s_rel_std': metrics_train_full['nLi_s_rel_std'],
                    'train_full_nLi_e_mean': metrics_train_full['nLi_e_mean'],
                    'train_full_mean_pred_nLi_e': metrics_train_full['mean_pred_nLi_e'],
                    'train_full_std_pred_nLi_e': metrics_train_full['std_pred_nLi_e'],
                    'train_full_nLi_e_est': metrics_train_full['nLi_e_est'],
                    'train_full_nLi_e_rel_std': metrics_train_full['nLi_e_rel_std'],
                    'val_full_loss': metrics_val_full['loss'],
                    'val_full_loss_v': metrics_val_full['loss_v'],
                    'val_full_loss_li_s': metrics_val_full['loss_li_s'],
                    'val_full_loss_li_e': metrics_val_full['loss_li_e'],
                    'val_full_loss_integ': metrics_val_full['loss_integ'],
                    'val_full_loss_integ_n': metrics_val_full['loss_integ_n'],
                    'val_full_loss_integ_e': metrics_val_full['loss_integ_e'],
                    'val_full_loss_integ_p': metrics_val_full['loss_integ_p'],
                    'val_full_nLi_s_mean': metrics_val_full['nLi_s_mean'],
                    'val_full_mean_pred_nLi_s': metrics_val_full['mean_pred_nLi_s'],
                    'val_full_std_pred_nLi_s': metrics_val_full['std_pred_nLi_s'],
                    'val_full_nLi_s_est': metrics_val_full['nLi_s_est'],
                    'val_full_nLi_s_rel_std': metrics_val_full['nLi_s_rel_std'],
                    'val_full_nLi_e_mean': metrics_val_full['nLi_e_mean'],
                    'val_full_mean_pred_nLi_e': metrics_val_full['mean_pred_nLi_e'],
                    'val_full_std_pred_nLi_e': metrics_val_full['std_pred_nLi_e'],
                    'val_full_nLi_e_est': metrics_val_full['nLi_e_est'],
                    'val_full_nLi_e_rel_std': metrics_val_full['nLi_e_rel_std'],
                })
                history_row.update({
                    f'train_full_{key}': metrics_train_full[key]
                    for key in ('delta_sei_est', 'delta_sei_true', 'delta_sei_error',
                                'epsilon_e_n_est', 'epsilon_e_n_true', 'epsilon_e_n_error',
                                'nLi_sei_est', 'nLi_sei_true', 'R_f_n_est')
                })
                history_row.update({
                    f'val_full_{key}': metrics_val_full[key]
                    for key in ('delta_sei_est', 'delta_sei_true', 'delta_sei_error',
                                'epsilon_e_n_est', 'epsilon_e_n_true', 'epsilon_e_n_error',
                                'nLi_sei_est', 'nLi_sei_true', 'R_f_n_est')
                })
                log_payload.update({
                    'train_full/loss': metrics_train_full['loss'],
                    'train_full/loss_voltage': metrics_train_full['loss_v'],
                    'train_full/loss_li_s': metrics_train_full['loss_li_s'],
                    'train_full/loss_li_e': metrics_train_full['loss_li_e'],
                    'train_full/loss_integ': metrics_train_full['loss_integ'],
                    'train_full/loss_integ_n': metrics_train_full['loss_integ_n'],
                    'train_full/loss_integ_e': metrics_train_full['loss_integ_e'],
                    'train_full/loss_integ_p': metrics_train_full['loss_integ_p'],
                    'train_full/nLi_s_mean': metrics_train_full['nLi_s_mean'],
                    'train_full/mean_pred_nLi_s': metrics_train_full['mean_pred_nLi_s'],
                    'train_full/std_pred_nLi_s': metrics_train_full['std_pred_nLi_s'],
                    'train_full/nLi_s_est': metrics_train_full['nLi_s_est'],
                    'train_full/nLi_s_rel_std': metrics_train_full['nLi_s_rel_std'],
                    'train_full/nLi_e_mean': metrics_train_full['nLi_e_mean'],
                    'train_full/mean_pred_nLi_e': metrics_train_full['mean_pred_nLi_e'],
                    'train_full/std_pred_nLi_e': metrics_train_full['std_pred_nLi_e'],
                    'train_full/nLi_e_est': metrics_train_full['nLi_e_est'],
                    'train_full/nLi_e_rel_std': metrics_train_full['nLi_e_rel_std'],
                    'val_full/loss': metrics_val_full['loss'],
                    'val_full/loss_voltage': metrics_val_full['loss_v'],
                    'val_full/loss_li_s': metrics_val_full['loss_li_s'],
                    'val_full/loss_li_e': metrics_val_full['loss_li_e'],
                    'val_full/loss_integ': metrics_val_full['loss_integ'],
                    'val_full/loss_integ_n': metrics_val_full['loss_integ_n'],
                    'val_full/loss_integ_e': metrics_val_full['loss_integ_e'],
                    'val_full/loss_integ_p': metrics_val_full['loss_integ_p'],
                    'val_full/nLi_s_mean': metrics_val_full['nLi_s_mean'],
                    'val_full/mean_pred_nLi_s': metrics_val_full['mean_pred_nLi_s'],
                    'val_full/std_pred_nLi_s': metrics_val_full['std_pred_nLi_s'],
                    'val_full/nLi_s_est': metrics_val_full['nLi_s_est'],
                    'val_full/nLi_s_rel_std': metrics_val_full['nLi_s_rel_std'],
                    'val_full/nLi_e_mean': metrics_val_full['nLi_e_mean'],
                    'val_full/mean_pred_nLi_e': metrics_val_full['mean_pred_nLi_e'],
                    'val_full/std_pred_nLi_e': metrics_val_full['std_pred_nLi_e'],
                    'val_full/nLi_e_est': metrics_val_full['nLi_e_est'],
                    'val_full/nLi_e_rel_std': metrics_val_full['nLi_e_rel_std'],
                })
                log_payload.update({
                    f'train_full/{key}': metrics_train_full[key]
                    for key in ('delta_sei_est', 'delta_sei_true', 'delta_sei_error',
                                'epsilon_e_n_est', 'epsilon_e_n_true', 'epsilon_e_n_error',
                                'nLi_sei_est', 'nLi_sei_true', 'R_f_n_est')
                })
                log_payload.update({
                    f'val_full/{key}': metrics_val_full[key]
                    for key in ('delta_sei_est', 'delta_sei_true', 'delta_sei_error',
                                'epsilon_e_n_est', 'epsilon_e_n_true', 'epsilon_e_n_error',
                                'nLi_sei_est', 'nLi_sei_true', 'R_f_n_est')
                })

            if timing_enabled:
                timer.update(model.timing_summary())
                timing_payload = timer.payload()
                history_row.update(timer.history_fields())
                log_payload.update(timing_payload)

            history.append(history_row)

            if epoch_display == 1 or _is_due_epoch(epoch_display, int(cfg.get('print_every', 50))):
                msg = (f"Epoch {epoch_display:5d} | train loss {epoch_loss:.6f} | "
                       f"V {epoch_loss_v:.6f} | Li_s {epoch_loss_li_s:.6f} | "
                       f"Li_e {epoch_loss_li_e:.6f} | integ {epoch_loss_integ:.6f}")
                if metrics_val_subset is not None:
                    msg += f" | val subset {metrics_val_subset['loss']:.6f}"
                if metrics_val_full is not None:
                    msg += f" | val full {metrics_val_full['loss']:.6f}"
                print(msg)

            if run is not None:
                run.log(log_payload, step=epoch_display)

            if epoch_display == 1 or _is_due_epoch(epoch_display, plot_every):
                if run is not None:
                    _log_profile_predictions(run, nn_models, model, {'train': train_data, 'val': val_data}, epoch_display, cfg)
                _log_degradation_history_plot(run, history, epoch_display, cfg)

            if timing_enabled and _is_due_epoch(epoch_display, profile_timing_every):
                timer.print_report(epoch_display)
                if run is not None and profile_timing_log_wandb:
                    run.log(timing_payload, step=epoch_display)

            if metrics_val_full is not None and metrics_val_full['loss'] < best_val:
                best_val = metrics_val_full['loss']
                _save_spme_nn_models(nn_models, cfg)
                _save_best_spme_nn_models(nn_models, cfg)

            _save_periodic_checkpoint(epoch, nn_models, optimizers, best_val, history, cfg, model=model)

        pd.DataFrame(history).to_csv(os.path.join(cfg.get('save_dir', 'results'), cfg['history_save_name']), index=False)
        elapsed = time.time() - start_time
        print(f'Training time: {elapsed:.2f} sec')
    except KeyboardInterrupt:
        interrupted_epoch = epoch if 'epoch' in locals() else start_epoch
        _save_interrupt_checkpoint(interrupted_epoch, nn_models, optimizers, best_val, history, cfg, model=model)
        print(f"Interrupted. Saved checkpoint at epoch {interrupted_epoch + 1}.")
    finally:
        if run is not None:
            run.finish()
