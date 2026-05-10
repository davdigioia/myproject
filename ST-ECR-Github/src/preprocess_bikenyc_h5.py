import argparse
from pathlib import Path
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler

def infer_dataset_key(h5f):
    # Try common keys
    for key in ['data', 'array', 'value', 'values', 'X', 'traffic']:
        if key in h5f:
            return key
    # fallback: first dataset
    for k in h5f.keys():
        return k
    raise RuntimeError('No datasets found in HDF5')


def sliding_windows(data, seq_len, pred_len, stride=1):
    # data shape: (T, C, H, W) or (T, H, W, C)
    if data.ndim == 4 and data.shape[1] <= 4:  # assume (T,C,H,W)
        T, C, H, W = data.shape
    elif data.ndim == 4 and data.shape[-1] <= 4:  # (T,H,W,C)
        data = np.transpose(data, (0, 3, 1, 2))
        T, C, H, W = data.shape
    else:
        raise RuntimeError('Unsupported data shape for sliding windows: ' + str(data.shape))

    Xs = []
    Ys = []
    for start in range(0, T - seq_len - pred_len + 1, stride):
        x = data[start:start+seq_len]
        y = data[start+seq_len:start+seq_len+pred_len]
        Xs.append(x)
        Ys.append(y)
    Xs = np.stack(Xs)  # (N, seq_len, C, H, W)
    Ys = np.stack(Ys)  # (N, pred_len, C, H, W)
    return Xs, Ys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5_path', required=True)
    p.add_argument('--data_key', default=None)
    p.add_argument('--seq_len', type=int, default=12)
    p.add_argument('--pred_len', type=int, default=12)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--save_dir', default='preprocessed')
    args = p.parse_args()

    h5_path = Path(args.h5_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'r') as h5f:
        key = args.data_key or infer_dataset_key(h5f)
        data = h5f[key][()]
        # Ensure float32
        data = data.astype(np.float32)

        # Expect data shape (T, C, H, W)
        if data.ndim == 3:
            # (T, H, W) -> add channel dim
            data = data[:, None, :, :]

        X, Y = sliding_windows(data, args.seq_len, args.pred_len, args.stride)

        # Split train/val/test 80/10/10
        N = X.shape[0]
        n_train = int(N * 0.8)
        n_val = int(N * 0.1)
        train_x, train_y = X[:n_train], Y[:n_train]
        val_x, val_y = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
        test_x, test_y = X[n_train + n_val:], Y[n_train + n_val:]

        # Normalize per-channel with StandardScaler over spatial flattened dims
        B, S, C, H, W = train_x.shape
        scaler = StandardScaler()
        train_flat = train_x.reshape(-1, C * H * W)
        scaler.fit(train_flat)

        def transform(arr):
            b, s, c, h, w = arr.shape
            flat = arr.reshape(-1, c * h * w)
            out_flat = scaler.transform(flat)
            return out_flat.reshape(b, s, c, h, w)

        np.save(save_dir / 'train_x.npy', transform(train_x))
        np.save(save_dir / 'train_y.npy', transform(train_y))
        np.save(save_dir / 'val_x.npy', transform(val_x))
        np.save(save_dir / 'val_y.npy', transform(val_y))
        np.save(save_dir / 'test_x.npy', transform(test_x))
        np.save(save_dir / 'test_y.npy', transform(test_y))
        # Save scaler params
        np.savez(save_dir / 'scaler.npz', mean=scaler.mean_, var=scaler.var_)

        print(f'Saved preprocessed arrays to {save_dir}')


if __name__ == '__main__':
    main()
"""
Preprocess BikeNYC HDF5 to NumPy tensors suitable for training.

This script expects the HDF5 file with keys:
- 'date' : list of timeslots
- 'data' : shape (T, 2, H, W) where data[i][0] is new-flow (inflow)

It will:
- Optionally verify MD5 checksum of the file against an md5 file
- Extract inflow (data[:,0,:,:]) and reshape to (T, 1, H, W)
- Create sliding-window sequences: for seq_len L, input X = data[t:t+L], target Y = data[t+1:t+1+L]
  so X and Y both have shape (N, L, 1, H, W)
- Split sequences in time order into train/val/test by fractions
- Normalize data (zscore or minmax or none) using training data stats and save scaler
- Save NumPy arrays: train_x.npy, train_y.npy, val_x.npy, val_y.npy, test_x.npy, test_y.npy

Usage:
  python preprocess_bikenyc_h5.py --h5_file path/to/NYC14_M16x8_T60_NewEnd.h5 --seq_len 8 --save_dir preprocessed

"""
import argparse
from pathlib import Path
import hashlib
import h5py
import numpy as np


def md5_check(file_path: Path, md5_file: Path) -> bool:
    # md5_file should contain a line with: <md5sum>  <filename>
    if not md5_file.exists():
        raise FileNotFoundError(f"MD5 file not found: {md5_file}")
    wanted = None
    with md5_file.open('r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                checksum, name = parts[0], parts[-1]
                if Path(name).name == file_path.name or name == str(file_path):
                    wanted = checksum
                    break
    if wanted is None:
        raise ValueError('No matching entry in md5 file for the provided HDF5 file')

    h = hashlib.md5()
    with file_path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    got = h.hexdigest()
    return got == wanted


def load_h5_extract_inflow(h5_path: Path) -> np.ndarray:
    with h5py.File(str(h5_path), 'r') as f:
        # Inspect keys
        if 'data' not in f:
            raise KeyError('HDF5 does not contain key "data"')
        data = f['data'][:]  # shape (T, 2, H, W)
        # Extract inflow (new-flow)
        inflow = data[:, 0, :, :]  # shape (T, H, W)
        # Add channel dim
        inflow = inflow[:, None, :, :]  # (T, 1, H, W)
        return inflow


def create_sequences(arr: np.ndarray, seq_len: int):
    # arr shape: (T, C, H, W)
    T = arr.shape[0]
    if T <= seq_len:
        raise ValueError('Time dimension T must be > seq_len')
    N = T - seq_len
    X = np.stack([arr[i:i+seq_len] for i in range(N)], axis=0)  # (N, L, C, H, W)
    Y = np.stack([arr[i+1:i+1+seq_len] for i in range(N)], axis=0)
    return X, Y


def split_time_order(X, Y, train_frac, val_frac):
    N = X.shape[0]
    i_train = int(N * train_frac)
    i_val = i_train + int(N * val_frac)
    train_x, train_y = X[:i_train], Y[:i_train]
    val_x, val_y = X[i_train:i_val], Y[i_train:i_val]
    test_x, test_y = X[i_val:], Y[i_val:]
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def normalize_data(train_x, val_x, test_x, method='zscore'):
    # Input shapes: (N, L, C, H, W)
    if method == 'none':
        scaler = {'method': 'none'}
        return train_x, val_x, test_x, scaler
    # Compute stats on training data
    train_flat = train_x.reshape(-1, *train_x.shape[2:])  # (N*L, C, H, W)
    if method == 'zscore':
        mean = train_flat.mean(axis=0)
        std = train_flat.std(axis=0)
        std[std < 1e-6] = 1.0
        train_x_n = (train_x - mean) / std
        val_x_n = (val_x - mean) / std
        test_x_n = (test_x - mean) / std
        scaler = {'method': 'zscore', 'mean': mean, 'std': std}
        return train_x_n, val_x_n, test_x_n, scaler
    elif method == 'minmax':
        mn = train_flat.min(axis=0)
        mx = train_flat.max(axis=0)
        rng = mx - mn
        rng[rng < 1e-6] = 1.0
        train_x_n = (train_x - mn) / rng
        val_x_n = (val_x - mn) / rng
        test_x_n = (test_x - mn) / rng
        scaler = {'method': 'minmax', 'min': mn, 'max': mx}
        return train_x_n, val_x_n, test_x_n, scaler
    else:
        raise ValueError('Unknown normalization method')


def save_numpy_splits(save_dir: Path, splits, scaler):
    save_dir.mkdir(parents=True, exist_ok=True)
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = splits
    np.save(save_dir / 'train_x.npy', train_x)
    np.save(save_dir / 'train_y.npy', train_y)
    np.save(save_dir / 'val_x.npy', val_x)
    np.save(save_dir / 'val_y.npy', val_y)
    np.save(save_dir / 'test_x.npy', test_x)
    np.save(save_dir / 'test_y.npy', test_y)
    # Save scaler params
    np.savez(save_dir / 'scaler.npz', **{k: (v if isinstance(v, np.ndarray) else np.array(v)) for k,v in scaler.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_file', type=str, required=True, help='Path to NYC14_M16x8_T60_NewEnd.h5')
    parser.add_argument('--seq_len', type=int, default=8, help='Sequence length L')
    parser.add_argument('--save_dir', type=str, default='preprocessed', help='Directory to save NumPy arrays')
    parser.add_argument('--norm', type=str, choices=['zscore', 'minmax', 'none'], default='zscore')
    parser.add_argument('--train_frac', type=float, default=0.7)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--md5_file', type=str, default=None, help='Optional md5 file to verify h5')
    args = parser.parse_args()

    h5_path = Path(args.h5_file)
    save_dir = Path(args.save_dir)

    if args.md5_file is not None:
        ok = md5_check(h5_path, Path(args.md5_file))
        if not ok:
            raise ValueError('MD5 mismatch for the HDF5 file')
        print('MD5 checksum OK')

    inflow = load_h5_extract_inflow(h5_path)  # (T, 1, H, W)
    print('Loaded inflow shape:', inflow.shape)

    X, Y = create_sequences(inflow, args.seq_len)
    print('Created sequences X,Y shapes:', X.shape, Y.shape)

    splits = split_time_order(X, Y, args.train_frac, args.val_frac)
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = splits
    print('Splits shapes:', train_x.shape, val_x.shape, test_x.shape)

    train_x_n, val_x_n, test_x_n, scaler = normalize_data(train_x, val_x, test_x, method=args.norm)

    # Targets are counts; we will apply same normalization to targets for training
    # Convert targets using same scaler
    def apply_scaler_to_targets(Y, scaler):
        if scaler['method'] == 'none':
            return Y
        if scaler['method'] == 'zscore':
            mean = scaler['mean']
            std = scaler['std']
            return (Y - mean) / std
        if scaler['method'] == 'minmax':
            mn = scaler['min']
            mx = scaler['max']
            rng = mx - mn
            rng[rng < 1e-6] = 1.0
            return (Y - mn) / rng
    train_y_n = apply_scaler_to_targets(train_y, scaler)
    val_y_n = apply_scaler_to_targets(val_y, scaler)
    test_y_n = apply_scaler_to_targets(test_y, scaler)

    save_numpy_splits(save_dir, ((train_x_n, train_y_n), (val_x_n, val_y_n), (test_x_n, test_y_n)), scaler)
    print('Saved preprocessed arrays to', save_dir)
