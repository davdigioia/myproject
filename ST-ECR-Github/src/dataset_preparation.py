"""
dataset_preparation.py

Utilities to convert common academic video datasets into a per-sequence
folder layout suitable for `ImageSequenceDataset` and `run_real_dataset_experiment.py`.

Included converters:
- BAIR HDF5 -> per-sequence PNG folders (common format used in many video-pred
  benchmarks; you'll need the BAIR hdf5 file locally).
- Generic video file(s) -> frames in per-video folder (uses OpenCV).
- Folder normalizer: copy/symlink existing per-sequence image folders into a
  single destination.

Notes:
- This script does not download datasets. It only converts local files into
  the folder layout required by the training scripts.
- Dependencies: `h5py`, `opencv-python` (cv2), `tqdm`.

Usage examples:
python dataset_preparation.py bair_to_folders --h5 /path/to/bair_videos.h5 --out /path/to/out_root --seq-len 20
python dataset_preparation.py video_to_folders --video /path/to/video.mp4 --out /path/to/out_root --seq-id myvideo --frame-step 1

"""

import os
import argparse
import math
import shutil
from pathlib import Path
from tqdm import tqdm

try:
    import h5py
except Exception:
    h5py = None

try:
    import cv2
except Exception:
    cv2 = None


def bair_h5_to_folders(h5_path: str, out_root: str, seq_len: int = 20, key: str = 'videos'):
    """Convert BAIR-style HDF5 file (shape: [N, T, H, W, C]) to per-sequence png folders.

    Arguments:
        h5_path: path to .h5 file containing dataset (common BAIR releases)
        out_root: destination root directory to create sequence folders
        seq_len: number of frames to write per sequence (T in HDF5 must be >= seq_len)
        key: dataset key in the HDF5 file (often 'videos')
    """
    if h5py is None:
        raise RuntimeError('h5py is required for BAIR conversion. pip install h5py')

    with h5py.File(h5_path, 'r') as f:
        if key not in f:
            raise KeyError(f"Key '{key}' not found in {h5_path}. Available keys: {list(f.keys())}")
        ds = f[key]
        N, T, H, W, C = ds.shape
        os.makedirs(out_root, exist_ok=True)
        written = 0
        for idx in tqdm(range(N), desc='BAIR sequences'):
            if T < seq_len:
                continue
            seq = ds[idx, :seq_len]  # (T,H,W,C)
            seq_dir = os.path.join(out_root, f'seq_{idx:06d}')
            os.makedirs(seq_dir, exist_ok=True)
            for t in range(seq_len):
                frame = seq[t]  # H,W,C in [0,1] or [0,255]
                # normalize to uint8
                if frame.dtype != 'uint8':
                    # assume floats 0..1
                    frame = (frame * 255.0).astype('uint8')
                # if color, convert to grayscale
                if frame.ndim == 3 and frame.shape[2] == 3:
                    # cv2 expects BGR but to save grayscale we'll average
                    import numpy as np
                    frame_gray = frame.mean(axis=2).astype('uint8')
                else:
                    frame_gray = frame
                out_path = os.path.join(seq_dir, f'{t:05d}.png')
                # write using OpenCV if available, else use imageio
                if cv2 is not None:
                    cv2.imwrite(out_path, frame_gray)
                else:
                    try:
                        from imageio import imwrite
                        imwrite(out_path, frame_gray)
                    except Exception:
                        raise RuntimeError('No suitable image writer found (cv2 or imageio)')
            written += 1
        print(f'Wrote {written} sequences to {out_root}')


    def hko_to_folders(src_path: str, out_root: str, seq_len: int = 20):
        """Convert HKO-style files into per-sequence folders.

        Accepts either:
          - an HDF5 file with shape (N, T, H, W, C) under key 'radar' or similar, or
          - a directory containing .npy files where each file is a (T, H, W) array, or
          - a directory of per-sequence folders (already suitable) in which case we
            normalize and copy them to out_root.

        The converter normalizes to 0..255 uint8 grayscale images.
        """
        os.makedirs(out_root, exist_ok=True)
        # If it's an h5 file, attempt to read
        if os.path.isfile(src_path) and src_path.lower().endswith(('.h5', '.hdf5')):
            if h5py is None:
                raise RuntimeError('h5py is required for HKO conversion. pip install h5py')
            with h5py.File(src_path, 'r') as f:
                # find likely dataset key
                key = None
                for candidate in ['radar', 'videos', 'data', 'array']:
                    if candidate in f:
                        key = candidate
                        break
                if key is None:
                    # pick first dataset-like key
                    keys = [k for k in f.keys()]
                    if not keys:
                        raise RuntimeError('No datasets found inside HDF5 file')
                    key = keys[0]
                ds = f[key]
                N, T, H, W = ds.shape[:4]
                written = 0
                for idx in range(N):
                    if T < seq_len:
                        continue
                    seq = ds[idx, :seq_len]
                    seq_dir = os.path.join(out_root, f'seq_{idx:06d}')
                    os.makedirs(seq_dir, exist_ok=True)
                    # seq may be (T,H,W) or (T,H,W,1)
                    for t in range(seq_len):
                        frame = seq[t]
                        import numpy as np
                        if frame.dtype != 'uint8':
                            # normalize per-frame to 0..255
                            fmin, fmax = frame.min(), frame.max()
                            if fmax - fmin < 1e-6:
                                img = (frame * 0).astype('uint8')
                            else:
                                img = ((frame - fmin) / (fmax - fmin) * 255.0).astype('uint8')
                        else:
                            img = frame
                        if img.ndim == 3 and img.shape[2] == 3:
                            img = img.mean(axis=2).astype('uint8')
                        out_path = os.path.join(seq_dir, f'{t:05d}.png')
                        if cv2 is not None:
                            cv2.imwrite(out_path, img)
                        else:
                            try:
                                from imageio import imwrite
                                imwrite(out_path, img)
                            except Exception:
                                raise RuntimeError('No suitable image writer found (cv2 or imageio)')
                    written += 1
                print(f'Wrote {written} sequences to {out_root} from {src_path}')
                return

        # If src_path is a directory of .npy files (each file a sequence)
        if os.path.isdir(src_path):
            files = sorted([p for p in os.listdir(src_path) if p.lower().endswith('.npy')])
            if files:
                written = 0
                for i, fname in enumerate(files):
                    arr = np.load(os.path.join(src_path, fname))  # expect (T,H,W) or (T,H,W,1)
                    if arr.shape[0] < seq_len:
                        continue
                    seq = arr[:seq_len]
                    seq_dir = os.path.join(out_root, f'seq_{i:06d}')
                    os.makedirs(seq_dir, exist_ok=True)
                    for t in range(seq_len):
                        frame = seq[t]
                        if frame.dtype != 'uint8':
                            fmin, fmax = frame.min(), frame.max()
                            if fmax - fmin < 1e-6:
                                img = (frame * 0).astype('uint8')
                            else:
                                img = ((frame - fmin) / (fmax - fmin) * 255.0).astype('uint8')
                        else:
                            img = frame
                        if img.ndim == 3 and img.shape[2] == 3:
                            img = img.mean(axis=2).astype('uint8')
                        out_path = os.path.join(seq_dir, f'{t:05d}.png')
                        if cv2 is not None:
                            cv2.imwrite(out_path, img)
                        else:
                            try:
                                from imageio import imwrite
                                imwrite(out_path, img)
                            except Exception:
                                raise RuntimeError('No suitable image writer found (cv2 or imageio)')
                    written += 1
                print(f'Wrote {written} sequences to {out_root} from numpy files in {src_path}')
                return

        # If src_path is already a folder with per-sequence subfolders, normalize
        normalize_folders(src_path, out_root)



def video_to_folders(video_path: str, out_root: str, seq_id: str = 'video', frame_step: int = 1):
    """Extract frames from a single video file and write them into a folder.

    This is helpful for datasets distributed per-video (e.g., DAVIS or custom MP4s).
    """
    if cv2 is None:
        raise RuntimeError('opencv-python is required for video_to_folders. pip install opencv-python')
    os.makedirs(out_root, exist_ok=True)
    seq_dir = os.path.join(out_root, seq_id)
    os.makedirs(seq_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {video_path}')
    frame_idx = 0
    written = 0
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f'Frames {seq_id}')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            # convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out_path = os.path.join(seq_dir, f'{written:05d}.png')
            cv2.imwrite(out_path, gray)
            written += 1
        frame_idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()
    print(f'Wrote {written} frames to {seq_dir}')


def normalize_folders(src_root: str, out_root: str):
    """Copy or symlink per-sequence folders from `src_root` into a normalized
    `out_root` structure. Useful when a dataset has many subfolders nested.
    """
    os.makedirs(out_root, exist_ok=True)
    count = 0
    for entry in sorted(os.listdir(src_root)):
        seq_dir = os.path.join(src_root, entry)
        if os.path.isdir(seq_dir):
            dst = os.path.join(out_root, f'seq_{count:06d}')
            if os.path.exists(dst):
                shutil.rmtree(dst)
            try:
                os.symlink(os.path.abspath(seq_dir), dst)
            except Exception:
                # fallback to copying if symlink not allowed on Windows
                shutil.copytree(seq_dir, dst)
            count += 1
    print(f'Normalized {count} sequence folders into {out_root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preparation utilities')
    sub = parser.add_subparsers(dest='cmd')

    p1 = sub.add_parser('bair_to_folders')
    p1.add_argument('--h5', required=True)
    p1.add_argument('--out', required=True)
    p1.add_argument('--seq-len', type=int, default=20)
    p1.add_argument('--key', type=str, default='videos')

    p2 = sub.add_parser('video_to_folders')
    p2.add_argument('--video', required=True)
    p2.add_argument('--out', required=True)
    p2.add_argument('--seq-id', default='video')
    p2.add_argument('--frame-step', type=int, default=1)

    p3 = sub.add_parser('normalize_folders')
    p3.add_argument('--src', required=True)
    p3.add_argument('--out', required=True)

    args = parser.parse_args()
    if args.cmd == 'bair_to_folders':
        bair_h5_to_folders(args.h5, args.out, seq_len=args.seq_len, key=args.key)
    elif args.cmd == 'video_to_folders':
        video_to_folders(args.video, args.out, seq_id=args.seq_id, frame_step=args.frame_step)
    elif args.cmd == 'normalize_folders':
        normalize_folders(args.src, args.out)
    else:
        parser.print_help()
