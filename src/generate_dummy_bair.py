"""
Generate a small synthetic BAIR-style dataset (train/val/test) with simple moving-square frames.
This is only for smoke-testing `spatio_temporal_activation_bair.py` when the real BAIR data
is not available locally.

Usage (PowerShell):
& "D:\conda_envs\torch_env\python.exe" "D:\Third_paper\new_spatio_temporal_activation\generate_dummy_bair.py" --out "D:\Third_paper\new_spatio_temporal_activation\data\bair" --num-clips 12 --seq-len 20 --resize 64

The script creates folders:
out_dir/train/clip_0000/...frame_00000.png
out_dir/val/...
out_dir/test/...

Each clip contains `seq_len` PNG frames.
"""
import os
import argparse
from PIL import Image, ImageDraw


def make_moving_square_frames(seq_len, size=(64,64), square_size=10, speed=(1,1), start_pos=None):
    W, H = size
    if start_pos is None:
        x, y = 0, 0
    else:
        x, y = start_pos
    dx, dy = speed
    frames = []
    for t in range(seq_len):
        img = Image.new('RGB', (W,H), color=(0,0,0))
        draw = ImageDraw.Draw(img)
        x_t = int((x + dx * t) % (W - square_size))
        y_t = int((y + dy * t) % (H - square_size))
        draw.rectangle([x_t, y_t, x_t + square_size, y_t + square_size], fill=(255,255,255))
        frames.append(img)
    return frames


def write_clip_frames(frames, out_dir, prefix='frame'):
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(frames):
        fname = f"{prefix}_{i:05d}.png"
        img.save(os.path.join(out_dir, fname))


def create_dataset(out_root, num_clips=12, seq_len=20, resize=64):
    # split clips into train/val/test roughly 60/20/20
    os.makedirs(out_root, exist_ok=True)
    train_dir = os.path.join(out_root, 'train')
    val_dir = os.path.join(out_root, 'val')
    test_dir = os.path.join(out_root, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for i in range(num_clips):
        if i < int(num_clips * 0.6):
            split_dir = train_dir
        elif i < int(num_clips * 0.8):
            split_dir = val_dir
        else:
            split_dir = test_dir
        clip_name = f"clip_{i:04d}"
        clip_dir = os.path.join(split_dir, clip_name)
        # vary speed and start position
        speed = ((i % 3) + 1, ((i+1) % 3) + 1)
        start = ( (i*7) % (resize//2), (i*11) % (resize//2) )
        frames = make_moving_square_frames(seq_len, size=(resize, resize), square_size=max(4, resize//10), speed=speed, start_pos=start)
        write_clip_frames(frames, clip_dir)
    print(f"Created dummy BAIR-style dataset at {out_root} with {num_clips} clips (train/val/test split).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='data/bair', help='Output root for BAIR-like dataset')
    parser.add_argument('--num-clips', type=int, default=12)
    parser.add_argument('--seq-len', type=int, default=20)
    parser.add_argument('--resize', type=int, default=64)
    args = parser.parse_args()
    create_dataset(args.out, num_clips=args.num_clips, seq_len=args.seq_len, resize=args.resize)
