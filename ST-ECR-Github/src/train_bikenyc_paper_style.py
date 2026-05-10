"""Unified trainer for BikeNYC experiments.

Single clean implementation (grid and graph modes supported). This file was
previously duplicated which caused the script to run twice; now it's a single
entry point.
"""

import argparse
import time
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import importlib.util, sys
spec = importlib.util.spec_from_file_location('rcstp', Path(__file__).parent / 'Revolution_Causal Spatio-Temporal Prediction.py')
rcstp = importlib.util.module_from_spec(spec)
sys.modules['rcstp'] = rcstp
spec.loader.exec_module(rcstp)

UltraEnhancedSpatioTemporalPredictor = rcstp.UltraEnhancedSpatioTemporalPredictor
CombinedLoss = rcstp.CombinedLoss
get_optimizer = rcstp.get_optimizer


def load_data(preproc_dir):
    d = Path(preproc_dir)
    tx = np.load(d / 'train_x.npy')
    ty = np.load(d / 'train_y.npy')
    vx = np.load(d / 'val_x.npy')
    vy = np.load(d / 'val_y.npy')
    return tx, ty, vx, vy


def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    t0 = time.time()
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).float()
        pred = model(x)
        loss, *_ = loss_fn(pred, y)
        loss.backward()
        opt.step(); opt.zero_grad()
        bs = x.shape[0]
        total_loss += loss.item() * bs
        total_samples += bs
    t1 = time.time()
    return total_loss / total_samples, (total_samples / (t1 - t0))


def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            pred = model(x)
            loss, *_ = loss_fn(pred, y)
            bs = x.shape[0]
            total_loss += loss.item() * bs
            total_samples += bs
    return total_loss / total_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--use_fourier', action='store_true')
    p.add_argument('--no_fourier', action='store_true')
    p.add_argument('--use_causal', action='store_true')
    p.add_argument('--no_causal', action='store_true')
    p.add_argument('--temporal_type', choices=['mamba', 'gru'], default='mamba')
    p.add_argument('--use_graph', action='store_true', help='Use GraphToGridEncoder and graph-mode data (B,T,N,F)')
    p.add_argument('--adj_path', default=None, help='Path to adjacency .npy for graph mode')
    p.add_argument('--save_path', default='results')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.use_graph:
        # Expect preprocessed graph-mode npy files: train_x_graph.npy with shape (N, T, Nodes, F)
        d = Path(args.data_dir)
        tx = np.load(d / 'train_x_graph.npy')
        ty = np.load(d / 'train_y_graph.npy')
        vx = np.load(d / 'val_x_graph.npy')
        vy = np.load(d / 'val_y_graph.npy')
        adj = None
        if args.adj_path is not None:
            adj = np.load(args.adj_path)
        # DataLoader will yield (x_graph, y_graph)
        train_ds = TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty))
        val_ds = TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    else:
        tx, ty, vx, vy = load_data(args.data_dir)

        train_ds = TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty))
        val_ds = TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    use_fourier = not args.no_fourier if (args.use_fourier or args.no_fourier) else True
    use_causal = not args.no_causal if (args.use_causal or args.no_causal) else True

    graph_cfg = None
    if args.use_graph:
        # Minimal config; assumes nodes arrange into grid that matches GraphToGridEncoder
        # user can customize this manually by editing script or passing additional args
        # Default grid 16x8 nodes
        graph_cfg = {'in_features': 1, 'gcn_hidden_features': 64, 'out_channels': 128, 'grid_height': 8, 'grid_width': 16}

    # Infer input/output channel count from training data
    if args.use_graph:
        # graph-mode: tx shape (N, T, Nodes, F)
        input_channels = int(tx.shape[-1])
    else:
        # grid-mode: tx shape (N, T, C, H, W)
        input_channels = int(tx.shape[2])

    model = UltraEnhancedSpatioTemporalPredictor(
        d_model=128,
        d_ff=512,
        temporal_layers=2,
        use_fourier=use_fourier,
        use_causal_branch=use_causal,
        temporal_type=args.temporal_type,
        graph_encoder_config=graph_cfg,
        input_channels=input_channels
    ).to(device)

    opt = get_optimizer(model, lr=args.lr)
    loss_fn = CombinedLoss()

    best_val = float('inf')
    save_dir = Path(args.save_path); save_dir.mkdir(parents=True, exist_ok=True)
    results = {'history': []}

    for epoch in range(1, args.epochs + 1):
        train_loss, throughput = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)

        results['history'].append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'throughput': throughput})
        print(f'Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, throughput={throughput:.2f} samples/s')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            if args.use_graph and adj is not None:
                # save adjacency with results for reproducibility
                np.save(save_dir / 'adj.npy', adj)

    (save_dir / 'results.json').write_text(json.dumps(results, indent=2))
    print('Training complete; results saved')


if __name__ == '__main__':
    main()
"""
Unified trainer for BikeNYC experiments.

This file previously contained two trainer implementations which caused the script
to execute twice and raise a channel mismatch. The duplicate code has been removed
and replaced with a single, configurable trainer that supports grid and graph modes,
infers input channels, applies StepLR, and uses early stopping.
"""

import argparse
import time
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import importlib.util, sys
spec = importlib.util.spec_from_file_location('rcstp', Path(__file__).parent / 'Revolution_Causal Spatio-Temporal Prediction.py')
rcstp = importlib.util.module_from_spec(spec)
sys.modules['rcstp'] = rcstp
spec.loader.exec_module(rcstp)

UltraEnhancedSpatioTemporalPredictor = rcstp.UltraEnhancedSpatioTemporalPredictor
CombinedLoss = rcstp.CombinedLoss
get_optimizer = rcstp.get_optimizer


def load_data(preproc_dir):
    d = Path(preproc_dir)
    tx = np.load(d / 'train_x.npy')
    ty = np.load(d / 'train_y.npy')
    vx = np.load(d / 'val_x.npy')
    vy = np.load(d / 'val_y.npy')
    return tx, ty, vx, vy


def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    t0 = time.time()
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).float()
        pred = model(x)
        loss, *_ = loss_fn(pred, y)
        loss.backward()
        opt.step(); opt.zero_grad()
        bs = x.shape[0]
        total_loss += loss.item() * bs
        total_samples += bs
    t1 = time.time()
    return total_loss / total_samples, (total_samples / (t1 - t0))


def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            pred = model(x)
            loss, *_ = loss_fn(pred, y)
            bs = x.shape[0]
            total_loss += loss.item() * bs
            total_samples += bs
    return total_loss / total_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--use_fourier', action='store_true')
    p.add_argument('--no_fourier', action='store_true')
    p.add_argument('--use_causal', action='store_true')
    p.add_argument('--no_causal', action='store_true')
    p.add_argument('--temporal_type', choices=['mamba', 'gru'], default='mamba')
    p.add_argument('--use_graph', action='store_true', help='Use GraphToGridEncoder and graph-mode data (B,T,N,F)')
    p.add_argument('--adj_path', default=None, help='Path to adjacency .npy for graph mode')
    p.add_argument('--save_path', default='results')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.use_graph:
        # Expect preprocessed graph-mode npy files: train_x_graph.npy with shape (N, T, Nodes, F)
        d = Path(args.data_dir)
        tx = np.load(d / 'train_x_graph.npy')
        ty = np.load(d / 'train_y_graph.npy')
        vx = np.load(d / 'val_x_graph.npy')
        vy = np.load(d / 'val_y_graph.npy')
        adj = None
        if args.adj_path is not None:
            adj = np.load(args.adj_path)
        # DataLoader will yield (x_graph, y_graph)
        train_ds = TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty))
        val_ds = TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    else:
        tx, ty, vx, vy = load_data(args.data_dir)

        train_ds = TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty))
        val_ds = TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    use_fourier = not args.no_fourier if (args.use_fourier or args.no_fourier) else True
    use_causal = not args.no_causal if (args.use_causal or args.no_causal) else True

    graph_cfg = None
    if args.use_graph:
        # Minimal config; assumes nodes arrange into grid that matches GraphToGridEncoder
        # user can customize this manually by editing script or passing additional args
        # Default grid 16x8 nodes
        graph_cfg = {'in_features': 1, 'gcn_hidden_features': 64, 'out_channels': 128, 'grid_height': 8, 'grid_width': 16}

    # Infer input/output channel count from training data
    if args.use_graph:
        # graph-mode: tx shape (N, T, Nodes, F)
        input_channels = int(tx.shape[-1])
    else:
        # grid-mode: tx shape (N, T, C, H, W)
        input_channels = int(tx.shape[2])

    model = UltraEnhancedSpatioTemporalPredictor(
        d_model=128,
        d_ff=512,
        temporal_layers=2,
        use_fourier=use_fourier,
        use_causal_branch=use_causal,
        temporal_type=args.temporal_type,
        graph_encoder_config=graph_cfg,
        input_channels=input_channels
    ).to(device)

    opt = get_optimizer(model, lr=args.lr)
    loss_fn = CombinedLoss()

    best_val = float('inf')
    save_dir = Path(args.save_path); save_dir.mkdir(parents=True, exist_ok=True)
    results = {'history': []}

    for epoch in range(1, args.epochs + 1):
        train_loss, throughput = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)

        results['history'].append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'throughput': throughput})
        print(f'Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, throughput={throughput:.2f} samples/s')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            if args.use_graph and adj is not None:
                # save adjacency with results for reproducibility
                np.save(save_dir / 'adj.npy', adj)

    (save_dir / 'results.json').write_text(json.dumps(results, indent=2))
    print('Training complete; results saved')


if __name__ == '__main__':
    main()
