import time
import json
import torch
import numpy as np
from pathlib import Path
import importlib.util
import sys

# Import the model module by path to avoid module name issues
spec = importlib.util.spec_from_file_location(
    'rcstp',
    Path(__file__).parent / 'Revolution_Causal Spatio-Temporal Prediction.py'
)
rcstp = importlib.util.module_from_spec(spec)
sys.modules['rcstp'] = rcstp
spec.loader.exec_module(rcstp)

UltraEnhancedSpatioTemporalPredictor = rcstp.UltraEnhancedSpatioTemporalPredictor
CombinedLoss = rcstp.CombinedLoss
get_optimizer = rcstp.get_optimizer


def synthetic_loader(batch_size=8, seq_len=8, H=16, W=8, channels=1, batches=10):
    for _ in range(batches):
        x = torch.randn(batch_size, seq_len, channels, H, W)
        y = torch.randn(batch_size, seq_len, channels, H, W)
        yield x, y


def run_one_exp(config, device='cpu'):
    model = UltraEnhancedSpatioTemporalPredictor(
        d_model=config.get('d_model', 128),
        d_ff=config.get('d_ff', 512),
        temporal_layers=config.get('temporal_layers', 2),
        use_fourier=config.get('use_fourier', True),
        use_causal_branch=config.get('use_causal_branch', True),
        temporal_type=config.get('temporal_type', 'mamba')
    )
    model.to(device)
    opt = get_optimizer(model, lr=1e-3)
    loss_fn = CombinedLoss()

    # warmup
    loader = synthetic_loader(batch_size=config.get('batch_size', 4), seq_len=8, H=16, W=8, batches=5)
    times = []
    max_mem = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        t0 = time.time()
        pred = model(x)
        loss, *_ = loss_fn(pred, y)
        loss.backward(retain_graph=False)
        opt.step(); opt.zero_grad()
        t1 = time.time()
        times.append(t1 - t0)
        if device == 'cuda':
            torch.cuda.synchronize()
            max_mem = max(max_mem, torch.cuda.max_memory_allocated())

    throughput = (len(times) * config.get('batch_size', 4)) / sum(times)
    mem_mb = max_mem / (1024 ** 2) if device == 'cuda' else None
    return {'throughput_samples_per_sec': throughput, 'peak_mem_mb': mem_mb}


def main():
    out = []
    configs = [
        {'name': 'full', 'use_fourier': True, 'use_causal_branch': True, 'temporal_type': 'mamba', 'batch_size': 4},
        {'name': 'no_fourier', 'use_fourier': False, 'use_causal_branch': True, 'temporal_type': 'mamba', 'batch_size': 4},
        {'name': 'no_causal', 'use_fourier': True, 'use_causal_branch': False, 'temporal_type': 'mamba', 'batch_size': 4},
        {'name': 'gru_temp', 'use_fourier': True, 'use_causal_branch': True, 'temporal_type': 'gru', 'batch_size': 4},
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for cfg in configs:
        seeds = [0, 1, 2]
        seed_results = []
        for s in seeds:
            torch.manual_seed(s); np.random.seed(s)
            res = run_one_exp(cfg, device=device)
            seed_results.append(res)
        # aggregate
        tp = [r['throughput_samples_per_sec'] for r in seed_results]
        mem = [r['peak_mem_mb'] for r in seed_results if r['peak_mem_mb'] is not None]
        record = {
            'config': cfg,
            'throughput_mean': float(np.mean(tp)),
            'throughput_std': float(np.std(tp)),
            'mem_mean_mb': float(np.mean(mem)) if mem else None,
            'mem_std_mb': float(np.std(mem)) if mem else None,
            'seed_runs': seed_results
        }
        out.append(record)

    p = Path(__file__).with_suffix('.results.json')
    p.write_text(json.dumps(out, indent=2))
    print(f"Wrote results to {p}")


if __name__ == '__main__':
    main()
