"""
Simple runner for `Revolution_Causal Spatio-Temporal Prediction.py` on BikeNYC preprocessed data.
Loads `preprocessed/test_x.npy` if available, otherwise attempts to read HDF5 in `BikeNYC/`.
Performs a forward pass through the model created by `create_model()` and saves predictions to `results_bikenyc/`.
"""
import os
import sys
import numpy as np
import torch

# Add workspace to path so we can import the module file as a script
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

MODEL_PY = os.path.join(ROOT, "Revolution_Causal Spatio-Temporal Prediction.py")

# Import the create_model factory by executing the file's namespace
import runpy
ns = runpy.run_path(MODEL_PY)
create_model = ns.get('create_model')

if create_model is None:
    raise RuntimeError("create_model() not found in the model file.")

# Try to load preprocessed data
pre_dir = os.path.join(ROOT, 'preprocessed')
results_dir = os.path.join(ROOT, 'results_bikenyc')
os.makedirs(results_dir, exist_ok=True)

x_path = os.path.join(pre_dir, 'test_x.npy')
if os.path.exists(x_path):
    print(f"Loading preprocessed test_x from {x_path}")
    test_x = np.load(x_path)
    # Expecting shape (N, T, C, H, W) or (T, C, H, W)
    if test_x.ndim == 4:
        test_x = test_x[np.newaxis, ...]
else:
    # Fallback: try to load HDF5 dataset in BikeNYC folder
    try:
        import h5py
        h5_path = os.path.join(ROOT, 'BikeNYC', 'NYC14_M16x8_T60_NewEnd.h5')
        if os.path.exists(h5_path):
            print(f"Loading HDF5 data from {h5_path}")
            with h5py.File(h5_path, 'r') as f:
                # heuristics: look for datasets
                keys = list(f.keys())
                print('Datasets in file:', keys)
                # Try typical names
                for key in ['test', 'X_test', 'data', 'test_x']:
                    if key in f:
                        data = f[key]
                        test_x = np.array(data)
                        break
                else:
                    # pick first dataset
                    data = f[keys[0]]
                    test_x = np.array(data)
            if test_x.ndim == 4:
                test_x = test_x[np.newaxis, ...]
        else:
            raise FileNotFoundError
    except Exception as e:
        raise RuntimeError('No preprocessed data found and HDF5 fallback failed.') from e

# Convert to torch tensor
x_tensor = torch.from_numpy(test_x).float()
print('Input numpy shape:', test_x.shape)

# Create model and run on a small batch
# Infer input channels from data and pass to create_model
_, seq_len, in_channels, H, W = test_x.shape
config = {'input_channels': int(in_channels)}
print(f"Creating model with config: {config}")
model = create_model(config)
model.eval()
with torch.no_grad():
    # Put on CPU (or GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x_tensor = x_tensor.to(device)

    # Limit to first batch of size 2 for speed
    batch = x_tensor[:2]
    print('Running forward pass with batch shape:', batch.shape)
    out = model(batch)

print('Output type:', type(out))
if isinstance(out, tuple):
    preds, uncert = out
    print('Preds shape:', preds.shape)
    print('Uncertainty shape:', uncert.shape)
    np.save(os.path.join(results_dir, 'preds.npy'), preds.cpu().numpy())
    np.save(os.path.join(results_dir, 'uncert.npy'), uncert.cpu().numpy())
else:
    preds = out
    print('Preds shape:', preds.shape)
    np.save(os.path.join(results_dir, 'preds.npy'), preds.cpu().numpy())

print('Saved predictions to', results_dir)
