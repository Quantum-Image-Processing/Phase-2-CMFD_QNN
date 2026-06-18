import torch
from heads import make_vqc

K = 4
L = 1
layer = make_vqc(K, L)

x = torch.randn(5, K)
try:
    y = layer(x)
    print('y type:', type(y))
    print('y repr:', y)
    if isinstance(y, tuple):
        print('tuple len:', len(y))
        for i, t in enumerate(y):
            print(f'item {i}: type={type(t)}, shape={tuple(t.shape) if hasattr(t, "shape") else None}')
    else:
        print('shape=', tuple(y.shape) if hasattr(y, 'shape') else None)
except Exception:
    import traceback
    traceback.print_exc()
