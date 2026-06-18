import torch
from heads import make_vqc

K = 4
L = 1
layer = make_vqc(K, L)

for batch in (1, 5):
    x = torch.randn(batch, K)
    print(f'--- batch={batch} x shape={tuple(x.shape)} ---')
    try:
        y = layer(x)
        print('y type:', type(y))
        print('y repr:', y)
        if hasattr(y, 'shape'):
            print('y shape:', tuple(y.shape))
        else:
            print('y len:', len(y))
            for i, item in enumerate(y):
                print(' item', i, 'type=', type(item), 'shape=', tuple(item.shape) if hasattr(item, 'shape') else item)
    except Exception:
        import traceback
        traceback.print_exc()
