import torch
import time
from DIRTyTorch import DIRT

t = DIRT([torch.arange(2, dtype=torch.float32)], [1,1])

for device in ['cpu', 'cuda']:
    for prec in [torch.float32, torch.float64]:
        print(f"\nTesting {device} with precision {prec}")
        A = torch.tensor([[ 0.4598,  1.3859,  0.3691], [ 0.4487, -0.1023, -0.6304], [ 0.7000,  1.6681, -0.1714]], device=device, dtype=prec)
        # Approximate
        t = DIRT([torch.linspace(-7,7,65, dtype=prec, device=device)]*3, [1,16,16,1], 4.0**torch.arange(-3,4), \
                  ExactRatio=False, Ntest=10**3, reference_sigma=0.0, requires_grad=False, use_irt2=True)
        costf = lambda x: -0.5*torch.sum((x @ A)**2, dim=1)
        tempered_costf = lambda x, beta_high, beta_low: costf(x)*(beta_high - beta_low)
        t1 = time.time()
        ev = t.cross(tempered_costf, nswp=2)
        t2 = time.time()
        print(f"{device} with precision {prec}: total evals = {ev}, time = {t2-t1}")
        