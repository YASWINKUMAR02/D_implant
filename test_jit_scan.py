import time
import torch
import torch.nn.functional as F

# Test sequential vs JIT vs chunked scan
@torch.jit.script
def selective_scan_jit(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor):
    # u: (B, D, L), delta: (B, D, L), A: (D, N), B: (B, N, L), C: (B, N, L), D: (D,)
    b = u.size(0)
    d = u.size(1)
    l = u.size(2)
    n = A.size(1)

    deltaA = torch.exp(delta.unsqueeze(2) * A.unsqueeze(0).unsqueeze(-1)) # (B, D, N, L)
    deltaB_u = delta.unsqueeze(2) * B.unsqueeze(1) * u.unsqueeze(2)      # (B, D, N, L)

    x = torch.zeros((b, d, n), device=u.device, dtype=deltaA.dtype)
    y = torch.zeros((b, d, l), device=u.device, dtype=deltaA.dtype)

    for i in range(l):
        x = deltaA[:, :, :, i] * x + deltaB_u[:, :, :, i]
        y[:, :, i] = (x * C[:, :, i].unsqueeze(1)).sum(-1)

    return y + u * D.unsqueeze(0).unsqueeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B, D, N, L = 1, 48, 16, 8000
u = torch.randn(B, D, L, device=device)
delta = F.softplus(torch.randn(B, D, L, device=device))
A = -torch.exp(torch.randn(D, N, device=device))
B_t = torch.randn(B, N, L, device=device)
C_t = torch.randn(B, N, L, device=device)
D_t = torch.randn(D, device=device)

# Warmup
out = selective_scan_jit(u, delta, A, B_t, C_t, D_t)
if device.type == "cuda":
    torch.cuda.synchronize()

t0 = time.time()
for _ in range(5):
    out = selective_scan_jit(u, delta, A, B_t, C_t, D_t)
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.time()
print(f"JIT scan time for L={L}: {(t1-t0)/5*1000:.2f} ms")
