# train_hpr_inversion.py
# Learn ψ(r, ω) for HPR inversion with ∂ψ/∂r < 0 encouraged via hinge penalty.

import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_unit_sphere(n):
    """Uniform directions ω on S^2."""
    # Marsaglia method
    u = torch.rand(n) * 2 - 1          # in [-1, 1]
    t = torch.rand(n) * 2 * math.pi    # in [0, 2π]
    s = torch.sqrt(1 - u*u + 1e-12)
    x = s * torch.cos(t)
    y = s * torch.sin(t)
    z = u
    omega = torch.stack([x, y, z], dim=-1)
    return omega


def make_dataset(N=50_000, R=10.0, r_min=0.5, r_max=30.0, device="cpu"):
    """
    Build a dataset of (r, ω) -> r' targets from analytic HPR inversion:
       r' = R^2 / r
    """
    r = torch.rand(N, device=device) * (r_max - r_min) + r_min  # (N,)
    omega = sample_unit_sphere(N).to(device)                    # (N,3)
    # Target inversion (direction-preserving)
    r_inv = (R**2) / r                                          # (N,)

    # Inputs: [r, ωx, ωy, ωz]
    x = torch.cat([r.unsqueeze(-1), omega], dim=-1)             # (N,4)
    y = r_inv.unsqueeze(-1)                                     # (N,1)
    return x, y


class PsiMLP(nn.Module):
    """
    ψ(r, ω | θ) -> scalar radius'
    Input: [r, ωx, ωy, ωz]  (4D)
    Output: r' (1D)
    """
    def __init__(self, hidden=64, depth=3):
        super().__init__()
        layers = []
        in_dim = 4
        for i in range(depth):
            layers += [nn.Linear(in_dim if i == 0 else hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # No forced positivity; the mapping target is positive and MSE will drive it.
        return self.net(x)


def monotonicity_penalty(model, x, device="cpu"):
    """
    Compute hinge penalty for ∂ψ/∂r >= 0.
    We treat r as the first component of x.
    """
    x.requires_grad_(True)
    out = model(x)                   # (B,1)
    # Compute ∂ψ/∂r via autograd on the first input dim
    grad_outputs = torch.ones_like(out)
    grads = torch.autograd.grad(
        outputs=out,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]                              # (B,4), ∂ψ/∂x_i
    dpsi_dr = grads[:, 0]            # (B,)
    # Hinge: penalize non-negative derivatives
    penalty = F.relu(dpsi_dr) ** 2
    return penalty.mean(), dpsi_dr.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lambda_mono", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_N", type=int, default=80_000)
    parser.add_argument("--val_N", type=int, default=10_000)
    parser.add_argument("--R", type=float, default=10.0)
    parser.add_argument("--r_min", type=float, default=0.5)
    parser.add_argument("--r_max", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build data
    x_train, y_train = make_dataset(
        N=args.train_N, R=args.R, r_min=args.r_min, r_max=args.r_max, device=device
    )
    x_val, y_val = make_dataset(
        N=args.val_N, R=args.R, r_min=args.r_min, r_max=args.r_max, device=device
    )

    # Model + Optim
    model = PsiMLP(hidden=args.hidden, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Simple mini-batch loop
    num_batches = math.ceil(len(x_train) / args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Shuffle each epoch
        perm = torch.randperm(len(x_train), device=device)
        x_train = x_train[perm]
        y_train = y_train[perm]

        running_mse = 0.0
        running_mono = 0.0

        for b in range(num_batches):
            start = b * args.batch_size
            end = start + args.batch_size
            xb = x_train[start:end]
            yb = y_train[start:end]

            opt.zero_grad()
            pred = model(xb)
            mse = F.mse_loss(pred, yb)

            mono_pen, dpsi_dr = monotonicity_penalty(model, xb, device=device)
            loss = mse + args.lambda_mono * mono_pen
            loss.backward()
            opt.step()

            running_mse += mse.item() * len(xb)
            running_mono += (dpsi_dr >= 0).float().mean().item() * len(xb)

        train_mse = running_mse / len(x_train)
        train_viol = running_mono / len(x_train)

        # Validation
        model.eval()
        with torch.no_grad():
            pred_val = model(x_val)
            val_mse = F.mse_loss(pred_val, y_val).item()

        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"[{epoch:4d}/{args.epochs}] "
                  f"train_mse={train_mse:.6e}  val_mse={val_mse:.6e}  "
                  f"mono_viol_frac={train_viol:.4f}")

    # Final check: report a few samples
    model.eval()
    with torch.no_grad():
        idx = torch.randint(0, len(x_val), (5,), device=device)
        r = x_val[idx, 0]
        omega = x_val[idx, 1:]
        y_true = y_val[idx, 0]
        y_pred = model(x_val[idx]).squeeze(-1)
        print("\nExamples (r, true r', pred r'):")
        for i in range(len(idx)):
            print(f"r={r[i].item():.4f}  target={y_true[i].item():.6f}  pred={y_pred[i].item():.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
