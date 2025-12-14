import torch, random
from torch import nn
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True; torch.set_float32_matmul_precision('high')
device='cuda'

############################################################################################################################################

def rosa(x):
	n=len(x); y=[-1]*n; s=2*n+1; b=[None]*s; c=[-1]*s; d=[0]*s; e=[-1]*s; b[0]={}; g=0; z=1
	for i,t in enumerate(x):
		r=z; z+=1; b[r]={}; d[r]=d[g]+1; p=g
		while p!=-1 and t not in b[p]: b[p][t]=r; p=c[p]
		if p==-1: c[r]=0
		else:
			q=b[p][t]
			if d[p]+1==d[q]: c[r]=q
			else:
				u=z; z+=1; b[u]=b[q].copy(); d[u]=d[p]+1; c[u]=c[q]; e[u]=e[q]
				while p!=-1 and b[p][t]==q: b[p][t]=u; p=c[p]
				c[q]=c[r]=u
		v=g=r; a=-1
		while v!=-1:
			if d[v]>0 and e[v]>=0: a=x[e[v]+1]; break
			v=c[v]
		y[i]=a; v=g
		while v!=-1 and e[v]<i: e[v]=i; v=c[v]
	return y

def rosa_torch(z: torch.Tensor) -> torch.Tensor:
    assert z.dtype==torch.long and z.ndim==2
    zc = z.detach().contiguous().cpu()
    return torch.stack([torch.as_tensor(rosa(r.tolist()), dtype=torch.long) for r in zc]).to(z.device)

class Emb_ROSA(nn.Module):
    def __init__(s,V,C):
        super().__init__()
        s.emb = nn.Embedding(V,C)
    def forward(s,idx):
        idx = rosa_torch(idx)
        out = s.emb(idx.clamp_min(0))
        return out.masked_fill(idx.eq(-1).unsqueeze(-1), 0.0)

############################################################################################################################################

class ROSA_1bit(torch.autograd.Function): # !!! extremely slow !!!
    @staticmethod
    def forward(ctx, x, emb0, emb1, tau: float):
        B,T,C = x.shape
        bits = (x>0).to(torch.long)
        idx = torch.empty_like(bits)
        for b in range(B):
            for c in range(C):
                idx[b,:,c] = torch.tensor(rosa(bits[b,:,c].tolist()), dtype=torch.long, device=x.device)
        e0 = emb0.expand_as(x); e1 = emb1.expand_as(x)
        out = torch.where(idx.eq(-1), torch.zeros_like(x), torch.where(idx.eq(1), e1, e0))
        ctx.save_for_backward(bits, idx, x, emb0, emb1)
        ctx.tau = float(tau)
        return out
    @staticmethod
    def backward(ctx, gy):
        bits, idx, x, emb0, emb1 = ctx.saved_tensors
        tau = ctx.tau
        B,T,C = x.shape
        mask0 = idx.eq(0).to(gy.dtype)
        mask1 = idx.eq(1).to(gy.dtype)
        g_emb0 = (gy * mask0).sum(dim=(0,1), keepdim=True)
        g_emb1 = (gy * mask1).sum(dim=(0,1), keepdim=True)
        gx = torch.zeros_like(x)
        e0v = emb0.view(-1)
        e1v = emb1.view(-1)
        for b in range(B):
            print('doing bwd for sample', b, 'in batch')
            for c in range(C):
                row_bits = bits[b,:,c].tolist()
                base_idx = idx[b,:,c].tolist()
                vrow = gy[b,:,c].detach().cpu().tolist()
                e0c = float(e0v[c]); e1c = float(e1v[c])
                base_phi = 0.0 # base phi for reuse
                for t in range(T):
                    it = base_idx[t]
                    if it==1: base_phi += vrow[t]*e1c
                    elif it==0: base_phi += vrow[t]*e0c                
                def phi_from_idx(idx_list): # helper to score an idx
                    s = 0.0
                    for t in range(T):
                        it = idx_list[t]
                        if it==1: s += vrow[t]*e1c
                        elif it==0: s += vrow[t]*e0c
                    return s                
                for t in range(T): # get gradient by flipping this bit
                    mag = max(abs(float(x[b,t,c].item())), tau)
                    if row_bits[t]==1:
                        phi_pos = base_phi
                    else:
                        seq = list(row_bits); seq[t]=1
                        phi_pos = phi_from_idx(rosa(seq))
                    if row_bits[t]==0:
                        phi_neg = base_phi
                    else:
                        seq = list(row_bits); seq[t]=0
                        phi_neg = phi_from_idx(rosa(seq))
                    gx[b,t,c] = (phi_pos - phi_neg) / (2.0*mag)
        return gx, g_emb0, g_emb1, None

class ROSA_1bit_LAYER(nn.Module): # !!! extremely slow !!!
    def __init__(self, C: int, tau: float = 1e-3):
        super().__init__()
        self.emb0 = nn.Parameter(torch.full((1,1,C), -1e-5)) # init
        self.emb1 = nn.Parameter(torch.full((1,1,C),  1e-5)) # init
        self.tau = tau
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ROSA_1bit.apply(x, self.emb0, self.emb1, self.tau)

class _HNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, s_mean, s_var, p_mean, p_var):
        x = torch.stack([s_mean, s_var, p_mean, p_var], dim=-1)
        return self.net(x).squeeze(-1)

@torch.no_grad()
def _rosa_bits_batch(bits: torch.Tensor) -> torch.Tensor:
    # bits: [B,T,C] long {0,1} -> idx: [B,T,C] long in {-1,0,1}
    B,T,C = bits.shape
    idx = torch.empty_like(bits)
    bcpu = bits.detach().cpu()
    for b in range(B):
        for c in range(C):
            idx[b,:,c] = torch.tensor(rosa(bcpu[b,:,c].tolist()), dtype=torch.long, device=bits.device)
    return idx

def _summ(bits2d: torch.Tensor, p2d: torch.Tensor):
    # [B,D] -> 4 scalars per sample
    s_mean = bits2d.mean(-1)
    s_var  = bits2d.var(-1, unbiased=False)
    p_mean = p2d.mean(-1)
    p_var  = p2d.var(-1, unbiased=False)
    return s_mean, s_var, p_mean, p_var

class ROSA_1bit_RODEO_BWD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, emb0, emb1, tau_forward: float, tau_prob: float, hnet: nn.Module, M: int):
        B,T,C = x.shape
        bits = (x > 0).to(torch.long)
        idx = _rosa_bits_batch(bits)
        e0 = emb0.expand_as(x); e1 = emb1.expand_as(x)
        out = torch.where(idx.eq(-1), torch.zeros_like(x), torch.where(idx.eq(1), e1, e0))

        ctx.save_for_backward(x, idx, emb0, emb1)
        ctx.tau_forward = float(tau_forward)
        ctx.tau_prob = float(tau_prob)
        ctx.hnet = hnet
        ctx.M = int(M)
        return out

    @staticmethod
    def backward(ctx, gy):
        x, idx_det, emb0, emb1 = ctx.saved_tensors
        tau_prob = ctx.tau_prob
        hnet = ctx.hnet
        M = ctx.M
        B,T,C = x.shape
        D = T*C

        mask0 = idx_det.eq(0).to(gy.dtype)
        mask1 = idx_det.eq(1).to(gy.dtype)
        g_emb0 = (gy * mask0).sum(dim=(0,1), keepdim=True)
        g_emb1 = (gy * mask1).sum(dim=(0,1), keepdim=True)

        logits = x / float(tau_prob)
        p = torch.sigmoid(logits).clamp_(1e-4, 1-1e-4)

        u = torch.rand_like(p)
        s1 = (u < p).to(torch.long)
        s2 = (u > (1.0 - p)).to(torch.long)

        idx1 = _rosa_bits_batch(s1)
        idx2 = _rosa_bits_batch(s2)

        e0 = emb0.expand_as(x); e1 = emb1.expand_as(x)
        out1 = torch.where(idx1.eq(-1), torch.zeros_like(x), torch.where(idx1.eq(1), e1, e0))
        out2 = torch.where(idx2.eq(-1), torch.zeros_like(x), torch.where(idx2.eq(1), e1, e0))

        f1 = (out1 * gy).sum(dim=(1,2))
        f2 = (out2 * gy).sum(dim=(1,2))

        # ---- (A) K=2 RLOO / DisARM-like score gradient
        # d/dlogit log q(s) = s - p
        s1f = s1.to(gy.dtype)
        s2f = s2.to(gy.dtype)
        g_logit_rloo = ((f1 - f2).view(B,1,1) * (s1f - s2f)) * 0.5  # [B,T,C]

        p_flat  = p.detach().reshape(B, D)
        s1_flat = s1f.reshape(B, D)
        s2_flat = s2f.reshape(B, D)

        def h_of(bits_flat):
            s_mean, s_var, p_mean, p_var = _summ(bits_flat, p_flat)
            return hnet(s_mean, s_var, p_mean, p_var)  # [B]

        coords = torch.randint(0, D, (M,), device=x.device)

        def stein_cv(bits_flat):
            hval = h_of(bits_flat.detach())  # [B]
            cv = torch.zeros_like(bits_flat)
            for j in range(M):
                i = int(coords[j].item())
                pi = p_flat[:, i]
                si = bits_flat[:, i]

                b1 = bits_flat.clone(); b1[:, i] = 1.0
                b0 = bits_flat.clone(); b0[:, i] = 0.0
                h1 = h_of(b1)
                h0 = h_of(b0)

                Eh_tilde = pi*(h1*(1.0-pi)) + (1.0-pi)*(h0*(0.0-pi))
                htilde   = hval*(si - pi)
                Ahtilde  = Eh_tilde - htilde

                cv[:, i] = Ahtilde * (D / float(M))
            return cv

        g_logit = g_logit_rloo

        gx = g_logit * (1.0 / float(tau_prob))

        return gx, g_emb0, g_emb1, None, None, None, None

class ROSA_1bit_LAYER_RODEO(nn.Module):
    def __init__(self, C: int, tau_forward: float = 1e-3, tau_prob: float = 0.2, M: int = 8):
        super().__init__()
        self.emb0 = nn.Parameter(torch.full((1,1,C), -1e-5))
        self.emb1 = nn.Parameter(torch.full((1,1,C),  1e-5))
        self.tau_forward = tau_forward
        self.tau_prob = tau_prob
        self.hnet = _HNet(hidden=64)
        self.M = M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ROSA_1bit_RODEO_BWD.apply(x, self.emb0, self.emb1, self.tau_forward, self.tau_prob, self.hnet, self.M)


############################################################################################################################################

V,C,B,T,steps=11,64,128,128,1000
lr0,lr1=1e-3,1e-6

print('Training EmbROSA + ROSA 1bit (EXTREMELY SLOW)')

def batch(B,T,nn=None):
    s=[]
    for _ in range(B):
        if nn == None:
            k=random.randint(1,3); lo=0 if k==1 else 10**(k-1); n=random.randint(lo,10**k-1)
        else:
            assert B == 1
            n = nn
        a=[10]
        while len(a)<T:
            a+=[ord(c)-48 for c in str(n)]+[10]; n+=1
        s.append(a[:T])
    return torch.tensor(s,device=device,dtype=torch.long)

import time
import numpy as np

class MODEL(nn.Module):
    def __init__(s, use_rodeo: bool):
        super().__init__()
        s.e=nn.Embedding(V,C)
        s.emb_rosa=Emb_ROSA(V,C)
        if use_rodeo:
            s.rosa_1bit = ROSA_1bit_LAYER_RODEO(C, tau_forward=1e-3, tau_prob=0.2, M=8)
        else:
            s.rosa_1bit = ROSA_1bit_LAYER(C)
        s.o=nn.Linear(C,V)

    def forward(s,x):
        x = s.e(x) + s.emb_rosa(x)
        x = x + s.rosa_1bit(x)
        x = s.o(x)
        return x

@torch.no_grad()
def eval_acc(model, samples=64, T_eval=128):
    # model acc (next-token)
    model.eval()
    correct = 0
    total = 0
    for _ in range(samples):
        x = batch(1, T_eval)
        y = x[:,1:]
        xin = x[:,:-1]
        pred = model(xin).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct/total

@torch.no_grad()
def eval_rosa_baseline_acc(samples=64, T_eval=128):
    correct = 0
    total = 0
    for _ in range(samples):
        x = batch(1, T_eval)
        y = x[:,1:]
        r = rosa_torch(x)[:, :-1]  # align to predict y
        correct += (r == y).sum().item()
        total += y.numel()
    return correct/total

def run_train(use_rodeo: bool, eval_samples=64, T_eval=128):
    model = MODEL(use_rodeo).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr1)

    losses = []
    step_times = []

    model.train()
    t_global0 = time.perf_counter()

    for step in range(steps):
        t0 = time.perf_counter()

        x = batch(B,T)
        y = x[:,1:]
        xin = x[:,:-1]

        z = model(xin)
        loss = F.cross_entropy(z.reshape(-1,V), y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()

        dt = time.perf_counter() - t0
        step_times.append(dt)
        losses.append(float(loss.item()))

        print(f'{"RODEO" if use_rodeo else "FLIP"} {step+1}/{steps} '
              f'loss {losses[-1]:.4f} lr {sch.get_last_lr()[0]:.6g} step_time {dt:.3f}s')

    train_time = time.perf_counter() - t_global0

    # eval
    acc = eval_acc(model, samples=eval_samples, T_eval=T_eval)
    rosa_acc = eval_rosa_baseline_acc(samples=eval_samples, T_eval=T_eval)

    stats = {
        "use_rodeo": use_rodeo,
        "train_time_s": train_time,
        "mean_step_time_s": float(np.mean(step_times)),
        "p50_step_time_s": float(np.percentile(step_times, 50)),
        "p90_step_time_s": float(np.percentile(step_times, 90)),
        "final_loss": losses[-1],
        "best_loss": float(np.min(losses)),
        "acc": acc,
        "rosa_acc": rosa_acc,
        "losses": losses,
        "step_times": step_times,
    }
    return model, stats

print("="*80)
print("Run 2/2: RODEO estimator")
model_rodeo, stats_rodeo = run_train(use_rodeo=True, eval_samples=64, T_eval=128)

def print_stats(s):
    name = "RODEO" if s["use_rodeo"] else "FLIP"
    print(f"\n[{name}] summary")
    print(f"  train_time_s      : {s['train_time_s']:.2f}")
    print(f"  mean_step_time_s  : {s['mean_step_time_s']:.3f} (p50 {s['p50_step_time_s']:.3f}, p90 {s['p90_step_time_s']:.3f})")
    print(f"  best_loss         : {s['best_loss']:.4f}")
    print(f"  final_loss        : {s['final_loss']:.4f}")
    print(f"  final_acc         : {s['acc']:.4f}")
    print(f"  ROSA baseline acc : {s['rosa_acc']:.4f}")

# print_stats(stats_flip)
print_stats(stats_rodeo)

# print("\nSpeedup (mean step time):", stats_flip["mean_step_time_s"] / max(stats_rodeo["mean_step_time_s"], 1e-9))

def print_loss_curve(losses, every=50, name=""):
    print(f"\n{name} loss curve (every {every} steps):")
    for i in range(0, len(losses), every):
        print(f"  step {i+1:4d}: {losses[i]:.4f}")
    print(f"  step {len(losses):4d}: {losses[-1]:.4f}")

# print_loss_curve(stats_flip["losses"], every=50, name="[FLIP]")
print_loss_curve(stats_rodeo["losses"], every=50, name="[RODEO]")

# -------------------------
# 7) Qualitative sample print (use the right model!)
# -------------------------
@torch.no_grad()
def show_samples(model, title, num_samples=5):
    S='0123456789A'
    print("\n" + "#"*40)
    print(title)
    print("#"*40)
    model.eval()
    for SAMPLE in range(num_samples):
        x = batch(1,128,int(3.5**(SAMPLE+1)))
        y = x[:,1:]
        pred = model(x[:,:-1]).argmax(-1)
        n = y.numel()

        r = rosa_torch(x)[:,:-1]
        rr = ''.join([S[t] if t >= 0 else 'X' for t in r[0].tolist()])

        xx = ''.join(S[t] for t in x[0,:-1].tolist())
        yy = ''.join(S[t] for t in y[0].tolist())
        zz = ''.join(S[t] for t in pred[0].tolist())

        ry = ''.join('.' if r[0,i].item()==y[0,i].item() else '^' for i in range(y.size(1)))
        zy = ''.join('.' if pred[0,i].item()==y[0,i].item() else '^' for i in range(y.size(1)))

        nry = (r==y).sum().item()
        nzy = (pred==y).sum().item()

        print('in  ',xx)
        print('gold',yy)
        print('rosa',rr)
        print('diff',ry)
        print(f'rosa_correct {nry}/{n}  acc {nry/n:.3f}')
        print('pred',zz)
        print('diff',zy)
        print(f'model_correct {nzy}/{n} acc {nzy/n:.3f}')
        print('-'*80)

# show_samples(model_flip,  "Qualitative check: FLIP model")
show_samples(model_rodeo, "Qualitative check: RODEO model")
