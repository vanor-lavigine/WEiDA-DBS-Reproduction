"""Exact K4 multi-start objective with equivalent permutations evaluated once.

Members and their positive bootstrap counts are passed separately. A scan's
own deterministic peak decomposition can be cached across participant draws.
"""
import numpy as np
from itertools import permutations
from scipy.optimize import linear_sum_assignment

PERMS = np.array(list(permutations(range(4))))
AX = np.arange(4)


def assignment(sim):
    return PERMS[(sim[..., PERMS, AX] ** 2).sum(axis=-1).argmax(axis=-1)]


def orient_axes(axes):
    return axes * np.where(axes[AX, np.abs(axes).argmax(axis=1)] < 0, -1., 1.)[:, None]


def fit4(maps, weights):
    """All legacy starts, collapsing assignments identical up to label permutation."""
    maps = np.asarray(maps)
    w = np.asarray(weights, float)
    if np.any(w <= 0) or maps.shape[:2] != (len(w),4):
        raise ValueError("Expected four maps per positively weighted scan")
    # Compute all first assignments in a single matrix product.
    n, _, d = maps.shape
    all_r = (maps.reshape(-1,d) @ maps.reshape(-1,d).T).reshape(n,4,n,4)
    starts = assignment(all_r.transpose(2,0,1,3))
    seen = set()
    best = None
    best_loss = -np.inf
    for start in starts:
        take = start
        for _ in range(300):
            # First participant fixes a representative of the equivalent labelings.
            take = take[:,take[0].argsort()]
            key = take.tobytes()
            if key in seen:
                break
            seen.add(key)
            x = maps[np.arange(n)[:, None], take].transpose(1,0,2)
            xw = x * np.sqrt(w)[None,:,None]
            # Smaller Gram matrix, mapped back into ROI space.
            if n < d:
                ev, u = np.linalg.eigh(xw @ xw.transpose(0,2,1))
                axes = np.einsum('kn,knd->kd',u[:,:,-1],xw) / np.sqrt(ev[:,-1,None])
            else:
                ev, u = np.linalg.eigh(xw.transpose(0,2,1) @ xw)
                axes = u[:,:,-1]
            nxt = assignment(maps @ axes.T)
            if np.array_equal(nxt,take):
                score = ev[:,-1].sum() / (4*w.sum())
                if score > best_loss + 1e-12:
                    best_loss, best = score, orient_axes(axes)
                break
            take = nxt
        else:
            raise RuntimeError("AAHC group fit failed to converge")
    if best is None:
        raise RuntimeError("No stationary assignment")
    return best


def reference_match(maps, ref):
    s = np.abs(ref @ maps.T)
    idx, order = linear_sum_assignment(-s)
    best = s[idx,order]
    s[idx,order] = -np.inf
    margin = best - np.maximum(s.max(axis=1),s.max(axis=0)[order])
    return maps[order], best, margin
