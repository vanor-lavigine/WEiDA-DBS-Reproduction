"""Gap-aware backfitting and state metrics. Labels are zero-based internally."""
import numpy as np
import models_v3  # makes the frozen legacy implementation importable
from peak_states import interpolate_peaks

TR = 2.1


def blocks(valid):
    return np.flatnonzero(np.diff(np.r_[False, valid, False])).reshape(-1, 2)


def minimum_two(scores, valid):
    """Maximum total similarity subject to duration >=2 TR, including boundary runs."""
    result = np.full(len(scores), -1, int)
    for start, stop in blocks(valid):
        seg = scores[start:stop]
        n, k = seg.shape
        if n < 2:
            continue
        first, mature = seg[0].copy(), np.full(k, -np.inf)
        predecessor = np.full((n,k,2), -1, int)
        axis = np.arange(k)
        for t in range(1,n):
            stay = mature >= first
            predecessor[t,:,1] = 2*axis + stay.astype(int)
            # Determine the best mature predecessor other than the new class.
            ordered = np.argsort(-mature, kind="stable")
            other = np.where(axis == ordered[0], ordered[1], ordered[0])
            predecessor[t,:,0] = 2*other + 1
            first, mature = mature[other] + seg[t], np.maximum(mature,first) + seg[t]
        node = 2*mature.argmax()+1
        for t in range(n-1,-1,-1):
            j, phase = divmod(node,2)
            result[start+t] = j
            node = predecessor[t,j,phase]
    return result


def backfill(p, fitted, rule="B1"):
    signed_similarity = p.units @ fitted.maps.T
    similarity = np.abs(signed_similarity) if fitted.polarity == "P0" else signed_similarity
    # Resolve template variants before computing episodes.
    n = len(fitted.stable)
    scores = np.full((len(p.valid), n), -np.inf)
    for j, family in enumerate(fitted.family):
        scores[:,family] = np.maximum(scores[:,family], similarity[:,j])
    labels = scores.argmax(axis=1)
    if rule == "B0":
        labels, _ = interpolate_peaks(len(labels), p.peaks, labels[p.peaks], p.valid, scores)
    elif rule == "B2":
        labels = minimum_two(scores, p.valid)
    elif rule != "B1":
        raise ValueError(rule)
    labels[~p.valid] = -1
    if fitted.polarity == "P1" and rule == "B1":
        axes = np.abs(signed_similarity[:,::2])
        winners = axes.argmax(axis=1)
        unique = np.diff(np.sort(axes,axis=1)[:,-2:],axis=1)[:,0] > 1e-12
        check = p.valid & unique
        np.testing.assert_array_equal(labels[check]//2, winners[check])
    return labels, scores, signed_similarity


def segments(labels):
    cuts = np.r_[0,np.flatnonzero(np.diff(labels))+1,len(labels)]
    return [(int(a),int(b),int(labels[a])) for a,b in zip(cuts[:-1],cuts[1:])
            if labels[a]>=0]


def measure(p, labels, scores, edge_censor=False):
    """Calculate exposure, transitions and episode features from merged labels.

    Boundary sensitivity discards each boundary-touching run. Exposure and
    counts then both refer to the continuously retained frames of that analysis.
    """
    if edge_censor:
        labels = labels.copy()
        for start, stop in blocks(labels >= 0):
            runs = segments(labels[start:stop])
            labels[start:start+runs[0][1]] = -1
            labels[start+runs[-1][0]:stop] = -1
    valid = labels>=0
    if not valid.any():
        return {}
    n, k = valid.sum(), scores.shape[1]
    runs = np.asarray(segments(labels))
    a,b,cls = runs.T
    durations = (b-a)*TR
    exposure = n*TR
    transitions = np.zeros((k,k),int)
    adj = valid[:-1] & valid[1:]
    np.add.at(transitions,(labels[:-1][adj], labels[1:][adj]),1)
    changes = transitions.sum()-transitions.trace()
    chosen = scores[valid,labels[valid]]
    gfp2 = p.amplitude[valid]**2
    order = np.sort(scores[valid],axis=1)
    margin = order[:,-1] - order[:,-2]
    state_metrics = {}
    avg = []
    for i in range(k):
        key = "_"+str(i+1)
        take = cls==i
        frames = labels[valid]==i
        duration = durations[take].mean() if take.any() else np.nan
        if take.any():
            avg.append(duration)
        recurrence = []
        for st, en in blocks(valid):
            contained = take & (a>=st) & (b<=en)
            recurrence.extend((a[contained][1:] - b[contained][:-1])*TR)
        fields = dict(coverage=frames.sum()/n, occurrence=take.sum()/exposure,
                      count=int(take.sum()), duration=duration,
                      duration_sd=durations[take].std(ddof=1) if take.sum()>1 else np.nan,
                      recurrence=np.mean(recurrence) if recurrence else np.nan,
                      gev=np.sum(gfp2[frames]*chosen[frames]**2)/gfp2.sum())
        state_metrics.update({name+key:value for name,value in fields.items()})
    counts = transitions.copy()
    np.fill_diagonal(counts,0)
    denom = counts.sum(axis=1)
    prob = np.divide(counts,denom[:,None],out=np.zeros((k,k)),where=denom[:,None]>0)
    entropy = -(prob*np.log2(np.where(prob>0,prob,1))).sum(axis=1)
    for i,j in np.ndindex(k,k):
        state_metrics[f"transition_{i+1}_{j+1}"] = prob[i,j] if denom[i] else np.nan
    return dict(duration_s=exposure,n_states=k,switch_count=int(changes),
                switching_rate=changes/exposure,global_occurrence=len(runs)/exposure,
                global_pooled_duration=durations.mean(),global_equal_duration=np.mean(avg),
                mean_similarity=chosen.mean(),mean_margin=margin.mean(),
                low_margin_fraction=(margin<0.05).mean(),
                transition_entropy=np.average(entropy,weights=denom) if changes else np.nan,
                **state_metrics)
