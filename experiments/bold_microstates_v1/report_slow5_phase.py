"""Render group correlation estimates and confidence intervals from the 105 analysis."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy import stats


def render(path):
    result = json.loads(path.read_text(encoding="utf8"))
    assert result["data_kind"] == "real" and len(result["effects"]) == 3
    available = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((f for f in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei") if f in available), None)
    if cjk:
        plt.rcParams["font.family"] = [cjk]
    fig, ax = plt.subplots(figsize=(8.6, 3.4), layout="constrained")
    labels = (["关机指标 与 关机运动评分", "开机指标 与 开机运动评分", "指标变化 与 运动改善"]
              if cjk else ["OFF metric vs OFF motor score", "ON metric vs ON motor score",
                           "Metric change vs motor improvement"])
    for i, row in enumerate(result["effects"]):
        r, n = row["r"], row["n"]
        t = r*np.sqrt((n-2)/(1-r**2))
        np.testing.assert_allclose(row["p_raw"], 2*stats.t.sf(abs(t), n-2), rtol=1e-10)
        expected = np.tanh(np.arctanh(r)+np.array([-1,1])*stats.norm.ppf(.975)/np.sqrt(n-3))
        np.testing.assert_allclose(expected, [row["ci_low"], row["ci_high"]], atol=1e-12)
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color="#175d76", lw=2.2)
        ax.scatter(r, i, s=58, color="#175d76", zorder=3)
        ax.annotate(f"{r:+.3f}", (r, i), xytext=(0, 10), textcoords="offset points", ha="center")
    ax.set(xlim=(-1, 1), ylim=(2.55, -.55), yticks=range(3), yticklabels=labels)
    ax.axvline(0, color="#888888", lw=1, ls="--", zorder=0)
    ax.set_xlabel("Pearson r（线段为逐项 95% 置信区间）" if cjk else "Pearson r (95% confidence intervals)")
    ax.set_title("slow-5 全脑相位同步波动与运动评分（25 人）" if cjk else
                 "Slow-5 global phase synchrony variability and motor score (25 patients)", loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    fig.savefig(path.with_name("correlations.png"), dpi=175, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    render(parser.parse_args().results)
