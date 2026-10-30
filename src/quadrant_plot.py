"""Shared quadrant chart used by Module 1 (regime) and Module 2 (cycle clock).

Design:
  - quadrants are soft tints in the module's regime/phase colour; names sit in the corners with a
    colour chip, so identity never rests on colour alone
  - history is a recessive grey trail; the last `recent` months are drawn in ink with arrowheads,
    so the eye goes to where the economy is heading, not to the oldest points
  - one dot per month, January of each year labelled (no colourbar needed)
  - the latest reading gets a callout with its coordinates and state
  - axes fit the data (origin always visible) instead of a fixed symmetric square
"""
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch

INK = "#1f1f1e"
INK_2 = "#52514e"
MUTED = "#8c8b86"
TRAIL = "#bdbcb6"
SURFACE = "#fcfcfb"
ACCENT = "#d6336c"
BAND = "#8c8b86"   # latest reading / nowcast only


def _font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]:
        if fam in names:
            return fam
    return "sans-serif"


def _crosses(p1, p2, p3, p4):
    """True if segment p1-p2 properly intersects segment p3-p4."""
    d = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (d(p1, p2, p3) * d(p1, p2, p4) < 0) and (d(p3, p4, p1) * d(p3, p4, p2) < 0)


def plot_quadrant_path(x, y, labels, path, *, quads, colors, display, band_x, band_y,
                       xlabel, ylabel, title, subtitle, months=60, recent=12,
                       nowcast=None, transition="Transition", footer=None):
    """x, y: monthly Series. labels: monthly state Series.
    quads: {state: (sx, sy)} with sx, sy in {-1, +1} giving the quadrant of each state.
    nowcast: optional (x, y, date) drawn as a hollow marker with a dashed arrow from the latest point.
    """
    tr = pd.concat([x, y], axis=1, keys=["x", "y"], sort=True).dropna().iloc[-months:]
    lab = labels.reindex(tr.index)
    share = lab.value_counts(normalize=True).mul(100)
    n = len(tr)
    recent = min(recent, n)

    plt.rcParams.update({"font.family": _font(), "axes.unicode_minus": True})
    fig, ax = plt.subplots(figsize=(10, 8.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # ---- limits: fit data (+ nowcast), always include origin and the bands, then pad
    xs = list(tr["x"]) + ([nowcast[0]] if nowcast is not None else []) + [-band_x, band_x]
    ys = list(tr["y"]) + ([nowcast[1]] if nowcast is not None else []) + [-band_y, band_y]
    def lim(v):
        lo, hi = min(min(v), 0), max(max(v), 0)
        span = max(hi - lo, 1.0)
        lo, hi = lo - 0.18 * span, hi + 0.18 * span
        side = 0.42 * (hi - lo)          # every quadrant keeps room for its corner label
        lo, hi = min(lo, -side), max(hi, side)
        return np.floor(lo * 4) / 4, np.ceil(hi * 4) / 4
    (x0, x1), (y0, y1) = lim(xs), lim(ys)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    X0, X1, Y0, Y1 = x0, x1, y0, y1

    # ---- quadrant tints + transition cross
    for st, (sx, sy) in quads.items():
        rx = (0, X1) if sx > 0 else (X0, 0)
        ry = (0, Y1) if sy > 0 else (Y0, 0)
        ax.add_patch(Rectangle((rx[0], ry[0]), rx[1] - rx[0], ry[1] - ry[0],
                               facecolor=colors[st], alpha=0.10, lw=0, zorder=0))
    ax.add_patch(Rectangle((-band_x, Y0), 2 * band_x, Y1 - Y0, facecolor=BAND, alpha=0.12, lw=0, zorder=0.5))
    ax.add_patch(Rectangle((X0, -band_y), X1 - X0, 2 * band_y, facecolor=BAND, alpha=0.12, lw=0, zorder=0.5))
    ax.axhline(0, color=INK_2, lw=0.9, zorder=1); ax.axvline(0, color=INK_2, lw=0.9, zorder=1)

    # ---- corner labels: name (bold ink) over share (secondary ink), framed in the state colour
    for st, (sx, sy) in quads.items():
        tx = 0.985 if sx > 0 else 0.015
        ty = 0.975 if sy > 0 else 0.025
        ha = "right" if sx > 0 else "left"
        name, *rest = display[st].split(" (")
        sub = f"{share.get(st, 0):.0f}% of months" + (f"  ·  {rest[0].rstrip(')')}" if rest else "")
        va = "top" if sy > 0 else "bottom"
        # invisible full-text box sizes the frame to both lines; the visible lines sit on top
        ax.text(tx, ty, f"{name}\n{sub}", transform=ax.transAxes, ha=ha, va=va, fontsize=10.5,
                fontweight="bold", color=(0, 0, 0, 0), linespacing=1.6, zorder=6,
                bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.25", facecolor=SURFACE,
                          edgecolor=colors[st], lw=1.3, alpha=0.95))
        ax.text(tx, ty, f"{name}\n ", transform=ax.transAxes, ha=ha, va=va, fontsize=10.5,
                fontweight="bold", color=INK, linespacing=1.6, zorder=7)
        ax.text(tx, ty, f" \n{sub}", transform=ax.transAxes, ha=ha, va=va, fontsize=9.5,
                color=INK_2, linespacing=1.77, zorder=7)

    # ---- trail: older months recessive, recent months in ink with direction arrows
    old = tr.iloc[: n - recent + 1]
    new = tr.iloc[n - recent:]
    ax.plot(old["x"], old["y"], color=TRAIL, lw=1.1, zorder=2, solid_capstyle="round")
    ax.scatter(old["x"].iloc[:-1], old["y"].iloc[:-1], s=22, color=TRAIL, edgecolor=SURFACE, lw=0.8, zorder=3)
    for k in range(len(new) - 1):
        a = (new["x"].iloc[k], new["y"].iloc[k]); b = (new["x"].iloc[k + 1], new["y"].iloc[k + 1])
        alpha = 0.35 + 0.65 * (k + 1) / (len(new) - 1)
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.8, color=INK,
                                     alpha=alpha, shrinkA=4, shrinkB=4, zorder=4))
    ax.scatter(new["x"].iloc[:-1], new["y"].iloc[:-1], s=34, color=INK_2, edgecolor=SURFACE, lw=1.2, zorder=5)

    # ---- January markers (year labels) + start of window
    marks = [tr.index[0]] + [d for d in tr.index[1:-1] if d.month == 1]
    for d in marks:
        px, py = tr.loc[d, "x"], tr.loc[d, "y"]
        ax.annotate(d.strftime("%b %Y") if d == tr.index[0] else d.strftime("%Y"), (px, py),
                    xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=MUTED, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.75))

    # ---- latest reading
    lx, ly, ld = tr["x"].iloc[-1], tr["y"].iloc[-1], tr.index[-1]
    st_now = lab.iloc[-1]
    if st_now == transition or pd.isna(st_now):
        near = next(s for s, (sx, sy) in quads.items() if np.sign(lx or 1) == sx and np.sign(ly or 1) == sy)
        state_txt = f"Transition, leaning {display[near].split(' (')[0]}"
    else:
        state_txt = display[st_now].split(" (")[0]
    ax.scatter(lx, ly, s=170, color=ACCENT, edgecolor=SURFACE, lw=2.2, zorder=8)
    ax.scatter(lx, ly, s=520, facecolor="none", edgecolor=ACCENT, lw=1.2, alpha=0.45, zorder=7)
    # place the callout where it covers the fewest points (and stays clear of the corner labels)
    fx = lambda v: (v - X0) / (X1 - X0)
    fy = lambda v: (v - Y0) / (Y1 - Y0)
    pts = np.c_[fx(tr["x"].values), fy(tr["y"].values)]
    if nowcast is not None:
        pts = np.vstack([pts, [fx(nowcast[0]), fy(nowcast[1])]])
    trail = pts[: len(tr)]
    W, H = 0.30, 0.13
    best, ox0, oy0 = None, fx(lx), fy(ly)
    for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        for r in (0.14, 0.20, 0.27):
            cx, cy = ox0 + r * np.cos(ang), oy0 + r * np.sin(ang)
            bx0 = cx if np.cos(ang) >= 0 else cx - W
            by0 = cy if np.sin(ang) >= 0 else cy - H
            if bx0 < 0.01 or by0 < 0.01 or bx0 + W > 0.99 or by0 + H > 0.99:
                continue
            cover = ((pts[:, 0] > bx0 - 0.02) & (pts[:, 0] < bx0 + W + 0.02) &
                     (pts[:, 1] > by0 - 0.02) & (pts[:, 1] < by0 + H + 0.02)).sum()
            corner = sum(1 for cxl in (0, 1) for cyl in (0, 1)
                         if abs((bx0 + W / 2) - cxl) < 0.5 * W + 0.24 and abs((by0 + H / 2) - cyl) < 0.5 * H + 0.11)
            # points the leader line would cross
            seg = np.array([cx - ox0, cy - oy0]); t = np.clip(((pts - [ox0, oy0]) @ seg) / (seg @ seg), 0.15, 1)
            near = (np.hypot(*(pts - ([ox0, oy0] + t[:, None] * seg)).T) < 0.025).sum()
            cross = sum(_crosses((ox0, oy0), (cx, cy), tuple(trail[k]), tuple(trail[k + 1]))
                        for k in range(len(trail) - 3))
            score = cover * 10 + corner * 25 + near * 4 + cross * 6 + r * 20
            if best is None or score < best[0]:
                best = (score, cx, cy, np.cos(ang) >= 0, np.sin(ang) >= 0)
    _, cx, cy, right, up = best
    ax.annotate(f"{ld:%b %Y}  ·  {state_txt}\n{xlabel.split(',')[0].split(':')[0].strip()} {lx:+.2f}    "
                f"{ylabel.split(',')[0].split(':')[0].strip()} {ly:+.2f}",
                (lx, ly), xytext=(cx, cy), textcoords="axes fraction",
                ha="left" if right else "right", va="bottom" if up else "top",
                fontsize=9.5, color=INK, linespacing=1.55, zorder=9, multialignment="left",
                bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.3", facecolor="white",
                          edgecolor=ACCENT, lw=1.2),
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1, shrinkA=0, shrinkB=10))

    handles = [plt.Line2D([], [], color=TRAIL, lw=1.1, marker="o", ms=4.5, mec=SURFACE,
                          label=f"{tr.index[0]:%b %Y} – {tr.index[n - recent]:%b %Y}"),
               plt.Line2D([], [], color=INK, lw=1.8, marker="o", ms=5, mfc=INK_2, mec=SURFACE,
                          label=f"Last {recent} months (arrows = direction)"),
               plt.Line2D([], [], ls="", marker="o", ms=10, color=ACCENT, mec=SURFACE, label=f"Latest, {ld:%b %Y}")]
    if nowcast is not None and np.isfinite(nowcast[0]) and np.isfinite(nowcast[1]):
        nx, ny, nd = nowcast
        ax.add_patch(FancyArrowPatch((lx, ly), (nx, ny), arrowstyle="-|>", mutation_scale=12, lw=1.3,
                                     color=ACCENT, ls=(0, (4, 3)), shrinkA=8, shrinkB=9, zorder=7))
        ax.scatter(nx, ny, marker="D", s=95, facecolor="white", edgecolor=ACCENT, lw=2, zorder=8)
        handles.append(plt.Line2D([], [], ls="", marker="D", ms=8, mfc="white", mec=ACCENT, mew=2,
                                  label=f"Nowcast from timely data, {nd:%b %Y}"))
    handles.append(Rectangle((0, 0), 1, 1, facecolor=BAND, alpha=0.3, lw=0,
                             label=f"Transition zone ({share.get(transition, 0):.0f}% of months)"))

    # ---- frame
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=6)
    ax.grid(True, color="#e9e8e3", lw=0.6, zorder=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10, labelpad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10, labelpad=10)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8.8,
              frameon=False, labelcolor=INK_2, handlelength=2.2, columnspacing=1.6)

    fig.text(0.07, 0.975, title, fontsize=15.5, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.07, 0.94, textwrap.fill(subtitle, 115), fontsize=10.5, color=INK_2, ha="left", va="top")
    if footer:
        fig.text(0.07, 0.012, footer, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.865, bottom=0.17)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
