"""Side-view schematic of visual-hull carving: 1 view = a cone, 3 views = intersection."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, Polygon, Rectangle

FP = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FB = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
fm.fontManager.addfont(FP)
fm.fontManager.addfont(FB)
plt.rcParams["font.family"] = "Noto Sans CJK JP"

NAVY, BODY, TEAL, ORANGE, MAG, GREY = "#235876", "#44546A", "#34A9B3", "#E49140", "#D63AC8", "#898989"
O, R = np.array([0.0, 0.06]), 0.06  # object (ball) centre, radius, metres
CAMS = [("正上方", np.array([0.0, 0.34]), "#235876"),
        ("斜視 1", np.array([-0.27, 0.22]), "#E49140"),
        ("斜視 2", np.array([0.27, 0.22]), "#34A9B3")]


def cone(C):
    d = O - C
    th = np.arctan2(d[1], d[0])
    a = np.arcsin(R / np.linalg.norm(d))
    return th, a


def inside(C, X, Z):
    th, a = cone(C)
    ang = np.arctan2(Z - C[1], X - C[0]) - th
    ang = (ang + np.pi) % (2 * np.pi) - np.pi
    return (np.abs(ang) <= a) & (Z >= 0)


def ray_end(C, ang):
    v = np.array([np.cos(ang), np.sin(ang)])
    t = (0 - C[1]) / v[1]  # hit the table
    return C + t * v


def draw_cam(ax, name, C, col, show_cone=True, alpha=0.13):
    th, a = cone(C)
    e1, e2 = ray_end(C, th - a), ray_end(C, th + a)
    if show_cone:
        ax.add_patch(Polygon([C, e1, e2], closed=True, fc=col, ec="none", alpha=alpha, zorder=1))
        for e in (e1, e2):
            ax.plot([C[0], e[0]], [C[1], e[1]], color=col, lw=1.4, ls="--", zorder=2)
    # camera body: small triangle pointing at the object + image plane with mask (magenta)
    u = np.array([np.cos(th), np.sin(th)])
    n = np.array([-u[1], u[0]])
    f, w = 0.04, 0.035
    p0 = C + f * u
    ax.add_patch(Polygon([C, p0 + w * n, p0 - w * n], closed=True, fc="white", ec=col, lw=2, zorder=5))
    m = f * np.tan(a)
    ax.plot(*np.c_[p0 + w * n, p0 - w * n], color=col, lw=3, zorder=6, solid_capstyle="butt")
    ax.plot(*np.c_[p0 + m * n, p0 - m * n], color=MAG, lw=5, zorder=7, solid_capstyle="butt")
    off = np.array([0, 0.03]) if C[0] == 0 else np.array([np.sign(C[0]) * 0.035, 0.03])
    ax.text(*(C + off), name, ha="center", va="bottom", fontsize=15, color=col, fontweight="bold", zorder=8)


def base(ax, title):
    ax.set_xlim(-0.36, 0.36)
    ax.set_ylim(-0.03, 0.44)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-0.36, -0.03), 0.72, 0.03, fc="#E3C9A4", ec="none", zorder=0))
    ax.text(0.345, -0.017, "桌面", ha="right", va="center", fontsize=12, color="#8A6A3E")
    ax.set_title(title, fontsize=19, color=NAVY, fontweight="bold", loc="left", pad=8)


def obj(ax, ls="-", fill=True):
    ax.add_patch(Circle(O, R, fc="#F2EFEA" if fill else "none", ec="#B7473C", lw=2, ls=ls, zorder=9))


fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"wspace": 0.04})
fig.patch.set_facecolor("white")

# --- left: one view only
base(a1, "① 單一視角：只知道物件在這個視錐裡")
name, C, col = CAMS[1]
draw_cam(a1, name, C, col, alpha=0.2)
obj(a1)
a1.annotate("遮罩（SAM3）\n往空間反投影", xy=(-0.235, 0.215), xytext=(-0.33, 0.36), fontsize=13, color=MAG,
            arrowprops=dict(arrowstyle="->", color=MAG, lw=1.5))
a1.annotate("視錐內任何深度都有可能\n→ 無法決定距離與大小", xy=(0.14, 0.03), xytext=(0.06, 0.25), fontsize=13, color=BODY,
            arrowprops=dict(arrowstyle="->", color=GREY, lw=1.5))

# --- right: three views, intersection
base(a2, "② 三個視角：視錐交集 = visual hull")
for name, C, col in CAMS:
    draw_cam(a2, name, C, col, alpha=0.08)
X, Z = np.meshgrid(np.linspace(-0.36, 0.36, 1400), np.linspace(-0.01, 0.44, 900))
hull = np.ones_like(X, bool)
for _, C, _ in CAMS:
    hull &= inside(C, X, Z)
a2.contourf(X, Z, hull.astype(float), levels=[0.5, 1.5], colors=[TEAL], alpha=0.55, zorder=3)
a2.contour(X, Z, hull.astype(float), levels=[0.5], colors=[NAVY], linewidths=2, zorder=4)
obj(a2, ls="-", fill=False)
xs, zs = X[hull], Z[hull]
lo, hi = np.array([xs.min(), zs.min()]), np.array([xs.max(), zs.max()])
a2.add_patch(Rectangle(lo, *(hi - lo), fc="none", ec="#C0504D", lw=1.6, ls=":", zorder=10))
a2.annotate("visual hull\n三個視錐都包含的空間", xy=(0.035, 0.035), xytext=(0.17, 0.035), fontsize=13, color=NAVY,
            fontweight="bold", arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5))
a2.annotate("外框 → 中心、佔地、高度", xy=(hi[0], 0.10), xytext=(0.17, 0.12), fontsize=13, color="#C0504D",
            arrowprops=dict(arrowstyle="->", color="#C0504D", lw=1.5))
a2.annotate("角落多出的空間\n→ 高度略為高估", xy=(-0.045, hi[1] - 0.006), xytext=(-0.355, 0.05), fontsize=13, color=BODY,
            arrowprops=dict(arrowstyle="->", color=GREY, lw=1.5))

plt.savefig(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "docs/figures/visual_hull_diagram.png"), dpi=200, bbox_inches="tight", facecolor="white")
print("hull top", hi[1], "true top", 2 * R)
