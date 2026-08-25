"""生成第3章双触须主动嗅觉 PPO 网络架构图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figures" / "chapter03"
OUTPUT_STEM = OUTPUT_DIR / "figure_3_1_network_architecture"

FONT_REGULAR_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD_PATH = Path(r"C:\Windows\Fonts\msyhbd.ttc")

PAGE = "#FFFFFF"
INK = "#17212B"
MUTED = "#5C6B78"
FAINT = "#E7EDF2"
NEUTRAL = "#F5F7F9"
SENSOR = "#38A6B4"
WHISKER = "#6A8FB3"
BODY = "#9B8BB5"
TEMPORAL = "#2F7F72"
FEATURE = "#376FA6"
ACTOR = "#4D65A8"
MOVE = "#2E8BB2"
LEFT = "#7286C6"
RIGHT = "#D48645"
CRITIC = "#8A65A5"
TRAIN = "#B7792A"


def font(size: float, *, bold: bool = False) -> FontProperties:
    path = FONT_BOLD_PATH if bold else FONT_REGULAR_PATH
    return FontProperties(fname=str(path), size=size)


def add_text(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 7.0,
    color: str = INK,
    bold: bool = False,
    ha: str = "center",
    va: str = "center",
    zorder: int = 10,
) -> None:
    ax.text(
        x,
        y,
        text,
        color=color,
        fontproperties=font(size, bold=bold),
        ha=ha,
        va=va,
        linespacing=1.18,
        zorder=zorder,
    )


def add_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str = NEUTRAL,
    edge: str = FAINT,
    linewidth: float = 0.9,
    radius: float = 0.012,
    linestyle: str = "-",
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    linewidth: float = 1.2,
    linestyle: str = "-",
    mutation_scale: float = 9,
    connectionstyle: str = "arc3",
    zorder: int = 5,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def stage_label(ax, x: float, number: str, title: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, 0.874),
            0.026,
            0.041,
            boxstyle="round,pad=0.002,rounding_size=0.008",
            facecolor=INK,
            edgecolor=INK,
            linewidth=0,
            zorder=8,
        )
    )
    add_text(ax, x + 0.013, 0.894, number, size=7.4, color=PAGE, bold=True)
    add_text(ax, x + 0.034, 0.894, title, size=8.2, bold=True, ha="left")


def draw_observation_history(ax) -> None:
    x, y, width, height = 0.025, 0.355, 0.155, 0.475
    add_box(ax, x, y, width, height, face="#F8FAFB", edge="#C9D4DC")
    add_text(ax, x + width / 2, y + height - 0.043, r"$\mathcal{H}_t=(o_{t-19},\ldots,o_t)$", size=8.4, bold=True)
    add_text(ax, x + width / 2, y + height - 0.083, "20帧 × 16维/帧 ≈ 4 s", size=6.6, color=MUTED)

    row_y = [y + 0.292, y + 0.238, y + 0.184, y + 0.130]
    row_labels = ["o(t-19)", "o(t-18)", "…", "o(t)"]
    for index, (current_y, row_label) in enumerate(zip(row_y, row_labels)):
        if row_label == "…":
            add_text(ax, x + width / 2, current_y + 0.014, row_label, size=10.0, color=MUTED)
            continue
        add_box(
            ax,
            x + 0.015,
            current_y,
            width - 0.030,
            0.044,
            face=PAGE,
            edge="#D9E1E7",
            radius=0.006,
        )
        left = x + 0.060
        total = width - 0.080
        segments = [(8, SENSOR), (4, WHISKER), (4, BODY)]
        cursor = left
        for count, color in segments:
            segment_width = total * count / 16
            ax.add_patch(
                Rectangle(
                    (cursor, current_y + 0.009),
                    segment_width - 0.002,
                    0.026,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.88 if index == len(row_y) - 1 else 0.64,
                    zorder=4,
                )
            )
            cursor += segment_width
        add_text(ax, x + 0.050, current_y + 0.022, row_label, size=6.2, ha="right")

    legend_y = y + 0.071
    add_text(
        ax,
        x + width / 2,
        legend_y + 0.004,
        "气体动态 8  ·  触须几何 4\n本体与搜索阶段 4",
        size=5.8,
        color=MUTED,
    )

    add_text(
        ax,
        x + width / 2,
        y + 0.028,
        "不含真实风向或气源位置\n不含到源距离",
        size=5.8,
        color=MUTED,
    )


def draw_temporal_encoder(ax) -> None:
    x, y, width, height = 0.215, 0.355, 0.285, 0.475
    add_box(ax, x, y, width, height, face="#FBFCFC", edge="#BCCAD3", linewidth=1.0)
    add_text(ax, x + 0.016, y + height - 0.040, "可切换的历史编码器", size=8.1, bold=True, ha="left")
    add_text(ax, x + width - 0.016, y + height - 0.040, r"$F_\omega$", size=8.5, color=FEATURE, bold=True, ha="right")

    rows = [
        (y + 0.301, 0.091, "MLP 基线", "Flatten: 20×16 → 320", "无显式时序归纳偏置", "#F4F6F8", "#AAB8C2"),
        (y + 0.184, 0.100, "GRU（默认）", "GRU(64) → Linear + GELU", "末时刻隐藏状态 → 64维", "#E5F2EF", TEMPORAL),
        (y + 0.057, 0.108, "Transformer", "16→64 + [CLS]/位置编码", "2层 · 4头 · FFN 128 → 64维", "#EEF2F7", "#7892AA"),
    ]
    for row_y, row_h, title, first, second, face, edge in rows:
        add_box(ax, x + 0.018, row_y, width - 0.036, row_h, face=face, edge=edge, linewidth=1.15 if "默认" in title else 0.8, radius=0.009)
        add_text(ax, x + 0.032, row_y + row_h - 0.025, title, size=7.1, color=edge if "默认" in title else INK, bold=True, ha="left")
        add_text(ax, x + 0.032, row_y + row_h - 0.055, first, size=6.4, color=INK, ha="left")
        add_text(ax, x + 0.032, row_y + row_h - 0.079, second, size=5.9, color=MUTED, ha="left")
        add_arrow(
            ax,
            (x + width - 0.057, row_y + row_h / 2),
            (x + width - 0.025, row_y + row_h / 2),
            color=edge,
            linewidth=1.5 if "默认" in title else 0.8,
            linestyle="-" if "默认" in title else "--",
            mutation_scale=7,
        )

def draw_feature_and_heads(ax) -> None:
    feature_x, feature_y, feature_w, feature_h = 0.535, 0.505, 0.085, 0.167
    add_box(ax, feature_x, feature_y, feature_w, feature_h, face="#EAF1F7", edge=FEATURE, linewidth=1.2)
    add_text(ax, feature_x + feature_w / 2, feature_y + 0.119, "共享时序特征", size=7.0, color=FEATURE, bold=True)
    add_text(ax, feature_x + feature_w / 2, feature_y + 0.077, "f(t)", size=11.0, color=FEATURE, bold=True)
    add_text(
        ax,
        feature_x + feature_w / 2,
        feature_y + 0.033,
        "时序编码: 64维\nMLP基线: 320维",
        size=5.6,
        color=MUTED,
    )

    actor_x, actor_y, actor_w, actor_h = 0.655, 0.590, 0.178, 0.183
    critic_x, critic_y, critic_w, critic_h = 0.655, 0.375, 0.178, 0.145
    add_box(ax, actor_x, actor_y, actor_w, actor_h, face="#EEF1FA", edge=ACTOR, linewidth=1.1)
    add_text(ax, actor_x + 0.014, actor_y + actor_h - 0.031, "Actor 策略分支", size=7.4, color=ACTOR, bold=True, ha="left")
    add_text(ax, actor_x + actor_w / 2, actor_y + 0.094, "Linear 160 + Tanh", size=6.4)
    add_text(ax, actor_x + actor_w / 2, actor_y + 0.064, "Linear 160 + Tanh", size=6.4)
    add_text(ax, actor_x + actor_w / 2, actor_y + 0.031, "Linear 26 logits", size=6.4, color=ACTOR, bold=True)

    add_box(ax, critic_x, critic_y, critic_w, critic_h, face="#F3EEF7", edge=CRITIC, linewidth=1.1)
    add_text(ax, critic_x + 0.014, critic_y + critic_h - 0.030, "Critic 价值分支", size=7.4, color=CRITIC, bold=True, ha="left")
    add_text(ax, critic_x + critic_w / 2, critic_y + 0.067, "Linear 160 + Tanh", size=6.4)
    add_text(ax, critic_x + critic_w / 2, critic_y + 0.036, "Linear 160 + Tanh → 1", size=6.4, color=CRITIC, bold=True)

    add_arrow(ax, (feature_x + feature_w, feature_y + 0.112), (actor_x, actor_y + actor_h - 0.014), color=ACTOR, linewidth=1.5)
    add_arrow(ax, (feature_x + feature_w, feature_y + 0.054), (critic_x, critic_y + 0.012), color=CRITIC, linewidth=1.5)


def draw_outputs(ax) -> None:
    output_x = 0.865
    actor_rows = [
        (0.708, MOVE, "移动", "6类", "πθ,m"),
        (0.637, LEFT, "左触须", "10扇区", "πθ,L"),
        (0.566, RIGHT, "右触须", "10扇区", "πθ,R"),
    ]
    for row_y, color, label, count, symbol in actor_rows:
        add_box(ax, output_x, row_y, 0.110, 0.054, face="#FFFFFF", edge=color, linewidth=1.15, radius=0.009)
        add_text(ax, output_x + 0.012, row_y + 0.039, label, size=6.5, color=color, bold=True, ha="left")
        add_text(ax, output_x + 0.012, row_y + 0.011, count, size=5.5, color=MUTED, ha="left")
        add_text(ax, output_x + 0.096, row_y + 0.027, symbol, size=7.2, color=color, ha="right")
        add_arrow(ax, (0.833, row_y + 0.027), (output_x, row_y + 0.027), color=color, linewidth=1.1, mutation_scale=7)

    add_text(
        ax,
        output_x + 0.055,
        0.530,
        "πθ(a|H) = πm · πL · πR",
        size=6.8,
        color=ACTOR,
    )
    add_text(ax, output_x + 0.055, 0.493, "a = (am, aL, aR)", size=6.8, color=INK)

    add_box(ax, output_x, 0.397, 0.110, 0.079, face="#FFFFFF", edge=CRITIC, linewidth=1.15, radius=0.009)
    add_text(ax, output_x + 0.055, 0.447, "vφ[H(t)]", size=8.4, color=CRITIC, bold=True)
    add_text(ax, output_x + 0.055, 0.418, "状态价值标量", size=5.8, color=MUTED)
    add_arrow(ax, (0.833, 0.447), (output_x, 0.447), color=CRITIC, linewidth=1.1, mutation_scale=7)


def draw_training_path(ax) -> None:
    x, y, width, height = 0.215, 0.075, 0.760, 0.195
    add_box(ax, x, y, width, height, face="#FFF9F0", edge="#D8B784", linewidth=0.9, linestyle="--")
    add_text(ax, x + 0.018, y + height - 0.029, "仅训练期：on-policy PPO 更新", size=7.5, color=TRAIN, bold=True, ha="left")

    blocks = [
        (x + 0.020, 0.220, "Rollout buffer", "H(t), a(t), r(t), ι(t), log πold, vold"),
        (x + 0.265, 0.120, "GAE", "A-hat(t), R-hat(t)"),
        (x + 0.410, 0.325, "联合目标", "-Lclip + cv Lv - ce Ent"),
    ]
    for block_x, block_w, title, detail in blocks:
        add_box(ax, block_x, y + 0.053, block_w, 0.080, face=PAGE, edge="#D9C5A5", radius=0.008)
        add_text(ax, block_x + block_w / 2, y + 0.105, title, size=6.5, color=TRAIN, bold=True)
        add_text(ax, block_x + block_w / 2, y + 0.074, detail, size=6.5, color=INK)

    add_arrow(ax, (x + 0.240, y + 0.093), (x + 0.265, y + 0.093), color=TRAIN, linewidth=1.0, mutation_scale=7)
    add_arrow(ax, (x + 0.385, y + 0.093), (x + 0.410, y + 0.093), color=TRAIN, linewidth=1.0, mutation_scale=7)
    add_text(ax, x + width - 0.022, y + 0.029, "虚线箭头表示梯度更新，不属于部署时前向推理", size=5.8, color=MUTED, ha="right")

    add_arrow(
        ax,
        (x + 0.640, y + 0.133),
        (0.805, 0.375),
        color=TRAIN,
        linewidth=1.1,
        linestyle="--",
        mutation_scale=8,
        connectionstyle="arc3,rad=-0.15",
        zorder=3,
    )
    add_arrow(
        ax,
        (x + 0.592, y + 0.133),
        (0.810, 0.590),
        color=TRAIN,
        linewidth=1.1,
        linestyle="--",
        mutation_scale=8,
        connectionstyle="arc3,rad=0.18",
        zorder=3,
    )
    add_arrow(
        ax,
        (x + 0.520, y + 0.133),
        (0.578, 0.505),
        color=TRAIN,
        linewidth=1.0,
        linestyle="--",
        mutation_scale=8,
        connectionstyle="arc3,rad=0.10",
        zorder=3,
    )


def build_figure() -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )
    width_in = 180 / 25.4
    height_in = 105 / 25.4
    fig = plt.figure(figsize=(width_in, height_in), facecolor=PAGE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stage_label(ax, 0.025, "1", "历史观测")
    stage_label(ax, 0.215, "2", "时序信息提取")
    stage_label(ax, 0.535, "3", "共享表征")
    stage_label(ax, 0.655, "4", "PPO Actor–Critic")
    stage_label(ax, 0.852, "5", "策略/价值输出")

    draw_observation_history(ax)
    draw_temporal_encoder(ax)
    draw_feature_and_heads(ax)
    draw_outputs(ax)
    draw_training_path(ax)

    add_arrow(ax, (0.180, 0.590), (0.215, 0.590), color=TEMPORAL, linewidth=1.6)
    add_arrow(ax, (0.500, 0.590), (0.535, 0.590), color=FEATURE, linewidth=1.6)

    return fig


def main() -> None:
    for font_path in (FONT_REGULAR_PATH, FONT_BOLD_PATH):
        if not font_path.exists():
            raise FileNotFoundError(f"缺少中文字体: {font_path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    pdf_metadata = {
        "Title": "双触须主动嗅觉 PPO 时序 Actor-Critic 网络架构",
        "Subject": "第3章算法网络架构图",
        "Creator": "Matplotlib",
    }
    svg_metadata = {
        "Title": "双触须主动嗅觉 PPO 时序 Actor-Critic 网络架构",
        "Description": "20帧硬件可重建观测经时序编码后进入PPO策略与价值分支。",
    }
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), facecolor=PAGE, metadata=svg_metadata)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), facecolor=PAGE, metadata=pdf_metadata)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor=PAGE)
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tiff"),
        dpi=600,
        facecolor=PAGE,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(OUTPUT_STEM)


if __name__ == "__main__":
    main()
