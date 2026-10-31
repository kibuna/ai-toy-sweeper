"""最終課題のレポート（doc/report.md）の図を作る（2026-10-29）。

    uv run python scripts/make_report_figures.py [図 4 の元の画像のフォルダ]     # → doc/figures/*.png

図は doc/figures/ に生成済み。図 4 の元の画像（2D・Gazebo・実機の動画のこま）は同梱しないので、このリポジトリだけでは図 4 以外を作り直せる。

成功率などの数字は図にせず、doc/report.md の表に書く（2026-10-29。数字は表のほうが読みやすいため）。色は MacBook = 青、3090 の PC = オレンジ、
ロボットとカメラ = 水色（どの図でも同じ物に同じ色）。文字は色を付けず黒・灰で書く。
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "doc" / "figures"

plt.rcParams["font.family"] = "Hiragino Sans"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # MacBook / 3090 / ロボットとカメラ
TINT = {BLUE: "#e8f1fc", ORANGE: "#fdeee7", AQUA: "#e5f6ef", INK2: "#f2f1ee"}
DPI = 200


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), color=INK2, title_size=11, size=9):
    """色の枠と見出しの帯の箱。文字は黒。(x, y) は左下。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=TINT[color], ec=color, lw=1.6))
    ax.text(x + 0.12, y + h - 0.12, title, ha="left", va="top", fontsize=title_size, color=INK, weight="bold")
    for i, s in enumerate(lines):
        ax.text(x + 0.12, y + h - 0.45 - i * 0.27, s, ha="left", va="top", fontsize=size, color=INK)


def arrow(ax, p, q, label=None, off=(0, 0.12), rad=0.0, ha="center", size=8.5, color=INK2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.3, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        mx, my = (p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1]
        ax.text(mx, my, label, ha=ha, va="bottom", fontsize=size, color=INK2,
                bbox=dict(fc=SURFACE, ec="none", pad=0.6))


def legend_devices(ax, x, y):
    for i, (c, s) in enumerate([(BLUE, "MacBook（現場）"), (ORANGE, "RTX 3090 の PC"), (AQUA, "ロボット・カメラ")]):
        ax.add_patch(FancyBboxPatch((x + i * 2.3, y), 0.28, 0.2, boxstyle="round,pad=0.01", fc=TINT[c], ec=c, lw=1.4))
        ax.text(x + i * 2.3 + 0.38, y + 0.1, s, va="center", fontsize=9, color=INK)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name)


def fig_system():
    fig, ax = canvas(12, 6.6)
    ax.text(0.1, 6.45, "図 2  システム構成（機器とデータの流れ）", fontsize=13, weight="bold", color=INK, va="top")
    legend_devices(ax, 6.0, 6.25)
    box(ax, 0.1, 4.2, 2.4, 1.6, "子ども", ["声で指示する", "「みどりのレゴを", "  ひだりしたに」"], INK2)
    box(ax, 3.0, 3.2, 4.1, 2.6, "MacBook", ["ブラウザの画面: 録音・読み上げ・停止", "知覚: ArUco で姿勢、背景との差で物体、追跡",
                                         "方策（PPO）: 8 Hz で行動を選ぶ", "安全監視: 古い観測・通信切れで止める",
                                         "記録: 動画（印あり・なし）・録音・やりとり"], BLUE)
    box(ax, 8.6, 3.6, 3.3, 2.2, "RTX 3090 の PC", ["音声サーバー", "・Whisper large-v3: 声 → 文", "・Qwen3-VL-4B: 文 → 指示",
                                                "  画像 → 物の色・種類", "Gazebo（使うときだけ起動）"], ORANGE)
    box(ax, 3.2, 0.3, 2.6, 1.7, "USB カメラ（C920）", ["床を斜め上から", "1920×1080、10 fps"], AQUA)
    box(ax, 6.6, 0.3, 3.0, 1.7, "ロボット", ["Cutebot V3 + サーボの爪", "micro:bit V2", "指令の期限 0.5 s で自動停止"], AQUA)
    arrow(ax, (2.5, 5.0), (3.0, 5.0), "声", off=(0, 0.06))
    arrow(ax, (7.1, 5.45), (8.6, 5.45), "録音", off=(0, 0.04))
    arrow(ax, (8.6, 4.95), (7.1, 4.95), "指示", off=(0, 0.04))
    arrow(ax, (7.1, 4.35), (8.6, 4.35), "枠つきの画像", off=(0, 0.04))
    arrow(ax, (8.6, 3.85), (7.1, 3.85), "色・種類", off=(0, 0.04))
    arrow(ax, (4.5, 2.0), (4.5, 3.2), "画像", off=(-0.35, 0), ha="center")
    arrow(ax, (6.8, 3.2), (7.6, 2.0), "BLE で指令", off=(0.75, 0))
    arrow(ax, (6.6, 1.15), (5.8, 1.15), "写る", off=(0, 0.05))
    ax.text(0.1, 0.05, "同じ画面で、実機・2D シミュレーション・Gazebo を切り替えて、同じ声の指示で動かせる。", fontsize=9, color=INK2)
    save(fig, "fig2_system.png")


def fig_loop():
    fig, ax = canvas(12, 4.2)
    ax.text(0.1, 4.1, "図 3  1 周期の制御（0.125 s ごと、8 Hz）", fontsize=13, weight="bold", color=INK, va="top")
    steps = [("カメラの画像", ["10 fps"], AQUA), ("知覚", ["ArUco: 姿勢", "背景との差: 物", "追跡: 物に番号"], BLUE),
             ("観測を作る", ["ロボットから見た", "対象・ゴール・", "ほかの物の位置", "＋遅れの先読み"], BLUE),
             ("方策（PPO）", ["7 つの行動から 1 つ", "前進・後退・左右", "爪の開閉・止まる"], BLUE),
             ("見張り", ["つかみ損ねたら", "爪を開く", "見失ったら探し直す"], BLUE),
             ("指令", ["BLE で送る", "期限 0.5 s"], AQUA)]
    x, w, gap = 0.1, 1.75, 0.27
    for i, (t, ls, c) in enumerate(steps):
        box(ax, x + i * (w + gap), 1.3, w, 2.0, t, ls, c, title_size=10.5, size=8.5)
        if i:
            arrow(ax, (x + i * (w + gap) - gap, 2.3), (x + i * (w + gap), 2.3))
    ax.text(0.1, 0.95, "先読み: 画像は撮ってから約 0.2 s 古い（学習では 0 s）。撮った時刻から今までに送った指令で、ロボットの姿勢を"
            "「指令が効く時刻」まで進めてから方策に渡す。", fontsize=9, color=INK2, va="top")
    ax.text(0.1, 0.55, "見張り: 方策が学習で見ていない場面（閉じた爪の外に対象が離れて残る等）では、方策より先に決まった動きをして"
            "学習で見た場面に戻す。", fontsize=9, color=INK2, va="top")
    save(fig, "fig3_control_loop.png")


def lane(ax, x0, x1, y_top, y_lane, label):
    """箱の下の通り道を通る戻りの矢印（x0 の箱の下 → 下へ → 左へ → x1 の箱の下へ上がる）。"""
    ax.plot([x0, x0, x1], [y_top, y_lane, y_lane], color=INK2, lw=1.3)
    arrow(ax, (x1, y_lane), (x1, y_top))
    ax.text((x0 + x1) / 2, y_lane + 0.05, label, ha="center", va="bottom", fontsize=8.5, color=INK2,
            bbox=dict(fc=SURFACE, ec="none", pad=0.6))


def fig_pipeline():
    fig, ax = canvas(14, 5.2)
    ax.text(0.1, 5.1, "図 5  開発の流れ（シミュレーションで学習し、段階を踏んで実機へ）", fontsize=13, weight="bold", color=INK, va="top")
    legend_devices(ax, 0.1, 4.45)
    box(ax, 0.1, 2.0, 2.5, 2.2, "2D シミュレータ", ["平面の物理（自作）", "Python 版 → Rust 版", "1,024 並列", "盤面を毎回乱数で作る"], ORANGE)
    box(ax, 3.0, 2.0, 2.4, 2.2, "強化学習（PPO）", ["GPU で学習", "1 本 約 2.5 万 steps/s", "Stage を段階的に", "合計 6.1 億ステップ"], ORANGE)
    box(ax, 5.8, 2.0, 2.5, 2.2, "Gazebo リハーサル", ["3 次元の物理", "カメラの画像", "実機と同じ知覚のコード", "失敗の原因を切り分け"], ORANGE)
    box(ax, 8.7, 2.0, 2.6, 2.2, "実機試験", ["カメラの画像で知覚", "先読み・方策・見張り", "速さを走りながら測る", "動画と走りの記録"], BLUE)
    box(ax, 11.9, 2.0, 2.0, 2.2, "実機", ["ロボット", "カメラ", "床・照明は", "家ごとに違う"], AQUA)
    arrow(ax, (2.6, 3.1), (3.0, 3.1))
    arrow(ax, (5.4, 3.1), (5.8, 3.1), "方策")
    arrow(ax, (8.3, 3.1), (8.7, 3.1), "方策")
    arrow(ax, (11.3, 3.4), (11.9, 3.4), "指令", size=8)
    arrow(ax, (11.9, 2.75), (11.3, 2.75), "画像", off=(0, -0.38), size=8)
    lane(ax, 7.05, 1.8, 2.0, 1.45, "Gazebo で見つかった不完全さ（挟み損ね・隠れ）を学習に入れる")
    lane(ax, 10.0, 0.9, 2.0, 0.85, "実機で測った値（速さ・寸法・遅れ・惰性）と失敗の条件を入れて、学習し直す")
    ax.text(0.1, 0.2, "2 台の分担: 3090 の PC が学習と Gazebo、MacBook が実機試験。同じ git のリポジトリと実験記録でやりとりした。",
            fontsize=9, color=INK2)
    save(fig, "fig5_pipeline.png")


def fig_curriculum():
    fig, ax = canvas(12, 4.4)
    ax.text(0.1, 4.3, "図 6  段階的な学習（Stage）。やさしい盤面から始め、成功率が上がると次へ進む", fontsize=13, weight="bold",
            color=INK, va="top")
    st = [("0〜3", "ブロック 1 個を", "押す・挟んで運ぶ"), ("4", "ミニカー", "（押して運ぶ）"), ("5", "壁際", "挟まないと運べない"),
          ("6", "散らかった盤面", "対象外もある"), ("7", "対象が複数", "同じ色を全部"), ("8", "置き場あり", "最大 27 個"),
          ("9", "やり残しから", "後片付け")]
    w, gap = 1.55, 0.16
    for i, (n, a, b) in enumerate(st):
        x, y = 0.1 + i * (w + gap), 0.6 + i * 0.32
        ax.add_patch(FancyBboxPatch((x, 0.6), w, y - 0.6 + 1.2, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=TINT[ORANGE], ec=ORANGE, lw=1.4))
        ax.text(x + w / 2, y + 1.55, f"Stage {n}", ha="center", fontsize=10.5, weight="bold", color=INK)
        ax.text(x + w / 2, y + 1.0, a, ha="center", fontsize=9, color=INK)
        ax.text(x + w / 2, y + 0.72, b, ha="center", fontsize=8.5, color=INK2)
    ax.text(0.1, 0.2, "上の Stage に進んでも、前の Stage の盤面を 3 割混ぜて、できるようになったことを忘れないようにした。", fontsize=9,
            color=INK2)
    save(fig, "fig6_curriculum.png")


def fig_worlds(frames: Path):
    """図 4: 同じ片付けを、2D シミュレーション・Gazebo・実機で並べる。元の画像（frames/2d.jpg・gazebo.jpg・real.jpg）は
    3090 の動画（data/evaluations/video_v32_2d・video_v32_s7）と実機の録画（data/camera/、部屋が写るので git に入れない）から
    切り出したもの。実機の画像を含むので、この図も git に入れない（.gitignore）。"""
    import matplotlib.image as mpimg
    panels = [("2d.jpg", "(a) 2D シミュレーション", "学習に使う。上から見た平面の物理。\n毎回ちがう散らかり方の盤面を乱数で作る", ORANGE),
              ("gazebo.jpg", "(b) ROS 2 ＋ Gazebo", "確かめに使う。3 次元の物理と仮想カメラ。\n実機と同じ画像処理のプログラムで動かす", ORANGE),
              ("real.jpg", "(c) 実機", "床を斜め上から web カメラで撮る。\n赤い丸が運ぶ物、緑の枠がゴール", AQUA)]
    imgs = [mpimg.imread(frames / f) for f, *_ in panels]
    ratios = [im.shape[1] / im.shape[0] for im in imgs]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": ratios})
    fig.patch.set_facecolor(SURFACE)
    for ax, im, (_, title, cap, c) in zip(axes, imgs, panels):
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(c); sp.set_linewidth(2)
        ax.set_title(title, fontsize=11, color=INK, loc="left", weight="bold")
        ax.text(0, -0.04, cap, transform=ax.transAxes, va="top", fontsize=9, color=INK2)
    fig.suptitle("図 4  同じ「片付け」を 3 つの世界で（同じ方策を動かす）", x=0.01, y=0.99, ha="left", fontsize=13, weight="bold",
                 color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.14, wspace=0.04)
    save(fig, "fig4_worlds.png")


def fig_photos():
    """図 1: 運ぶ物と、ロボット（写真は doc/figures/photo_objects.jpg・photo_robot.jpg。机とカッターマットだけで部屋は写らない。
    撮影の情報は落としてある）。マットの目盛りは 1 cm。"""
    import matplotlib.image as mpimg
    a, b = mpimg.imread(OUT / "photo_objects.jpg"), mpimg.imread(OUT / "photo_robot.jpg")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [a.shape[1] / a.shape[0], b.shape[1] / b.shape[0]]})
    fig.patch.set_facecolor(SURFACE)
    for ax, im, title in ((axes[0], a, "(a) 運ぶ物（1 マス 1 cm）"), (axes[1], b, "(b) ロボット（左が前。目印の台紙は外した状態）")):
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(AQUA); sp.set_linewidth(2)
        ax.set_title(title, fontsize=11, color=INK, loc="left", weight="bold")

    def note(ax, xy, text, xytext):
        ax.annotate(text, xy=xy, xytext=xytext, fontsize=9.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=INK2, lw=0.8),
                    arrowprops=dict(arrowstyle="-", color=INK, lw=1.2))
    h, w = a.shape[:2]
    note(axes[0], (0.46 * w, 0.22 * h), "ブロック（2×4）\n約 3.2 × 1.6 cm、8 個（6 色）", (0.62 * w, 0.10 * h))
    note(axes[0], (0.67 * w, 0.62 * h), "ミニカー\n約 8 cm", (0.72 * w, 0.36 * h))
    h, w = b.shape[:2]
    note(axes[1], (0.12 * w, 0.68 * h), "爪", (0.03 * w, 0.88 * h))
    note(axes[1], (0.15 * w, 0.38 * h), "サーボ（爪を開け閉め）", (0.02 * w, 0.06 * h))
    note(axes[1], (0.53 * w, 0.30 * h), "micro:bit（Bluetooth）", (0.33 * w, 0.06 * h))
    note(axes[1], (0.63 * w, 0.42 * h), "電池（単 4 × 3）", (0.70 * w, 0.20 * h))
    note(axes[1], (0.63 * w, 0.73 * h), "車輪（二輪）", (0.75 * w, 0.90 * h))
    fig.suptitle("図 1  運ぶ物とロボット（Cutebot V3 ＋ サーボの爪）", x=0.01, y=0.99, ha="left", fontsize=13, weight="bold", color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.03)
    save(fig, "fig1_photos.png")


def main():
    fig_system()
    fig_loop()
    fig_pipeline()
    fig_curriculum()
    if (OUT / "photo_objects.jpg").exists():
        fig_photos()
    import sys
    if len(sys.argv) > 1:          # 図 4 の元の画像のフォルダ（2d.jpg・gazebo.jpg・real.jpg）
        fig_worlds(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
