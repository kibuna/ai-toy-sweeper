"""印刷用の ArUco マーカーと ChArUco ボード（カメラの校正用）の PDF を作る（config/markers.yaml、doc/camera_setup.md）。

    uv run python scripts/make_markers_pdf.py --out data/generated/markers

出力: markers_a4.pdf（1 ページ目: 床の 4 隅 ID 0〜3、2 ページ目: ロボット ID 4）、charuco_a4.pdf（内部パラメータ用のボード）。
A4、300 dpi。印刷は「実際のサイズ（100%）」で（「用紙に合わせる」にしない）。印刷後、黒い四角の 1 辺を定規で測る。
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from toy_sweeper.config import CONFIG_DIR, load_yaml

DPI = 300
A4_MM = (210, 297)
FONT = "/mnt/c/Windows/Fonts/BIZ-UDGothicR.ttc"   # ラベルの日本語（無ければ英字だけにする）


def px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def page() -> Image.Image:
    return Image.new("L", (px(A4_MM[0]), px(A4_MM[1])), 255)


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def aruco_dict(name: str):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def marker_pages(c: dict) -> list[Image.Image]:
    """1 ページ目: 床の 4 隅（2 × 2）、2 ページ目: ロボットの上。マーカーの周りの線が切り取り線（paper_margin_m）。"""
    f = c["field"]
    d = aruco_dict(f["dictionary"])
    names = {"near_left": "手前・左（壁の外側）", "near_right": "手前・右（壁の外側）", "far_right": "奥・右（内側の角）",
             "far_left": "奥・左（内側の角）", "robot": "ロボットの上"}
    corners = [(i, f["corner_marker_size_m"], label) for label, i in f["corner_ids"].items()]
    pages = []
    for items in (corners, [(f["robot_id"], f["robot_marker_size_m"], "robot")]):
        img = page()
        draw = ImageDraw.Draw(img)
        x0, y0 = px(15), px(15)
        for k, (mid, size_m, label) in enumerate(items):
            side = px(size_m * 1000)
            m = cv2.aruco.generateImageMarker(d, mid, side)
            col, row = k % 2, k // 2
            x, y = x0 + col * px(100), y0 + row * px(130)
            q = px(f["paper_margin_m"] * 1000)
            img.paste(Image.fromarray(m), (x + q, y + q))
            draw.rectangle((x, y, x + side + 2 * q, y + side + 2 * q), outline=150, width=2)   # 切り取り線
            draw.text((x, y + side + 2 * q + px(4)), f"ID {mid}  {names[label]}", fill=0, font=font(px(3.5)))
            draw.text((x, y + side + 2 * q + px(10)), f"黒い四角 {size_m * 1000:.0f} mm。線で切る", fill=0, font=font(px(3.2)))
        draw.text((x0, px(285)), f"{f['dictionary']}。実際のサイズ（100%）で印刷し、黒い四角の 1 辺を測る", fill=0, font=font(px(3.2)))
        pages.append(img)
    return pages


def charuco_page(c: dict) -> Image.Image:
    b = c["intrinsics"]
    cols, rows = b["squares"]
    board = cv2.aruco.CharucoBoard((cols, rows), b["square_size_m"], b["marker_size_m"], aruco_dict(b["dictionary"]))
    w, h = px(cols * b["square_size_m"] * 1000), px(rows * b["square_size_m"] * 1000)
    bimg = board.generateImage((w, h), marginSize=0, borderBits=1)
    img = page()
    x, y = (img.width - w) // 2, px(30)
    img.paste(Image.fromarray(bimg), (x, y))
    ImageDraw.Draw(img).text((x, y + h + px(8)),
                             f"カメラの校正用 ChArUco {cols}×{rows}、マス {b['square_size_m'] * 1000:.0f} mm（{b['dictionary']}）。"
                             "平らな板に貼る", fill=0, font=font(px(3.2)))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    c = load_yaml("markers.yaml", CONFIG_DIR)
    a.out.mkdir(parents=True, exist_ok=True)
    for name, pages in (("markers_a4.pdf", marker_pages(c)), ("charuco_a4.pdf", [charuco_page(c)])):
        pages[0].save(a.out / name, resolution=DPI, save_all=True, append_images=pages[1:])
        print("saved", a.out / name, f"{len(pages)} ページ")


if __name__ == "__main__":
    main()
