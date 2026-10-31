"""指示を受けたときの物体の同定を VLM で行う（現場のランタイムの ObjectIdentifier）。

追跡している物体ごとに、床の位置と大きさから画像の枠を作り（CameraModel で、床と物体の上の高さの両方に投影した外接矩形）、
音声サーバーの /identify で種類と色を聞く。"other"（おもちゃではない）は名前を付けない（対象にも対象外にもならない）。
VLM が決まった値で答えられなかった物体があれば止める（黙って飛ばさない）。
"""
from __future__ import annotations

import base64
import logging
import math
import time

import cv2
import numpy as np

from ..calibration import CameraModel
from ..geometry import box_corners
from ..sim2d import ObjectSpec
from ..spaces import ObjectType
from .interfaces import FieldStop, Frame, TrackView

log = logging.getLogger(__name__)

CLASS_TYPES = {"lego": ObjectType.LEGO, "toy_car": ObjectType.TOY_CAR}


def track_box_px(cam: CameraModel, t: TrackView, top_m: float, size_px: tuple[int, int]) -> list[int]:
    """追跡の床の外形を、床と高さ top_m に投影した外接矩形（画像の中に切り詰める）。"""
    corners = np.asarray(box_corners(t.pose, t.size_m))
    pts = np.vstack([cam.to_px(corners, 0.0), cam.to_px(corners, top_m)])
    w, h = size_px
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        raise FieldStop("identify_failed", f"追跡 {t.id} の枠が画像の外")
    return [int(x0), int(y0), int(x1), int(y1)]


def _hue_deg(rgb) -> float:
    """色相（Lab の a*・b* の角度、度）。"""
    lab = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    return math.degrees(math.atan2(lab[2] - 128, lab[1] - 128))


class VlmObjectIdentifier:
    def __init__(self, server, cam: CameraModel, top_m: float, green_lime_split_hue_deg: float | None = None):
        """server: voice.client.ServerClient（identify を持つもの）。top_m: 枠を作るときの物体の上の高さ（ミニカー）。
        green_lime_split_hue_deg: VLM が green・lime と答えた物を、追跡の平均色の色相（Lab の a*・b* の角度、度）で決め直す境目
        （これ以上は lime）。None は VLM の答えのまま（Gazebo のリハーサル）。"""
        self.server, self.cam, self.top_m = server, cam, top_m
        self.green_lime_split = green_lime_split_hue_deg
        self.last: dict | None = None   # 最後の問い合わせ（管理の画面 field.admin_web に出す: 時刻、枠、応答）

    def identify(self, frame: Frame, tracks: dict[int, TrackView]) -> dict[int, ObjectSpec]:
        ids = sorted(tracks)
        size = (frame.image.shape[1], frame.image.shape[0])
        boxes = [track_box_px(self.cam, tracks[i], self.top_m, size) for i in ids]
        ok, png = cv2.imencode(".png", cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("画像を PNG にできない")
        r = self.server.identify(base64.b64encode(png.tobytes()).decode(), boxes)
        self.last = {"time": time.strftime("%H:%M:%S"), "frame_seq": frame.seq, "tracks": ids, "boxes": boxes,
                     "objects": r["objects"], "seconds": r["seconds"]}
        out: dict[int, ObjectSpec] = {}
        for i, o in zip(ids, r["objects"]):
            if "error" in o:
                raise FieldStop("identify_failed", f"VLM が追跡 {i} を見分けられない: {o['error']}")
            if o["class"] == "other":
                continue
            t = tracks[i]
            color = o["color"]
            if self.green_lime_split is not None and color in ("green", "lime"):
                # 黄緑は緑より黄色寄り（色相が小さい）。明るさは照明と距離で変わるので使わない（2026-10-26）
                hue = _hue_deg(t.mean_rgb)
                color = "lime" if hue <= self.green_lime_split else "green"
                if color != o["color"]:
                    log.info("追跡 %d: VLM は %s と答えたが、色相 %.0f°（境目 %.0f°、RGB %s）で %s にする", i, o["color"], hue,
                             self.green_lime_split, tuple(round(x) for x in t.mean_rgb), color)
            out[i] = ObjectSpec(f"obj{i}", CLASS_TYPES[o["class"]], t.pose, color)
        log.info("VLM の同定（%.1f s）: %s", r["seconds"],
                 ", ".join(f"{i}={o.get('color')} {o.get('class')}" for i, o in zip(ids, r["objects"])))
        # 追跡の平均色（照明ごとの色の境目を、VLM の答えと見比べて決めるため。2026-10-26、橙を黄、青を桃と答えた）
        log.info("同定した追跡の色（VLM の答え: RGB、色相）: %s", ", ".join(
            f"{i}={o.get('color')}:{tuple(round(x) for x in tracks[i].mean_rgb)},{_hue_deg(tracks[i].mean_rgb):.0f}°"
            for i, o in zip(ids, r["objects"])))
        return out
