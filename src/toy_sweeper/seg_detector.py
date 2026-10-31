"""学習する物体の検出器（YOLO のインスタンスセグメンテーションを ONNX で）。

背景との差の検出（perception_oblique.BackgroundDetector）の「画像 → 物体ごとの画素の塊」だけを置き換える。塊から床の
中心・向き・大きさを出すところ、ロボットの範囲を除くところ、追跡は同じものを使う。推論は onnxruntime（CPU）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .perception_oblique import BackgroundDetector

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "seg_detector.yaml"
PAD_VALUE = 114     # 詰め物の灰色（ultralytics の letterbox と同じ。学習と推論で揃える）
CLASS_NAMES = ("lego", "toy_car")


@dataclass(frozen=True)
class SegDetectorCfg:
    model: str              # ONNX のパス（リポジトリの根から）
    conf: float             # 物体とみなすクラスの確からしさの下限
    iou: float              # 重なりの除去（NMS）の IoU の上限
    mask_threshold: float   # マスクの確からしさ（sigmoid）の閾値


def load_seg_detector_cfg(path: Path = CONFIG_PATH) -> SegDetectorCfg:
    return SegDetectorCfg(**yaml.safe_load(path.read_text()))


def letterbox(rgb: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """入力の大きさ size（高さ, 幅）に縦横比を保って縮め、真ん中に置いて周りを灰色で詰める（ultralytics と同じ）。
    → (1, 3, H, W) の float32（0〜1）、縮めた比 r、詰め物の (左, 上)。"""
    h, w = rgb.shape[:2]
    H, W = size
    r = min(H / h, W / w)
    nw, nh = round(w * r), round(h * r)
    left, top = round((W - nw) / 2 - 0.1), round((H - nh) / 2 - 0.1)
    out = np.full((H, W, 3), PAD_VALUE, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out.transpose(2, 0, 1)[None].astype(np.float32) / 255.0, r, (left, top)


def postprocess(out0: np.ndarray, protos: np.ndarray, r: float, pad: tuple[int, int], orig_hw: tuple[int, int], *,
                conf: float, iou: float, mask_threshold: float) -> list[tuple[int, float, np.ndarray, int, int]]:
    """ONNX の出力（out0 (1, 4 + nc + nm, N)、protos (1, nm, H/4, W/4)）→ 物体ごとの
    (クラス, 確からしさ, 元の画像の箱の範囲のマスク（uint8）, 箱の左上 x, y)。重なりはクラスを問わず除く。"""
    p = out0[0].T
    nm = protos.shape[1]
    nc = p.shape[1] - 4 - nm
    scores_all = p[:, 4:4 + nc]
    cls = scores_all.argmax(1)
    score = scores_all[np.arange(len(p)), cls]
    keep = np.nonzero(score >= conf)[0]
    if len(keep) == 0:
        return []
    xywh = p[keep, :4]
    boxes = np.column_stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 2], xywh[:, 3]])
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), score[keep].astype(float).tolist(), conf, iou)
    idx = keep[np.asarray(idx, int).ravel()]
    left, top = pad
    H, W = orig_hw
    pr = protos[0].reshape(nm, -1)
    ph, pw = protos.shape[2:]
    out = []
    for i in idx:
        cx, cy, bw, bh = p[i, :4]
        # 箱を元の画像の画素へ
        x1, x2 = ((cx - bw / 2 - left) / r, (cx + bw / 2 - left) / r)
        y1, y2 = ((cy - bh / 2 - top) / r, (cy + bh / 2 - top) / r)
        X0, Y0 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        X1, Y1 = min(W, int(np.ceil(x2))), min(H, int(np.ceil(y2)))
        if X1 <= X0 or Y1 <= Y0:
            continue
        m = (p[i, 4 + nc:] @ pr).reshape(ph, pw)
        # 元の画素 (X, Y) の中心 → プロトタイプの座標: u = ((X + 0.5) r + left) / 4 − 0.5
        M = np.array([[r / 4, 0, ((X0 + 0.5) * r + left) / 4 - 0.5],
                      [0, r / 4, ((Y0 + 0.5) * r + top) / 4 - 0.5]], np.float32)
        logit = cv2.warpAffine(m.astype(np.float32), M, (X1 - X0, Y1 - Y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)
        # sigmoid(logit) > mask_threshold ⇔ logit > log(t / (1 − t))
        mask = (logit > np.log(mask_threshold / (1 - mask_threshold))).astype(np.uint8)
        out.append((int(cls[i]), float(score[i]), mask, X0, Y0))
    return out


class SegModel:
    def __init__(self, path: Path, cfg: SegDetectorCfg):
        import onnxruntime as ort
        if not path.exists():
            raise FileNotFoundError(f"検出器のモデルが無い: {path}")
        self.cfg = cfg
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        shape = self.session.get_inputs()[0].shape
        if not all(isinstance(v, int) for v in shape[2:]):
            raise ValueError(f"入力の大きさが決まっていない ONNX（dynamic）: {shape}")
        self.size = (shape[2], shape[3])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, rgb: np.ndarray) -> list[tuple[int, float, np.ndarray, int, int]]:
        x, r, pad = letterbox(rgb, self.size)
        out0, protos = self.session.run(None, {self.input_name: x})
        c = self.cfg
        return postprocess(out0, protos, r, pad, rgb.shape[:2], conf=c.conf, iou=c.iou, mask_threshold=c.mask_threshold)


class SegDetector(BackgroundDetector):
    """BackgroundDetector の塊を作る部分だけを学習したモデルにしたもの（背景の画像は使わないが、同じ形で作る）。
    last_instances: 直前の画像のモデルの出力（クラス, 確からしさ, …）。"""

    def __init__(self, model: SegModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.last_instances: list = []

    @classmethod
    def like(cls, det: BackgroundDetector, model) -> "SegDetector":
        """校正済みの背景との差の検出（カメラ・設定・探す範囲・ロボットの範囲）と同じで、塊だけをモデルにしたもの
        （scripts/eval_rehearsal.py --detector learned）。"""
        new = cls.__new__(cls)
        new.__dict__.update(det.__dict__)
        new.model = model
        new.last_instances = []
        return new

    def blobs(self, rgb: np.ndarray, robot_mask_u8: np.ndarray | None) -> list[tuple[np.ndarray, int, int]]:
        self.last_instances = self.model(rgb)
        out = []
        for _, _, mask, x0, y0 in self.last_instances:
            h, w = mask.shape
            sl = (slice(y0, y0 + h), slice(x0, x0 + w))
            if robot_mask_u8 is not None:
                mask = mask & (robot_mask_u8[sl] == 0)
            if self.roi is not None:
                mask = mask & (self.roi[sl] > 0)
            if mask.any():
                out.append((mask.astype(np.uint8), x0, y0))
        return out
