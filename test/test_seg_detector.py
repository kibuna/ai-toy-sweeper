"""学習する検出器の推論（toy_sweeper.seg_detector）の前後の処理を、合成した出力で試す。"""
import numpy as np

from toy_sweeper.seg_detector import letterbox, postprocess


def test_letterbox_center_pad():
    img = np.full((1080, 1920, 3), 200, np.uint8)
    x, r, (left, top) = letterbox(img, (736, 1280))
    assert x.shape == (1, 3, 736, 1280) and x.dtype == np.float32
    assert abs(r - 2 / 3) < 1e-9 and (left, top) == (0, 8)
    assert np.allclose(x[0, :, :8], 114 / 255) and np.allclose(x[0, :, 8:728], 200 / 255) and np.allclose(x[0, :, 728:], 114 / 255)


def _outputs(boxes_in, cls_scores, coefs, protos):
    """ultralytics の ONNX の出力の形: output0 (1, 4 + nc + nm, N)（箱は入力の画素で中心・幅・高さ）、output1 (1, nm, H/4, W/4)。"""
    o = np.concatenate([np.asarray(boxes_in, np.float32).T, np.asarray(cls_scores, np.float32).T,
                        np.asarray(coefs, np.float32).T])[None]
    return o, np.asarray(protos, np.float32)[None]


def test_postprocess_masks_in_original_pixels():
    # 入力 736×1280、元 1080×1920（r = 2/3、上に 8 の詰め物）。プロトタイプ 184×320（入力の 1/4）
    protos = np.full((1, 184, 320), -10.0, np.float32)
    # 元の画素で x 600〜659、y 300〜329 の物体 → 入力で x 400〜440、y 208〜228 → プロトタイプで x 100〜110、y 52〜57
    protos[0, 52:57, 100:110] = 10.0
    boxes = [[420, 218, 40, 20], [420, 218, 40, 20], [900, 500, 30, 30]]   # 2 つ目は 1 つ目と重なる（NMS で消える）
    scores = [[0.9, 0.05], [0.6, 0.1], [0.1, 0.2]]                         # 3 つ目は閾値より低い
    coefs = [[1.0], [1.0], [1.0]]
    o0, o1 = _outputs(boxes, scores, coefs, protos)
    inst = postprocess(o0, o1, 2 / 3, (0, 8), (1080, 1920), conf=0.25, iou=0.7, mask_threshold=0.5)
    assert len(inst) == 1
    cls, score, mask, x0, y0 = inst[0]
    assert cls == 0 and abs(score - 0.9) < 1e-6
    ys, xs = np.nonzero(mask)
    xs, ys = xs + x0, ys + y0
    assert abs(xs.min() - 600) <= 2 and abs(xs.max() - 659) <= 2 and abs(ys.min() - 300) <= 2 and abs(ys.max() - 329) <= 2


def test_postprocess_mask_cropped_to_box():
    """マスクは箱の外を 0 にする（ultralytics の crop_mask と同じ）。"""
    protos = np.full((1, 184, 320), 10.0, np.float32)          # どこも物体
    o0, o1 = _outputs([[420, 218, 40, 20]], [[0.9, 0.0]], [[1.0]], protos)
    (_, _, mask, x0, y0), = postprocess(o0, o1, 2 / 3, (0, 8), (1080, 1920), conf=0.25, iou=0.7, mask_threshold=0.5)
    ys, xs = np.nonzero(mask)
    assert xs.min() + x0 >= 599 and xs.max() + x0 <= 660 and ys.min() + y0 >= 299 and ys.max() + y0 <= 330


class _FakeModel:
    def __init__(self, inst):
        self.inst = inst

    def __call__(self, rgb):
        return self.inst


def test_seg_detector_like_background_detector():
    """背景との差の検出と同じ設定・範囲で、塊だけをモデルの出力にする。ロボットの範囲の画素は除き、残らなければ捨てる。"""
    from toy_sweeper.calibration import camera_from_gazebo_pose
    from toy_sweeper.field.rehearsal import load_background_cfg
    from toy_sweeper.perception_oblique import BackgroundDetector
    from toy_sweeper.seg_data import look_at_pose
    from toy_sweeper.seg_detector import SegDetector
    cam = camera_from_gazebo_pose(look_at_pose((0.0, -0.85, 0.8), (0.0, 0.0, 0.0), 0.0), 1.22, (1920, 1080))
    bg = np.zeros((1080, 1920, 3), np.uint8)
    base = BackgroundDetector(cam, bg, load_background_cfg(), lambda r, j: [], 0.01, keep_half_m=(0.5, 0.5))
    u, v = np.round(cam.to_px(np.array([[0.1, 0.05]]), 0.008)[0]).astype(int)
    m1 = np.ones((20, 40), np.uint8)                       # 物体 1（中心が床の (0.1, 0.05) 付近）
    m2 = np.ones((10, 10), np.uint8)                       # 物体 2: 全部ロボットの範囲
    det = SegDetector.like(base, _FakeModel([(0, 0.9, m1, u - 20, v - 10), (1, 0.8, m2, 100, 100)]))
    assert det.cfg is base.cfg and det.keep_half == (0.5, 0.5) and isinstance(det, BackgroundDetector)
    rmask = np.zeros((1080, 1920), np.uint8)
    rmask[100:110, 100:110] = 1
    rmask[v - 10:v + 10, u + 10:u + 20] = 1                # 物体 1 の右の 1/4 もロボットの範囲
    dets = det.detect(bg, rmask)
    assert len(dets) == 1
    assert abs(dets[0].pose.x - 0.1) < 0.01 and abs(dets[0].pose.y - 0.05) < 0.01
    assert dets[0].bbox_px[2] <= u + 10                    # ロボットの範囲の画素は使わない
