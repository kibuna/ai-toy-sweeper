"""Gazebo の物体ごとのマスクを撮るカメラ（2026-10-25、学習する検出器のデータを作るため）。"""
import dataclasses

from toy_sweeper.config import load_all
from toy_sweeper.gazebo.sdf_gen import SEG_LABELS, SEG_TOPIC, bridge_args, world_sdf


def _with_seg(cfg, on: bool):
    return dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, gazebo=dataclasses.replace(cfg.sim.gazebo,
                                                                                                segmentation_camera=on)))


def test_segmentation_camera_off_by_default():
    cfg = load_all()
    assert not cfg.sim.gazebo.segmentation_camera
    s = world_sdf(cfg, camera=True)
    assert 'type="segmentation"' not in s and "gz-sim-label-system" not in s
    assert not any(SEG_TOPIC in a for a in bridge_args(cfg, camera=True))


def test_segmentation_camera_same_view_and_labels():
    cfg = _with_seg(load_all(), True)
    s = world_sdf(cfg, camera=True)
    g = cfg.sim.gazebo
    # RGB のカメラと同じ姿勢・画角・画素数（同じモデルの中の 2 つ目のセンサー）
    assert s.count(f"<horizontal_fov>{g.camera_hfov_rad}</horizontal_fov>") == 2
    assert "<segmentation_type>instance</segmentation_type>" in s
    # 物体のモデルごとに種類のラベル（ロボット・床・壁にはラベルを付けない = 背景）
    n_lego = s.count(f"<label>{SEG_LABELS['lego']}</label>")
    n_car = s.count(f"<label>{SEG_LABELS['toy_car']}</label>")
    assert n_lego > 0 and n_car > 0 and n_lego + n_car == s.count("gz-sim-label-system")
    assert f"{SEG_TOPIC}/labels_map@sensor_msgs/msg/Image[gz.msgs.Image" in bridge_args(cfg, camera=True)
    # カメラが無ければマスクのカメラも無い
    assert 'type="segmentation"' not in world_sdf(cfg, camera=False)
