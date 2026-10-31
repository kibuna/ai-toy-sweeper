"""Gazebo の World の照明・床の色を設定から作る（2026-10-19、sim2real のため照明を変えて試す）。"""
import dataclasses

from toy_sweeper.config import load_all
from toy_sweeper.gazebo.sdf_gen import world_sdf


def test_world_lighting_from_config():
    cfg = load_all()
    s = world_sdf(cfg)
    g = cfg.sim.gazebo
    # 既定は今までの描画と同じ値（太陽 0.9、向き (−0.3, 0.2, −0.9)、影あり、環境光は SDF の既定 0.4）
    assert "<diffuse>0.9 0.9 0.9 1</diffuse><direction>-0.3 0.2 -0.9</direction>" in s
    assert "<cast_shadows>true</cast_shadows>" in s and "<ambient>0.4 0.4 0.4 1</ambient>" in s
    assert "<ambient>0.55 0.45 0.35 1</ambient><diffuse>0.55 0.45 0.35 1</diffuse>" in s      # 床
    assert "<ambient>0.88 0.84 0.76 1</ambient><diffuse>0.88 0.84 0.76 1</diffuse>" in s      # 作業領域
    g2 = dataclasses.replace(g, light_diffuse=(1.0, 0.8, 0.6), light_direction=(0.5, 0.0, -0.5), light_cast_shadows=False,
                             scene_ambient=(0.2, 0.2, 0.25), work_area_rgb=(0.6, 0.6, 0.6))
    s2 = world_sdf(dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, gazebo=g2)))
    assert "<diffuse>1.0 0.8 0.6 1</diffuse><direction>0.5 0.0 -0.5</direction>" in s2
    assert "<cast_shadows>false</cast_shadows>" in s2 and "<ambient>0.2 0.2 0.25 1</ambient>" in s2
    assert "<ambient>0.6 0.6 0.6 1</ambient><diffuse>0.6 0.6 0.6 1</diffuse>" in s2
