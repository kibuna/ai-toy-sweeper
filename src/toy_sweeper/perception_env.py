"""Policy の入力を、真値ではなくカメラ画像の検出・追跡から作る評価用ラッパー（シミュレーション上の閉ループ試験）。

- 画像は toy_sweeper.render.draw_scene の合成画像（真値の姿勢から描く）。
- ロボットの姿勢は ArUco で測る想定で、真値に real_world_effects の観測ノイズを足したものを使う。
- 対象の同定: エピソード開始時に、真の物体と追跡を位置で対応付ける（VLM が最初に分類する役の代わり）。
  その後は追跡の ID だけで対象を追う。対象の追跡が消えたら "target_lost" として打ち切る。
- 報酬と終了判定は真値のまま（ToySweeperEnv）。
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from .geometry import box_corners
from .perception import Homography, PerceptionCfg, Tracker, detect_objects
from .render import SCENE_PX_PER_M, draw_scene
from .robot_model import SIDES, body_center_x, jaw_polygons
from .spaces import ObjectType, SlotObject


def scene_homography(cfg) -> Homography:
    hx, hy = cfg.sim.work_area.size_m[0] / 2, cfg.sim.work_area.size_m[1] / 2
    w, h = 2 * hx * SCENE_PX_PER_M, 2 * hy * SCENE_PX_PER_M
    return Homography([(0, 0), (w, 0), (w, h), (0, h)], [(-hx, hy), (hx, hy), (hx, -hy), (-hx, -hy)])


def robot_polygons(cfg, pose, jaw_angle: float) -> list[list[tuple[float, float]]]:
    """ロボットの部品（車体、爪の腕と梁）の外形（床面座標）。ArUco の姿勢とグリッパーの指令から分かる。"""
    bx, by = cfg.sim.robot.body_size_m[:2]
    cx = body_center_x(cfg.sim)
    body = [pose.transform_point((cx + sx * bx / 2, sy * by / 2)) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    jaws = [[pose.transform_point(p) for p in poly] for side in SIDES for poly in jaw_polygons(cfg.sim, side, jaw_angle)]
    return [body, *jaws]


class PerceptionObsWrapper(gym.Wrapper):
    def __init__(self, env, pcfg: PerceptionCfg):
        super().__init__(env)
        self.pcfg = pcfg
        self.hom = scene_homography(env.cfg)
        self.metrics: list[dict] = []

    def _perceive(self):
        e = self.env
        poses = {o.name: e.backend.object_pose(o.name) for o in e.episode.objects}
        g = e.cfg.sim.gripper
        jaw = g.jaw_closed_angle_rad if e.backend.gripper_state().name == "CLOSED" else g.jaw_open_angle_rad
        img, _ = draw_scene(e.cfg, e.backend.robot_pose(), e.episode.objects, poses, jaw)
        robot = e._noisy(e.backend.robot_pose())
        dets = detect_objects(img, self.hom, robot_polygons(e.cfg, robot, jaw), self.pcfg)
        return self.tracker.update(dets), robot, poses

    def _obs(self, tracks, robot, true_poses):
        e = self.env
        ep = e.episode
        # 物体ごとに、真の位置から 2 cm 以内に追跡があるか（検出の再現率）と、対象の位置誤差
        found = sum(any(math.dist(tr.pose.xy, p.xy) < 0.02 for tr in tracks.values()) for p in true_poses.values())
        errs = [math.dist(tracks[self.track_of[n]].pose.xy, true_poses[n].xy)
                for n in ep.target_names if self.track_of.get(n) in tracks]
        self.metrics.append({"target_err_m": float(np.mean(errs)) if errs else math.nan, "found": found,
                             "n_objects": len(true_poses), "n_tracks": len(tracks)})
        if any(self.track_of.get(n) not in tracks for n in ep.target_names):
            return None   # 対象を見失った
        poses = {n: tracks[tid].pose for n, tid in self.track_of.items() if tid in tracks}
        named = set(self.track_of.values())
        # 物体名と結び付いていない追跡: 開始時の位置が分からないので、変位 0・置き場なしとして渡す（明示的な扱い）
        extra = tuple(SlotObject(tr.pose, ObjectType.TOY_CAR if tr.size_m[0] >= self.pcfg.toy_car_min_length_m
                                 else ObjectType.LEGO, (0.0, 0.0), None)
                      for tid, tr in tracks.items() if tid not in named)
        return e.obs_builder.build(e.snapshot_from(poses, robot, extra))

    def reset(self, **kw):
        _, info = self.env.reset(**kw)
        e = self.env
        if not e.episode.target_names:
            raise ValueError("PerceptionObsWrapper は物体のある Stage 用（Stage 0 は不可）")
        self.tracker = Tracker(self.pcfg, 0.0)   # 色の門なし（合成画像の検出、今まで）
        tracks, robot, poses = self._perceive()
        # 開始時の同定（VLM の代わり）: 真の物体に最も近い追跡を結び付ける。後は追跡の ID だけで追う
        self.track_of: dict[str, int] = {}
        for o in e.episode.objects:
            near = min(tracks.values(), key=lambda tr: math.dist(tr.pose.xy, poses[o.name].xy), default=None)
            if near is None or math.dist(near.pose.xy, poses[o.name].xy) > 0.02 or near.id in self.track_of.values():
                if o.name in e.episode.target_names:
                    raise RuntimeError(f"開始時に対象 {o.name} を検出できない（盤面の生成か検出の設定の問題）")
                continue
            self.track_of[o.name] = near.id
        return self._obs(tracks, robot, poses), info

    def step(self, action):
        _, r, term, trunc, info = self.env.step(action)
        tracks, robot, poses = self._perceive()
        obs = self._obs(tracks, robot, poses)
        if obs is None:   # 対象を見失った: 打ち切る（失敗）
            info = {**info, "failure": "target_lost", "is_success": False}
            return np.zeros(self.observation_space.shape, np.float32), r, True, trunc, info
        return obs, r, term, trunc, info
