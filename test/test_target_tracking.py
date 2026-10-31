"""field.target_tracking.TargetTracking（リハーサルと実機のランタイムで共通の対象の追跡、2026-10-19）を、合成した検出で試す。"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from toy_sweeper.config import load_all
from toy_sweeper.field.config import load_field_cfg
from toy_sweeper.field.interfaces import FieldStop
from toy_sweeper.field.rehearsal import load_background_cfg
from toy_sweeper.field.runtime import Assignment
from toy_sweeper.field.target_tracking import TargetTracking
from toy_sweeper.geometry import Pose2D
from toy_sweeper.perception import Detection, load_perception_cfg
from toy_sweeper.robot_model import gripper_center_x
from toy_sweeper.spaces import GripperState, ObjectType
from toy_sweeper.task import EpisodeSpec, goal_rects
from toy_sweeper.task_manager import TidyInstruction

GREEN, BLUE = (81.0, 155.0, 106.0), (90.0, 129.0, 189.0)
FRAME = SimpleNamespace(image=np.zeros((4, 4, 3), np.uint8))


class FakeDetector:
    """BackgroundDetector と同じ形。dets と、ロボットの陰の判定 hidden(xy) を外から決める。"""

    def __init__(self):
        self.cfg = load_background_cfg()
        self.dets: list = []
        self.hidden = lambda xy: False

    def __call__(self, frame, robot, jaw_angle):
        return list(self.dets)

    def mask(self, shape, robot, jaw_angle):
        return None

    def hidden_by_robot(self, mask, xy):
        return self.hidden(xy)


def _det(xy, rgb=GREEN, length=0.032):
    return Detection(Pose2D(xy[0], xy[1], 0.0), (length, 0.016), rgb, (0, 0, 0, 0))


@pytest.fixture()
def setup():
    cfg = load_all()
    det = FakeDetector()
    tt = TargetTracking(cfg, load_field_cfg(cfg).perception, load_perception_cfg(), det)
    robot = Pose2D(0.0, 0.0, 0.0)
    gc = (gripper_center_x(cfg.sim), 0.0)
    goal = goal_rects(cfg.sim)["top_right"]
    objs = (SimpleNamespace(name="t", type=ObjectType.LEGO, pose=Pose2D(0.3, 0, 0), color="green"),
            SimpleNamespace(name="o", type=ObjectType.LEGO, pose=Pose2D(0.0, 0.2, 0), color="green"))
    ep = EpisodeSpec(2, robot, objs, ("t",), None, "top_right", goal)
    return cfg, det, tt, robot, gc, ep


def _start(tt, det, ep, t_xy, o_xy):
    det.dets = [_det(t_xy), _det(o_xy)]
    tracks = tt.process(FRAME, Pose2D(0, 0, 0), GripperState.OPEN)
    tid_t = next(i for i, tr in tracks.items() if math.dist(tr.pose.xy, t_xy) < 1e-6)
    tid_o = next(i for i, tr in tracks.items() if math.dist(tr.pose.xy, o_xy) < 1e-6)
    a = Assignment(ep, {"t": tid_t, "o": tid_o}, TidyInstruction(("lego",), ("green",), (), "top_right"), {"t": GREEN})
    tt.start(a, tracks)
    return a


def test_carried_in_open_jaws_while_hidden(setup):
    """開いた爪で押して運ぶ対象（爪の中で見えた後、ロボットの陰で見えない）を、爪の中心に補い続ける。陰でなければ補わない。"""
    cfg, det, tt, robot, gc, ep = setup
    a = _start(tt, det, ep, gc, (0.0, 0.2))
    tt.process(FRAME, robot, GripperState.OPEN)   # 割り当ての後のフレームで、爪の中に見えた対象を覚える
    assert tt.held_target == "t"
    det.dets = [_det((0.0, 0.2))]          # 対象が見えなくなった
    det.hidden = lambda xy: True
    moved = Pose2D(0.04, 0.0, 0.0)          # 2 歩進んだ
    for _ in range(3):
        tracks = tt.process(FRAME, moved, GripperState.OPEN)
    t = tracks[a.track_of["t"]]
    assert t.missed == 0 and math.dist(t.pose.xy, tt.gripper_center(moved)) < 1e-6
    det.hidden = lambda xy: False           # 陰でなければ（見えるはずなのに見えない）補わない
    tracks = tt.process(FRAME, moved, GripperState.OPEN)
    assert tracks[a.track_of["t"]].missed == 1


def test_jaw_swap_in_closed_jaws(setup):
    """閉じた爪の中に別の名前の追跡が見え、挟んだ対象の追跡が見えなければ、名前を入れ替える。"""
    cfg, det, tt, robot, gc, ep = setup
    a = _start(tt, det, ep, gc, (gc[0] + 0.04, 0.0))
    tid_t, tid_o = a.track_of["t"], a.track_of["o"]
    tt.note_close({tid_t: SimpleNamespace(pose=Pose2D(*gc, 0.0), missed=0)}, robot)
    assert tt.held_target == "t"
    # 対象外の追跡が爪の中の検出を取り、対象の追跡は見えない（対象外の元の位置には何もない）
    tt.tracker.tracks[tid_o].pose = Pose2D(gc[0] + 0.01, 0.0, 0.0)
    tt.tracker.tracks[tid_t].pose = Pose2D(gc[0] - 0.06, 0.0, 0.0)
    det.dets = [_det((gc[0] + 0.005, 0.0))]
    tt.process(FRAME, robot, GripperState.CLOSED)
    assert a.track_of == {"t": tid_o, "o": tid_t} and tt.stats["jaw_swap"] == 1


def test_reattach_after_recover(setup):
    """仕切り直しの後、追跡の無い対象を同じ色の名前のない追跡に付け直す。無ければ on_fail を呼んでから target_lost。"""
    cfg, det, tt, robot, gc, ep = setup
    a = _start(tt, det, ep, (0.3, 0.0), (0.0, 0.2))
    tt.tracker.tracks.pop(a.track_of["t"])           # 消えた
    det.dets = [_det((0.0, 0.2)), _det((0.35, 0.05), rgb=BLUE)]
    tt.process(FRAME, robot, GripperState.OPEN)
    calls = []
    with pytest.raises(FieldStop) as e:
        tt.reattach_after_recover(on_fail=lambda: calls.append(1))
    assert e.value.reason == "target_lost" and calls == [1]
    det.dets = [_det((0.0, 0.2)), _det((0.35, 0.05))]   # 同じ色の物体が見えた
    tracks = tt.process(FRAME, robot, GripperState.OPEN)
    new = next(i for i, tr in tracks.items() if math.dist(tr.pose.xy, (0.35, 0.05)) < 1e-6)
    tt.reattach_after_recover()
    assert a.track_of["t"] == new


def test_process_before_assignment_does_not_insert(setup):
    """割り当ての前（指示を受けて同定するまで）は補わない（爪の中心がロボットの陰でも）。2026-10-19、割り当ての前に
    補いを呼んで、覚えている対象が無く例外で止まった（リハーサルの 60 盤面の最初の盤面）。"""
    cfg, det, tt, robot, gc, ep = setup
    det.hidden = lambda xy: True
    det.dets = [_det((0.0, 0.2))]
    tracks = tt.process(FRAME, robot, GripperState.OPEN)
    assert len(tracks) == 1 and tt.last_held == []


@pytest.mark.parametrize("merged_max", [0.0, 0.15])
def test_reattach_to_merged_same_color_blob(setup, merged_max):
    """同じ色の物体とくっついて大きな 1 つの塊に見える対象（2026-10-23、Gazebo のリハーサルの盤面 20064: LEGO が同じ色の
    ミニカーと 133 mm の塊になり、大きさが違うので付け直せず見失った）。merged_max_length_m > 0 なら、普通の候補が無いとき、
    同じ色・対象より大きく merged_max_length_m 以下の名前のない塊に付け直す（塊ごと押して入れる。どちらも運べば
    よいので）。0（今まで）は付け直さない。上限より長い塊・違う色の塊には付け直さない。"""
    import dataclasses
    cfg, det, tt, robot, gc, ep = setup
    det.cfg = dataclasses.replace(det.cfg, merged_max_length_m=merged_max)
    tt.c = det.cfg
    a = _start(tt, det, ep, (0.3, 0.0), (0.0, 0.2))
    tt.tracker.tracks.pop(a.track_of["t"])
    det.dets = [_det((0.0, 0.2)), _det((0.32, 0.0), length=0.133), _det((0.25, 0.05), rgb=BLUE, length=0.1),
                _det((0.28, -0.05), length=0.2)]
    tracks = tt.process(FRAME, robot, GripperState.OPEN)
    merged = next(i for i, tr in tracks.items() if math.dist(tr.pose.xy, (0.32, 0.0)) < 1e-6)
    if merged_max == 0.0:
        with pytest.raises(FieldStop):
            tt.reattach_after_recover()
        return
    tt.reattach_after_recover()
    assert a.track_of["t"] == merged and tt.stats["merged_reattach"] == 1


def test_give_up_unfound_targets_after_recover(setup):
    """止めずに続ける（persist）とき、仕切り直しても見つからない対象はあきらめて、ほかの対象を続ける（2026-10-23。前は見つからないまま「4 歩下がる」を繰り返し、作業領域の外へ下がった）。見えていない（missed > 0）対象も
    あきらめる。persist でなければ今までどおり target_lost で止める。"""
    cfg, det, tt, robot, gc, ep = setup
    import dataclasses
    ep2 = dataclasses.replace(ep, target_names=("t", "o"))
    det.dets = [_det((0.3, 0.0)), _det((0.0, 0.2))]
    tracks = tt.process(FRAME, robot, GripperState.OPEN)
    ids = {n: next(i for i, tr in tracks.items() if math.dist(tr.pose.xy, xy) < 1e-6)
           for n, xy in (("t", (0.3, 0.0)), ("o", (0.0, 0.2)))}
    a = Assignment(ep2, dict(ids), TidyInstruction(("lego",), ("green",), (), "top_right"), {"t": GREEN, "o": GREEN})
    tt.start(a, tracks)
    tt.tracker.tracks.pop(ids["t"])                      # t は消えた。o は見えている
    det.dets = [_det((0.0, 0.2))]
    tt.process(FRAME, robot, GripperState.OPEN)
    with pytest.raises(FieldStop):
        tt.settle_after_recover(persist=False)
    assert tt.settle_after_recover(persist=True) == ["t"]
    assert a.episode.target_names == ("o",) and a.given_up == ["t"]
    assert tt.settle_after_recover(persist=True) == []   # o は見えている


def test_controller_after_give_up(setup):
    """あきらめた後も、歩数の上限は始めの対象の数のまま。主対象があきらめた対象なら残りの対象に移る。"""
    import dataclasses
    from toy_sweeper.field.config import load_field_cfg as lf
    from toy_sweeper.field.runtime import Controller
    cfg, det, tt, robot, gc, ep = setup
    a = Assignment(dataclasses.replace(ep, target_names=("t", "o")), {"t": 1, "o": 2},
                   TidyInstruction(("lego",), ("green",), (), "top_right"), {"t": GREEN, "o": GREEN})
    ctl = Controller(cfg, load_perception_cfg(), None, a, lf(cfg).perception)
    before = ctl.max_steps()
    a.give_up(["t"])
    assert ctl.max_steps() == before and a.episode.target_names == ("o",)
    with pytest.raises(ValueError):
        a.give_up(["x"])                                  # 対象でない名前


def test_note_close_ignores_unseen_target(setup):
    """爪を閉じる直前に見えていない（missed > 0、最後の位置が爪の間に残っているだけの）対象は、挟んだことにしない（2026-10-27
    実機 23:51:19: 見えない 3 フレームのまま閉じ、爪の中心に補って「持っている」になり、空の爪でゴールへ運んだ）。"""
    cfg, det, tt, robot, gc, ep = setup
    a = _start(tt, det, ep, gc, (gc[0] + 0.04, 0.0))
    tid_t = a.track_of["t"]
    tt.note_close({tid_t: SimpleNamespace(pose=Pose2D(*gc, 0.0), missed=3)}, robot)
    assert tt.held_target is None
    tt.note_close({tid_t: SimpleNamespace(pose=Pose2D(*gc, 0.0), missed=0)}, robot)
    assert tt.held_target == "t"


def test_reattach_target_whose_track_is_kept_in_robot_shadow(setup):
    """仕切り直しで下がった後、対象の追跡がロボットの陰で消されずに残っていても（missed > 0）、同じ色の名前のない追跡が
    見えていれば付け直す（2026-10-30 実機 002_105703: 爪の前で赤いブロックが車体の陰に入り、下がると 113 mm 先に見えたのに、
    古い追跡が陰で残っていたので付け直さず、あきらめて target_lost で止まった）。見えなければ今までどおりあきらめる。"""
    cfg, det, tt, robot, gc, ep = setup
    a = _start(tt, det, ep, (0.3, 0.0), (0.0, 0.2))
    old = a.track_of["t"]
    det.hidden = lambda xy: math.dist(xy, (0.3, 0.0)) < 0.01   # 最後の位置はロボットの陰
    det.dets = [_det((0.0, 0.2))]
    for _ in range(3):
        tt.process(FRAME, robot, GripperState.OPEN)
    assert tt.tracker.tracks[old].missed > 0
    det.dets = [_det((0.0, 0.2)), _det((0.36, 0.0))]           # 下がったら、同じ色の物体が少し先に見えた
    tracks = tt.process(FRAME, robot, GripperState.OPEN)
    new = next(i for i, tr in tracks.items() if math.dist(tr.pose.xy, (0.36, 0.0)) < 1e-6)
    assert new != old
    assert tt.settle_after_recover(persist=True) == []
    assert a.track_of["t"] == new and a.given_up == []


def test_give_up_target_kept_in_robot_shadow_without_candidate(setup):
    """陰で残った追跡の対象でも、同じ色の候補が見えなければ今までどおりあきらめる（persist）。"""
    cfg, det, tt, robot, gc, ep = setup
    a = _start(tt, det, ep, (0.3, 0.0), (0.0, 0.2))
    det.hidden = lambda xy: math.dist(xy, (0.3, 0.0)) < 0.01
    det.dets = [_det((0.0, 0.2)), _det((0.36, 0.0), rgb=BLUE)]
    for _ in range(3):
        tt.process(FRAME, robot, GripperState.OPEN)
    assert tt.settle_after_recover(persist=True) == ["t"]


def test_held_ignores_other_color_blob_in_jaws(setup):
    """閉じた爪の中に、挟んだ対象と違う色の塊（爪の先の灰色の梁。ブロックを挟むと爪が閉じ切らず、除く範囲の外に出る）が
    見えても、挟んだ対象を爪の中心に補い続ける（2026-10-30 実機 12:37:52: 爪の中心から 21〜25 mm に灰色 (140, 140, 105) の塊が
    出て補いが止まり、見えない 8 周期で爪を開いて下がった）。同じ色の塊が爪の中に見えれば、それが対象なので補わない。"""
    cfg, det, tt, robot, gc, ep = setup
    tt.tracker.chroma_gate = load_field_cfg(cfg).real.track_chroma_gate   # 実機と同じく、色の違う検出は追跡につながない
    a = _start(tt, det, ep, gc, (gc[0] + 0.3, 0.0))
    gray = (140.0, 140.0, 105.0)
    tracks = tt.process(FRAME, robot, GripperState.OPEN)    # 閉じる直前、対象は爪の中に見えている
    tt.note_close(tracks, robot)
    assert tt.held_target == "t"
    det.hidden = lambda xy: True                             # 閉じた後は隠れて、灰色の塊だけが写る
    det.dets = [_det((gc[0] + 0.3, 0.0)), _det((gc[0] + 0.01, 0.015), rgb=gray)]
    for _ in range(10):
        tracks = tt.process(FRAME, robot, GripperState.CLOSED)
    assert tracks[a.track_of["t"]].missed == 0 and tt.held_target == "t"
    det.dets = [_det((gc[0] + 0.3, 0.0)), _det((gc[0] + 0.005, 0.0))]   # 同じ色が爪の中に見えた
    tt.process(FRAME, robot, GripperState.CLOSED)
    assert tt.last_held == []
