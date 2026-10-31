"""field.rehearsal の判定のうち、Gazebo なしで試せる純粋な部分（2026-10-19）。"""
from toy_sweeper.field.target_tracking import kicked_detection
from toy_sweeper.geometry import Pose2D
from toy_sweeper.perception import Detection

GREEN, BLUE = (81.0, 155.0, 106.0), (90.0, 129.0, 189.0)
DARK_GREEN = (54.0, 103.0, 71.0)   # 同じ緑の 2/3 の明るさ


def _det(x, y, rgb, length=0.032):
    return Detection(Pose2D(x, y, 0.0), (length, 0.016), rgb, (0, 0, 0, 0))


def test_kicked_detection_ignores_other_known_objects():
    """挟み損ねの証拠は、爪の外の近くに見える同じ色・大きさの検出のうち、ほかの名前のある物体の追跡に当たらないもの。
    2026-10-19 の盤面 20011・20003: ほかの物体（色の近い青い LEGO、同じ色のもう 1 つの対象）を弾かれた対象とみなした。"""
    gc = (0.0, 0.0)
    kw = dict(gc=gc, ref_rgb=GREEN, ref_len=0.032, chroma_tol=8.0, inner_m=0.025, outer_m=0.2, size_ratio=1.4, gate_m=0.05)
    other = _det(0.15, 0.0, DARK_GREEN)                # 同じ色度。名前のある物体の追跡が 1 cm の所にある
    assert kicked_detection([other], others_xy=[(0.16, 0.0)], **kw) is None
    assert kicked_detection([other], others_xy=[], **kw) is other   # 名前のある追跡が無ければ、証拠にする
    assert kicked_detection([_det(0.15, 0.0, BLUE)], others_xy=[], **kw) is None   # 色度の差 52（RGB では 88 で前は通った）
    kicked = _det(0.08, 0.02, GREEN)
    assert kicked_detection([other, kicked], others_xy=[(0.16, 0.0)], **kw) is kicked
    assert kicked_detection([_det(0.01, 0.0, GREEN)], others_xy=[], **kw) is None        # 爪の中（25 mm 以内）
    assert kicked_detection([_det(0.25, 0.0, GREEN)], others_xy=[], **kw) is None        # 遠い
    assert kicked_detection([_det(0.08, 0.0, GREEN, 0.065)], others_xy=[], **kw) is None  # 大きさが違う


def test_lost_reason_classifies_by_truth():
    """見失った対象の真の位置での理由（field.rehearsal.lost_reason、調べるための記録）。"""
    from toy_sweeper.field.rehearsal import lost_reason
    kw = dict(ref_rgb=GREEN, ref_len=0.032, chroma_tol=8.0, size_ratio=1.6, near_m=0.03)
    p = (0.1, 0.1)
    assert lost_reason(p, [], hidden=True, **kw) == "hidden_by_robot"
    assert lost_reason(p, [], hidden=False, **kw) == "no_detection"
    assert lost_reason(p, [_det(0.2, 0.2, GREEN)], hidden=False, **kw) == "no_detection"     # 近くに無い
    assert lost_reason(p, [_det(0.11, 0.1, BLUE)], hidden=False, **kw) == "color_off"
    assert lost_reason(p, [_det(0.11, 0.1, GREEN, 0.065)], hidden=False, **kw) == "size_off"
    assert lost_reason(p, [_det(0.11, 0.1, GREEN)], hidden=False, **kw) == "detected"
    # 色の合う検出と合わない検出が両方あれば、合う方
    assert lost_reason(p, [_det(0.105, 0.1, BLUE), _det(0.11, 0.1, GREEN)], hidden=False, **kw) == "detected"


def test_jaw_target_follows_visibility():
    """爪の中にある対象（field.rehearsal.jaw_target、2026-10-19）: 爪の中心から radius 以内に見えた対象を覚え、爪の外で
    見えたら忘れる。見えていなければ（爪の中で隠れている）覚えたまま。爪の開閉によらない（開いた爪で押して運ぶ間も）。"""
    from toy_sweeper.field.target_tracking import jaw_target
    gc, r = (0.0, 0.0), 0.025
    assert jaw_target(None, visible={"a": (0.3, 0.0)}, gc=gc, radius=r) is None
    assert jaw_target(None, visible={"a": (0.3, 0.0), "b": (0.01, 0.0)}, gc=gc, radius=r) == "b"
    assert jaw_target(None, visible={"a": (0.02, 0.0), "b": (0.01, 0.0)}, gc=gc, radius=r) == "b"   # 近い方
    assert jaw_target("b", visible={"b": None}, gc=gc, radius=r) == "b"            # 隠れた: 覚えたまま
    assert jaw_target("b", visible={"b": (0.005, 0.0)}, gc=gc, radius=r) == "b"
    assert jaw_target("b", visible={"b": (0.06, 0.0)}, gc=gc, radius=r) is None    # 爪の外で見えた（置いて下がった）
    # 覚えている対象が爪の外に出た同じフレームで、別の対象が爪の中に見えれば、そちらを覚える
    assert jaw_target("b", visible={"a": (0.0, 0.01), "b": (0.06, 0.0)}, gc=gc, radius=r) == "a"


def test_jaw_swap_gives_closed_jaws_to_held_target():
    """閉じた爪の中の名前の付け替え（field.target_tracking.jaw_swap、2026-10-19）: 挟んだ対象の追跡が見えず、閉じた爪の
    中に別の名前の追跡が見えていて、その色が挟んだ対象の色と合えば、2 つの名前の追跡を入れ替える（閉じた爪に入る物体は
    1 つで、挟んだのは対象。盤面 20021 で近くの黒い物体の追跡が、挟んだ黒い LEGO の検出を取った）。色が違えば入れ替えない
    （挟んだのは別の物体。盤面 20006・20030・20031 で、黄色・オレンジ・ピンクの物体を対象の名前にして取り違えで止まった）。
    長さの比が size_ratio を超えても入れ替えない（2026-10-25、v32 の Stage 6 盤面 20013: 挟んだ赤い LEGO 33 mm の名前を、
    爪に寄ってきて LEGO とくっついた赤いミニカーの塊 91 mm に移し、ミニカーを対象として運んだ）。"""
    from toy_sweeper.field.target_tracking import jaw_swap
    gc, r, tol = (0.0, 0.0), 0.025, 8.0
    track_of = {"t": 0, "o": 6, "p": 3}
    kw = dict(gc=gc, radius=r, ref_rgb=GREEN, chroma_tol=tol, ref_len=0.032, size_ratio=1.4)
    # 追跡ごとの (見えている位置, 色, 長辺)、見えなければ None
    vis = {0: None, 6: ((0.01, 0.0), GREEN, 0.034), 3: ((0.2, 0.0), GREEN, 0.032)}
    assert jaw_swap("t", track_of, vis, **kw) == {"t": 6, "o": 0, "p": 3}
    assert jaw_swap("t", track_of, {0: None, 6: ((0.01, 0.0), BLUE, 0.032), 3: None}, **kw) is None     # 色が違う
    assert jaw_swap("t", track_of, {0: None, 6: ((0.01, 0.0), GREEN, 0.091), 3: None}, **kw) is None    # 大きさが違う
    assert jaw_swap("t", track_of, {0: ((0.005, 0.0), GREEN, 0.032), 6: ((0.01, 0.0), GREEN, 0.032), 3: None},
                    **kw) is None   # 対象は見えている
    assert jaw_swap("t", track_of, {0: None, 6: ((0.05, 0.0), GREEN, 0.032), 3: None}, **kw) is None    # 爪の外
    assert jaw_swap(None, track_of, vis, **kw) is None
    # 対象の追跡が消えていれば、相手の名前は消えた追跡に付く（相手は見失った扱いにして、付け直しに任せる）
    assert jaw_swap("t", {"t": 9, "o": 6}, {6: ((0.01, 0.0), GREEN, 0.032)}, **kw) == {"t": 6, "o": 9}

def _truth_frame(poses):
    import numpy as np
    from toy_sweeper.field.interfaces import Frame, SimTruth
    from toy_sweeper.spaces import GripperState
    from toy_sweeper.sim2d import ObjectSpec
    from toy_sweeper.spaces import ObjectType
    objs = tuple(ObjectSpec(n, ObjectType.LEGO, p, "red") for n, p in poses.items())
    return Frame(np.zeros((2, 2, 3), np.uint8), 0.0, 0, SimTruth(Pose2D(0, 0, 0), objs, poses, 0.0, GripperState.OPEN))


def test_truth_identifier_strict_and_real_rules():
    """開始時の同定（field.sim.SimTruthIdentifier）: 2 個の物体が 1 つの塊（追跡 1 つ）に見えるとき、今まで（strict）は止める。
    実機の決まり（allow_missing、2026-10-23）は、塊を近い方の物体に結び、もう 1 個は同定しないで missing に残す（実機の VLM は
    見えた塊だけを同定して始める）。"""
    import pytest
    from toy_sweeper.field.interfaces import FieldStop, TrackView
    from toy_sweeper.field.sim import SimTruthIdentifier
    frame = _truth_frame({"a": Pose2D(0.0, 0.0, 0.0), "b": Pose2D(0.018, 0.0, 0.0), "c": Pose2D(0.3, 0.0, 0.0)})
    tracks = {1: TrackView(1, Pose2D(0.005, 0.0, 0.0), (0.05, 0.016), (200.0, 0.0, 0.0), 0),
              2: TrackView(2, Pose2D(0.3, 0.0, 0.0), (0.032, 0.016), (200.0, 0.0, 0.0), 0)}
    with pytest.raises(FieldStop, match="start_undetected"):
        SimTruthIdentifier(0.02).identify(frame, tracks)
    ident = SimTruthIdentifier(0.02, allow_missing=True)
    out = ident.identify(frame, tracks)
    assert {k: v.name for k, v in out.items()} == {1: "a", 2: "c"} and ident.missing == ["b"]


def test_unidentified_at_start_only_with_truth_identifier():
    """開始時に同定しなかった物体（数える）は、真値を知る同定（SimTruthIdentifier）でだけ分かる。VLM の同定
    （--identify vlm）は真の物体を知らないので None（数えない。0 と測ったことにしない）。2026-10-27、管理のアプリの Gazebo の
    世界（VLM）で、VLM の同定に missing が無く指示ごとに例外で終わった。"""
    from toy_sweeper.field.rehearsal import unidentified_at_start
    from toy_sweeper.field.sim import SimTruthIdentifier
    from toy_sweeper.field.vlm_identify import VlmObjectIdentifier
    ident = SimTruthIdentifier(0.02, allow_missing=True)
    ident.missing = ["b"]
    assert unidentified_at_start(ident) == ["b"]
    assert unidentified_at_start(VlmObjectIdentifier(None, None, 0.05)) is None


def test_gazebo_rig_has_what_the_talk_screen_reads():
    """話しかける画面（field.session・field.web）がリグから読むものを、Gazebo のリグも持つ（2026-10-27、管理のアプリから Gazebo を
    選ぶと、映像の送り出しが reload_if_changed の無いことで落ち、画面が出なかった）。2D のリグと同じ口（実機のリグの area_size は
    校正から起動のときに決まるので、ここでは比べない）。"""
    from toy_sweeper.field.rehearsal import GazeboRehearsalRig
    from toy_sweeper.field.rig import SimRig
    needed = ["reload_if_changed", "goal_rects", "outcome", "make_runtime", "reset_board", "run_instruction", "area_size"]
    for rig in (SimRig, GazeboRehearsalRig):
        assert [n for n in needed if not hasattr(rig, n)] == [], rig.__name__


def test_truth_check_judges_true_matching_objects_with_vlm():
    """完了の真値の確かめで見る物体。真値の同定は、同定した対象の名前（真の物体の名前）。VLM の同定の名前（obj + 追跡の番号）は
    真の物体と関係ないので、指示に合う真の物体を見る（2026-10-27、VLM の obj5 を真の obj5 と取り違え、ゴールに入っていたのに
    「まだ のこってる」と言った）。"""
    from toy_sweeper.field.rehearsal import judged_targets
    from toy_sweeper.field.sim import SimTruthIdentifier
    from toy_sweeper.field.vlm_identify import VlmObjectIdentifier
    assert judged_targets(SimTruthIdentifier(0.02), ["obj5"], ["obj1", "obj5"]) == ["obj5"]
    assert judged_targets(VlmObjectIdentifier(None, None, 0.05), ["obj5"], ["obj1"]) == ["obj1"]


def test_truth_names_of_identified_objects():
    """調べるための記録（取り違え・見失い・観測の誤差）が真値を引く名前。真値の同定はそのまま。VLM の同定は、開始時の追跡の位置に
    いちばん近い真の物体（tolerance_m より遠ければ対応なし。記録から外す）。2026-10-27、VLM の名前で真値を引いて別の物を見ていた。"""
    from toy_sweeper.field.interfaces import TrackView
    from toy_sweeper.field.rehearsal import truth_names
    from toy_sweeper.field.sim import SimTruthIdentifier
    from toy_sweeper.field.vlm_identify import VlmObjectIdentifier
    truth = {"obj0": Pose2D(0.0, 0.0, 0.0), "obj1": Pose2D(0.3, 0.0, 0.0)}
    tracks = {5: TrackView(5, Pose2D(0.29, 0.01, 0.0), (0.03, 0.016), (0.0, 0.0, 0.0), 0),
              7: TrackView(7, Pose2D(0.0, 0.2, 0.0), (0.03, 0.016), (0.0, 0.0, 0.0), 0)}
    assert truth_names(SimTruthIdentifier(0.02), {"obj0": 1}, tracks, truth, 0.02) == {"obj0": "obj0"}
    vlm = VlmObjectIdentifier(None, None, 0.05)
    assert truth_names(vlm, {"obj5": 5, "obj7": 7}, tracks, truth, 0.02) == {"obj5": "obj1"}
