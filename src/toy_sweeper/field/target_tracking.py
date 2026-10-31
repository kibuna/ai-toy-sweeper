"""対象の追跡（斜めのカメラの背景との差の検出 + 爪の中の補い・付け替え・付け直し）。リハーサル（field.rehearsal）と
実機のランタイム（field.runtime）で同じものを使う（2026-10-19）。

真値・Gazebo には依らない。真値での調べ（取り違えの記録、見失いの理由）は呼ぶ側（リハーサル）に残す。
仕切り直しの動作（爪を開いて下がる）は制御なので呼ぶ側が行い、下がった後の付け直しだけをここで行う。
"""
from __future__ import annotations

import logging
import math
from collections import Counter

from ..config import AllCfg
from ..geometry import Pose2D
from ..perception import Detection, PerceptionCfg, Tracker, chroma_distance
from ..robot_model import gripper_center_x
from ..spaces import GripperState
from .config import FieldPerceptionCfg
from .interfaces import FieldStop

log = logging.getLogger(__name__)


def kicked_detection(dets, *, gc, ref_rgb, ref_len: float, chroma_tol: float, inner_m: float, outer_m: float,
                     size_ratio: float, gate_m: float, others_xy) -> "Detection | None":
    """挟み損ねて弾いた対象とみなす検出（最初の 1 個）。爪の中心から inner_m〜outer_m、色度の差（perception.chroma_distance）が
    ref_rgb から chroma_tol 未満、
    長さの比が size_ratio 以下で、ほかの名前のある物体の追跡（others_xy）から gate_m（追跡の対応の上限）より離れたもの。
    2026-10-19: ほかの物体（Gazebo の描画で緑との色の差 88 の青い LEGO、同じ色のもう 1 つの対象）を弾かれた対象と
    みなし、挟んでいる対象の補いをやめて見失った（盤面 20011・20003）ため、名前のある物体に当たる検出を外した。"""
    for d in dets:
        r = math.dist(d.pose.xy, gc)
        if (chroma_distance(d.mean_rgb, ref_rgb) < chroma_tol and inner_m <= r <= outer_m
                and max(d.size_m[0], ref_len) / max(min(d.size_m[0], ref_len), 1e-6) <= size_ratio
                and all(math.dist(d.pose.xy, o) > gate_m for o in others_xy)):
            return d
    return None


def jaw_target(prev: "str | None", *, visible: dict, gc, radius: float) -> "str | None":
    """爪の中にある対象（2026-10-19）。visible は対象ごとの、今のフレームで見えている追跡の位置（見えていなければ None）。
    爪の中心から radius 以内に見えている対象があれば、最も近いもの。無ければ、前の対象が見えていない（爪の中で隠れている）
    なら前のまま、爪の外で見えたら None。爪の開閉によらない: フローリングの床の 60 盤面で、見失いの 8 割が爪の中で
    ロボットの陰に入った対象で、うち 3 回は開いた爪で押して運んでいる間（閉じたときだけ覚えていたので、後で閉じたときには
    追跡が元の場所に残っていて、挟んだと分からなかった）。"""
    inside = [(math.dist(xy, gc), n) for n, xy in visible.items() if xy is not None and math.dist(xy, gc) < radius]
    if inside:
        return min(inside)[1]
    if prev is not None and visible.get(prev) is None:
        return prev
    return None


def jaw_swap(held: "str | None", track_of: dict, visible: dict, *, gc, radius: float, ref_rgb,
             chroma_tol: float, ref_len: float, size_ratio: float) -> "dict | None":
    """閉じた爪の中の名前の付け替え（2026-10-19）。visible は追跡ごとの (見えている位置, 平均色, 長辺)（見えていなければ None、
    無い追跡はキーが無い）。挟んだ対象 held の追跡が見えず、別の名前の追跡が爪の中心から radius 以内に見えていて、その色が
    held の色 ref_rgb と色度で chroma_tol 未満、長辺と held の長辺 ref_len の比が size_ratio 以下なら（2026-10-25、v32 の
    Stage 6 盤面 20013: 挟んだ赤い LEGO の名前を、LEGO とくっついた赤いミニカーの塊 91 mm に移した）、
    2 つの名前の追跡を入れ替えた track_of を返す（閉じた爪に入る物体は
    1 つで、挟んだのは held。フローリングの床の 60 盤面で、近くの同じ色の物体の追跡が挟んだ対象の検出を取り、対象を
    見失った: 盤面 20006・20021）。色が違えば、挟んだのは別の物体なので入れ替えない（色を見ずに入れ替えた版で、別の色の
    物体を対象の名前にして取り違えで止まった: 20006・20030・20031）。入れ替えなければ None。"""
    if held is None or visible.get(track_of[held]) is not None:
        return None
    inside = [(math.dist(visible[t][0], gc), m) for m, t in track_of.items()
              if m != held and visible.get(t) is not None and math.dist(visible[t][0], gc) < radius
              and chroma_distance(visible[t][1], ref_rgb) < chroma_tol
              and max(visible[t][2], ref_len) / max(min(visible[t][2], ref_len), 1e-6) <= size_ratio]
    if not inside:
        return None
    m = min(inside)[1]
    return track_of | {held: track_of[m], m: track_of[held]}


class TargetTracking:
    """1 つの指示の間の、検出・追跡と対象の名前の付け方。detector は perception_oblique.BackgroundDetector と同じ形
    （__call__(frame, robot, jaw_angle) → 検出、mask(画像の形, robot, jaw_angle)、hidden_by_robot(mask, xy)、cfg）。
    start() で割り当て（field.runtime.Assignment）を渡すまでは、検出と追跡だけ。割り当ての track_of はここで書き換える。"""

    def __init__(self, cfg: AllCfg, fp: FieldPerceptionCfg, pcfg: PerceptionCfg, detector, tracker=None,
                 stats: Counter | None = None):
        self.cfg, self.fp, self.pcfg = cfg, fp, pcfg
        self.detector = detector
        self.c = detector.cfg
        self.tracker = Tracker(pcfg, self.c.track_chroma_gate) if tracker is None else tracker
        self.stats: Counter = Counter() if stats is None else stats   # 挟み損ね・付け直し・付け替えなどの回数
        self.assignment = None
        self.held_target: str | None = None    # 爪の中にある対象（jaw_target）
        self.ref_len: dict[str, float] = {}     # 同定したときの対象の長辺
        self.last_xy: dict[str, tuple[float, float]] = {}   # 対象の追跡の最後の位置（見失ったときの探し始め）
        self.last_dets: list = []               # 直前のフレームの検出（補いを除く）・ロボットのマスク・補い
        self.last_mask = None
        self.last_held: list = []

    def gripper_center(self, robot: Pose2D) -> tuple[float, float]:
        return robot.transform_point((gripper_center_x(self.cfg.sim), 0.0))

    def jaw_angle(self, gripper: GripperState) -> float:
        g = self.cfg.sim.gripper
        return g.jaw_closed_angle_rad if gripper == GripperState.CLOSED else g.jaw_open_angle_rad

    def start(self, assignment, tracks: dict) -> None:
        """割り当てを決めた直後（tracks は割り当てに使った追跡）。実機のランタイムでは process（知覚のスレッド）と別の
        スレッドから呼ぶので、割り当ては最後に置く（process は割り当てがあれば ref_len・last_xy を読む）。"""
        a = assignment
        self.ref_len = {n: tracks[a.track_of[n]].size_m[0] for n in a.episode.target_names}
        self.last_xy = {n: tracks[a.track_of[n]].pose.xy for n in a.episode.target_names}
        self.held_target = None
        self.assignment = a

    # ---------- 1 フレーム ----------
    def process(self, frame, robot: Pose2D, gripper: GripperState) -> dict:
        """検出 → 爪の中の対象が見えなければ補う → 追跡 → 閉じた爪の中の付け替え → 爪の中の対象の更新 →
        ロボットの陰の追跡を消さない → 見えない対象の付け直し。追跡（ID → perception.Track）を返す。"""
        jaw = self.jaw_angle(gripper)
        dets = self.detector(frame, robot, jaw)
        m = self.detector.mask(frame.image.shape[:2], robot, jaw)
        self.last_dets, self.last_mask = dets, m
        gc = self.gripper_center(robot)
        held = []
        # 爪の中の対象が見えないとき爪の中心に補う。閉じた爪はいつも、開いた爪は爪の中心がロボットの陰のときだけ
        # （開いた爪の間は見えることが多く、見えないのに補うと、置いて下がった物体を連れて行ってしまう）
        if self.assignment is not None and (gripper == GripperState.CLOSED or self.detector.hidden_by_robot(m, gc)):
            held = self._held(robot, dets, gripper)
        self.last_held = held
        tracks = self.tracker.update(dets + held)
        if self.assignment is not None:
            a = self.assignment
            if gripper == GripperState.CLOSED:
                new = None if self.held_target is None else jaw_swap(
                    self.held_target, a.track_of,
                    {tid: (tr.pose.xy, tr.mean_rgb, tr.size_m[0]) if tr.missed == 0 else None for tid, tr in tracks.items()},
                    gc=gc, radius=self.c.held_radius_m, ref_rgb=a.ref_rgb[self.held_target],
                    chroma_tol=self.fp.identity_chroma_tolerance, ref_len=self.ref_len[self.held_target],
                    size_ratio=self.c.reassociate_size_ratio)
                if new is not None:
                    other = next(k for k in new if k != self.held_target and new[k] != a.track_of[k])
                    log.info("閉じた爪の中の追跡 %d は %s のものではなく、挟んだ %s のものとして付け替える",
                             new[self.held_target], other, self.held_target)
                    self.stats["jaw_swap"] += 1
                    a.track_of.update(new)
            self.held_target = jaw_target(
                self.held_target, gc=gc, radius=self.c.held_radius_m,
                visible={n: tracks[a.track_of[n]].pose.xy if a.track_of[n] in tracks and tracks[a.track_of[n]].missed == 0
                         else None for n in a.episode.target_names})
        # ロボットの陰に入っている追跡は消さない（斜め上のカメラでは、車体の向こうの物体が隠れる。2026-10-18、
        # ゴールの奥の角で運んでいる LEGO が車体に隠れて見失った）。見えている扱いにはしない（missed は 0 にしない）
        for tr in tracks.values():
            if tr.missed > 0 and self.detector.hidden_by_robot(m, tr.pose.xy):
                tr.missed = min(tr.missed, self.pcfg.max_missed - 1)
        if self.assignment is not None:
            self._reassociate(tracks)
        return tracks

    def note_close(self, tracks: dict, robot: Pose2D) -> None:
        """爪を閉じる直前: 爪の間に見える対象を覚える（挟んだはずの物体。_held が補う）。見えなければ、覚えている対象
        （jaw_target、開いた爪の中で隠れている）のまま。今見えていない（missed > 0）追跡は、最後の位置が爪の間でも数えない
        （2026-10-27 実機: 見えない 3 フレームのまま閉じて補い、空の爪でゴールへ運んだ）。"""
        a = self.assignment
        gc = self.gripper_center(robot)
        inside = [n for n in a.episode.target_names
                  if a.track_of[n] in tracks and tracks[a.track_of[n]].missed == 0
                  and math.dist(tracks[a.track_of[n]].pose.xy, gc) < self.c.held_radius_m]
        self.held_target = min(inside, key=lambda n: math.dist(tracks[a.track_of[n]].pose.xy, gc),
                               default=self.held_target)

    def _held(self, robot: Pose2D, dets: list, gripper: GripperState) -> list:
        """爪の中にある対象（jaw_target）が、今は見えないとき、爪の中心にある検出として補う（挟んだ物体・開いた爪で押して
        いる物体は爪と車体に隠れやすい）。爪の外の近くに同じ色・大きさの物体が見えていれば、挟み損ねて弾いた（2026-10-18、
        Gazebo で空の爪で運んだ）か、開いた爪から離れた（置いて下がった）とみなして補わない（_reassociate がそちらに付け直す）。"""
        n = self.held_target
        if n is None:
            return []
        c = self.c
        gc = self.gripper_center(robot)
        a = self.assignment
        # 爪の中に対象と同じ色の検出が見えていれば、それが対象（追跡がつながる）なので補わない。違う色の塊（爪の先の灰色の梁。
        # 挟むと爪が閉じ切らず、除く範囲の外に出る）では補いを止めない（2026-10-30 実機 12:37:52: 灰色の塊で補いが止まり、
        # 見えない 8 周期で爪を開いて下がった）
        if any(math.dist(d.pose.xy, gc) < c.held_radius_m
               and chroma_distance(d.mean_rgb, a.ref_rgb[n]) < self.fp.identity_chroma_tolerance for d in dets):
            return []
        tid = a.track_of[n]
        if tid not in self.tracker.tracks:
            return []
        t = self.tracker.tracks[tid]
        others = [self.tracker.tracks[o].pose.xy for name, o in a.track_of.items() if name != n and o in self.tracker.tracks]
        d = kicked_detection(dets, gc=gc, ref_rgb=a.ref_rgb[n], ref_len=self.ref_len[n],
                             chroma_tol=self.fp.identity_chroma_tolerance, inner_m=c.held_radius_m,
                             outer_m=c.reassociate_max_m, size_ratio=c.reassociate_size_ratio, gate_m=self.pcfg.gate_m,
                             others_xy=others)
        if d is not None:
            closed = gripper == GripperState.CLOSED
            log.info("%s（%s と同じ色の物体が爪の外 %.0f mm に見える。色 %s、対象の色 %s、距離 %.0f）",
                     "挟み損ねたとみなす" if closed else "開いた爪から離れたとみなす", n,
                     1000 * math.dist(d.pose.xy, gc), tuple(round(x) for x in d.mean_rgb),
                     tuple(round(x) for x in a.ref_rgb[n]), chroma_distance(d.mean_rgb, a.ref_rgb[n]))
            self.stats["missed_grasp" if closed else "left_jaws"] += 1
            self.held_target = None
            return []
        self.stats["held_frames"] += 1   # 爪の中に補ったフレームの数
        return [Detection(Pose2D(gc[0], gc[1], t.pose.yaw), t.size_m, t.mean_rgb, (0, 0, 0, 0))]

    def _reassociate(self, tracks: dict) -> None:
        """見えなくなった対象を、同じ色の名前のない追跡（今のフレームで見えている）に付け直す（弾かれた・隠れていた
        物体が離れた所で見つかったとき）。候補が複数なら、対象の最後の位置に近いもの。実機では VLM で確かめる（未実装）。"""
        a = self.assignment
        tol = self.fp.identity_chroma_tolerance
        c = self.c
        for n in a.episode.target_names:
            tid = a.track_of[n]
            if tid in tracks and tracks[tid].missed == 0:
                continue
            last = tracks[tid].pose.xy if tid in tracks else self.last_xy[n]
            ref_len = self.ref_len[n]
            named = set(a.track_of.values())
            cand = [t for t in tracks.values() if t.id not in named and t.missed == 0
                    and chroma_distance(t.mean_rgb, a.ref_rgb[n]) < tol
                    and math.dist(t.pose.xy, last) <= c.reassociate_max_m
                    and max(t.size_m[0], ref_len) / max(min(t.size_m[0], ref_len), 1e-6) <= c.reassociate_size_ratio]
            merged = not cand
            if merged:
                cand = self._merged(tracks, n, last, c.reassociate_max_m, named)
            if not cand:
                continue
            new = min(cand, key=lambda t: math.dist(t.pose.xy, last))
            log.info("対象 %s を追跡 %d から %d に付け直す（同じ色%s、%.0f mm 離れた所）", n, tid, new.id,
                     "の物とくっついた塊" if merged else "", 1000 * math.dist(new.pose.xy, last))
            if merged:
                self.stats["merged_reattach"] += 1
            self.stats["reassociate"] += 1
            a.track_of[n] = new.id
        for n in a.episode.target_names:    # 次に見失ったときの「最後の位置」
            if a.track_of[n] in tracks:
                self.last_xy[n] = tracks[a.track_of[n]].pose.xy

    def _merged(self, tracks: dict, n: str, last: tuple[float, float], max_m: float, named: set) -> list:
        """同じ色の物とくっついて大きく見える名前のない塊（merged_max_length_m、0 なら無し）: 見えていて、同じ色、last から
        max_m 以内、長辺が対象より長く merged_max_length_m 以下。普通の候補が無いときに付け直す（塊ごと押して入れる）。"""
        m = self.c.merged_max_length_m
        if m <= 0.0:
            return []
        tol = self.fp.identity_chroma_tolerance
        return [t for t in tracks.values() if t.id not in named and t.missed == 0
                and chroma_distance(t.mean_rgb, self.assignment.ref_rgb[n]) < tol
                and math.dist(t.pose.xy, last) <= max_m and self.ref_len[n] < t.size_m[0] <= m]

    # ---------- 仕切り直し ----------
    def begin_recover(self) -> None:
        """仕切り直しを始める（呼ぶ側が爪を開いて下がる前）。爪の中の対象は忘れる（開くので）。"""
        self.stats["recover"] += 1
        self.held_target = None

    def settle_after_recover(self, persist: bool, on_fail=None) -> list[str]:
        """仕切り直しで下がった後の付け直し（reattach_after_recover）。persist（止めずに続ける、実機）なら、付け直せない対象と、
        追跡はあるが今見えていない対象をあきらめ（Assignment.give_up）、その名前を返す（2026-10-23。
        前は見つからないまま仕切り直しを繰り返し、作業領域の外へ下がった）。persist でなければ今までどおり target_lost で止める。"""
        if not persist:
            self.reattach_after_recover(on_fail)
            return []
        gone: list[str] = []
        self.reattach_after_recover(on_fail, collect=gone)
        tracks = self.tracker.tracks
        a = self.assignment
        gone += [n for n in a.episode.target_names
                 if n not in gone and a.track_of[n] in tracks and tracks[a.track_of[n]].missed > 0]
        if gone:
            self.stats["given_up"] += len(gone)
            log.warning("仕切り直しても対象 %s が見つからない。あきらめて、ほかの対象を続ける", gone)
            a.give_up(gone)
        return gone

    def reattach_after_recover(self, on_fail=None, collect: list | None = None) -> None:
        """仕切り直しで下がった後: 追跡の無い対象を、同じ色・大きさの名前のない追跡（見えているもの）に、最後の位置から
        recover_search_m 以内で付け直す。見つからなければ on_fail()（呼ぶ側の調べ）を呼んでから target_lost で止める。"""
        a = self.assignment
        tracks = self.tracker.tracks
        c = self.c
        named = set(a.track_of.values())
        tol = self.fp.identity_chroma_tolerance
        for n in a.episode.target_names:
            if a.track_of[n] in tracks:
                continue
            last = self.last_xy[n]
            cand = [t for t in tracks.values() if t.id not in named and t.missed == 0
                    and chroma_distance(t.mean_rgb, a.ref_rgb[n]) < tol and math.dist(t.pose.xy, last) <= c.recover_search_m
                    and max(t.size_m[0], self.ref_len[n]) / max(min(t.size_m[0], self.ref_len[n]), 1e-6)
                    <= c.reassociate_size_ratio]
            merged = not cand
            if merged:
                cand = self._merged(tracks, n, last, c.recover_search_m, named)
                self.stats["merged_reattach"] += bool(cand)
            if not cand:
                if on_fail is not None:
                    on_fail()
                if collect is not None:   # settle_after_recover（persist）: 止めずに、あきらめる対象として返す
                    collect.append(n)
                    continue
                # 調べるための記録（2026-10-19）: 近くの追跡と、条件のどれで外れたか
                near = sorted(((math.dist(t.pose.xy, last), t) for t in tracks.values()
                               if math.dist(t.pose.xy, last) <= c.recover_search_m), key=lambda x: x[0])
                log.info("仕切り直しの候補が無い: 対象 %s（色 %s、長さ %.0f mm）の最後の位置から %.0f mm 以内の追跡: %s", n,
                         tuple(round(x) for x in a.ref_rgb[n]), 1000 * self.ref_len[n], 1000 * c.recover_search_m,
                         [(t.id, round(1000 * d), "名前あり" if t.id in named else "名前なし", f"missed {t.missed}",
                           round(chroma_distance(t.mean_rgb, a.ref_rgb[n]), 1), round(1000 * t.size_m[0]))
                          for d, t in near])
                raise FieldStop("target_lost", f"仕切り直しても対象 {n} が見つからない（{c.recover_search_m} m 以内）")
            new = min(cand, key=lambda t: math.dist(t.pose.xy, last))
            log.info("仕切り直し: 対象 %s を追跡 %d に付け直す（最後の位置から %.0f mm）", n, new.id,
                     1000 * math.dist(new.pose.xy, last))
            a.track_of[n] = new.id
            named.add(new.id)
