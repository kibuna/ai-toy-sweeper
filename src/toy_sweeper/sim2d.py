"""学習用の軽量 2D シミュレータ（俯瞰平面、pymunk）。

モデル:
- ロボット: dynamic 剛体（車体＋外周スカートの箱、回転する爪 2 本）。指令速度で動く制御用の kinematic 体と
  PivotJoint / GearJoint でつなぎ、車軸中点の速度と角速度を指令に合わせる（pymunk の俯瞰の戦車の定石）。
  力の上限はタイヤが滑り出す力 μ_wheel m g、トルクの上限はそれに車輪間隔の半分を掛けた値。
  これで壁に当たると止まり、物体を壁に押し付けても突き抜けない。横滑りも同じ上限で抑える（非ホロノミック）。
- 壁: 作業領域の外周（walls）。ロボットも物体も外には出ない。
- 物体: dynamic 剛体。床との摩擦は、毎刻みの速度更新で物体座標系の x / y 方向に別々のクーロン摩擦
  （μ_x g dt, μ_y g dt だけ速度を 0 へ近づける）を掛けて表す。ミニカーは前後 μ_x が小さく横 μ_y が大きい。
  回転の摩擦は μ_y m g r_eff（r_eff は一様接地の長方形の中心からの平均距離）。
- 物体は車体の下に潜れない（config.validate がスカートの高さで保証する）。
- 爪: 開閉の指令（set_gripper）は目標の角度を決めるだけで、角度は drive の物理の刻みごとに gripper.jaw_speed_radps
  で目標へ近づく（刻みの初め、space.step の前に爪の形を作り直す）。一瞬で置き換えると、爪が物体の中に現れる。挟み損ねの
  後に閉じた爪が LEGO に 16 mm めり込んで閉じ込め、その後の開閉でロボットが 1 歩で 0.6 rad 回った。
- 把持: 爪を閉じ始めたとき、爪の間の領域に物体中心があれば、その物体に触れる角度まで回し、着いた刻みで、その時点の
  物体の位置で触れる角度を決め直して止め、物体をロボットに固定する。着いたときに爪の間から出ていれば、つながずに閉じ切る。
  実機の保持力・脱落は未測定。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pymunk

from .config import SimCfg
from .geometry import Pose2D, box_corners
from .robot_model import SIDES, angle_touching, body_center_x, jaw_inner_half_gap, jaw_polygons
from .spaces import OBJECT_TYPE_NAMES, GripperState, ObjectType

GRAVITY = 9.81
_GRASP_GROUP = 1  # ロボットの形状と把持中の物体はこのグループにして、互いの衝突を切る


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    type: ObjectType
    pose: Pose2D
    color: str  # 描画と VLM 用画像のためだけの情報。物理と Observation には使わない


def effective_friction_radius(sx: float, sy: float, n: int = 64) -> float:
    """一様接地の長方形で、回転摩擦トルク = μ m g r_eff となる r_eff（中心からの平均距離）。"""
    xs = (np.arange(n) + 0.5) / n * sx - sx / 2
    ys = (np.arange(n) + 0.5) / n * sy - sy / 2
    gx, gy = np.meshgrid(xs, ys)
    return float(np.mean(np.hypot(gx, gy)))


def _toward_zero(v: float, dv: float) -> float:
    return 0.0 if abs(v) <= dv else v - math.copysign(dv, v)


def _floor_friction_velocity_func(mu_x: float, mu_y: float, r_eff: float, mass: float, moment: float):
    # 重力 0、外力・外トルクを掛けない（接触と拘束だけで動く）前提で、速度の積分（update_velocity）は何もしない。
    # 止まっている物体（速度・角速度とも 0）は摩擦も何もしないので、すぐ返す。盤面の物体はほとんど止まっていて、
    # 2026-10-17 の計測で env の時間の 65% がこの関数だった。軌跡は変わらない
    def f(body: pymunk.Body, gravity, damping, dt):
        vx, vy = body.velocity
        w = body.angular_velocity
        if vx == 0.0 and vy == 0.0 and w == 0.0:
            return
        c, s = math.cos(body.angle), math.sin(body.angle)
        lx, ly = c * vx + s * vy, -s * vx + c * vy
        lx = _toward_zero(lx, mu_x * GRAVITY * dt)
        ly = _toward_zero(ly, mu_y * GRAVITY * dt)
        body.velocity = (c * lx - s * ly, s * lx + c * ly)
        body.angular_velocity = _toward_zero(w, mu_y * mass * GRAVITY * r_eff / moment * dt)
    return f


class Sim2D:
    def __init__(self, cfg: SimCfg):
        self.cfg = cfg
        self.dt = cfg.sim2d.physics_dt_s
        self.rng: np.random.Generator | None = None   # 挟み損ね（v21）の乱数。ToySweeperEnv.reset が渡す
        # 車輪の速さが指令に追いつく時定数（v36、旋回の惰性）。0 なら指令どおり。ToySweeperEnv.reset が引いて入れる
        self.motor_lag_s = 0.0
        self._wheel_v = (0.0, 0.0)                    # 今の (前進 m/s, 旋回 rad/s)。motor_lag_s > 0 のとき指令に追いつく
        self._space: pymunk.Space | None = None
        self._robot: pymunk.Body | None = None
        self._control: pymunk.Body | None = None
        self._jaws: list[pymunk.Shape] = []
        self._jaw_angle = cfg.gripper.jaw_open_angle_rad
        self._jaw_target = self._jaw_angle            # 爪が向かっている角度
        self._closing_on: str | None = None           # 閉じながら向かっている対象（着いたらつなぐ）
        self._objects: dict[str, tuple[pymunk.Body, pymunk.Shape, ObjectSpec]] = {}
        self._gripper = GripperState.UNKNOWN
        self._grasp: tuple[str, list[pymunk.Constraint]] | None = None
        self.time_s = 0.0

    # ---------- 構築 ----------
    def reset(self, robot: Pose2D, objects: list[ObjectSpec], gripper: GripperState) -> None:
        if gripper not in (GripperState.OPEN, GripperState.CLOSED):
            raise ValueError(f"初期グリッパー状態は OPEN か CLOSED: {gripper}")
        names = [o.name for o in objects]
        if len(set(names)) != len(names):
            raise ValueError(f"物体名が重複している: {names}")
        c = self.cfg
        self._wheel_v = (0.0, 0.0)
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
        space.iterations = c.sim2d.solver_iterations
        space.collision_slop = c.sim2d.collision_slop_m
        self._space = space

        bx, by = c.robot.body_size_m[0], c.robot.body_size_m[1]
        cx = body_center_x(c)
        m = c.robot.mass_kg
        body = pymunk.Body(m, pymunk.moment_for_box(m, (bx, by)) + m * cx * cx)
        body.center_of_gravity = (cx, 0.0)
        body.angle = robot.yaw     # 重心がずれているので、向きを先に決めてから原点（車軸中点）の位置を置く
        body.position = robot.xy
        control = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        control.position = robot.xy
        f_max = c.robot.wheel_floor_friction * m * GRAVITY
        pivot = pymunk.PivotJoint(control, body, (0, 0), (0, 0))
        pivot.max_bias = 0.0
        pivot.max_force = f_max
        gear = pymunk.GearJoint(control, body, 0.0, 1.0)
        gear.max_bias = 0.0
        gear.max_force = f_max * c.robot.wheel_separation_m / 2
        space.add(control, pivot, gear)
        self._control = control
        if self.cfg.walls.enabled:
            self._add_walls(space)
        chassis = pymunk.Poly(body, [(cx - bx / 2, -by / 2), (cx + bx / 2, -by / 2),
                                     (cx + bx / 2, by / 2), (cx - bx / 2, by / 2)])
        chassis.friction = c.contact.robot_object_friction
        chassis.elasticity = 0.0
        chassis.filter = pymunk.ShapeFilter(group=_GRASP_GROUP)
        space.add(body, chassis)
        self._robot = body
        self._chassis = chassis
        self._touched: set[str] = set()   # 直前の drive の間に車体の形（アームは除く）に触れた物体（v19）
        self._jaws = []
        self._grasp = None

        self._objects = {}
        for spec in objects:
            ocfg = c.objects[OBJECT_TYPE_NAMES[spec.type]]
            sx, sy = ocfg.size_m[0], ocfg.size_m[1]
            moment = pymunk.moment_for_box(ocfg.mass_kg, (sx, sy))
            ob = pymunk.Body(ocfg.mass_kg, moment)
            ob.position = spec.pose.xy
            ob.angle = spec.pose.yaw
            ob.velocity_func = _floor_friction_velocity_func(*ocfg.floor_friction_xy, effective_friction_radius(sx, sy),
                                                             ocfg.mass_kg, moment)
            shape = pymunk.Poly.create_box(ob, (sx, sy))
            # Chipmunk は接触摩擦を積で合成する。物体側 1.0 にして、ロボットとの摩擦を
            # contact.robot_object_friction そのものにする。物体どうしは 1.0（仮、未検証）
            shape.friction = 1.0
            shape.elasticity = 0.0
            space.add(ob, shape)
            self._objects[spec.name] = (ob, shape, spec)
        self._name_of = {shape: name for name, (_, shape, _) in self._objects.items()}

        g = c.gripper
        self._set_jaws(g.jaw_open_angle_rad if gripper == GripperState.OPEN else g.jaw_closed_angle_rad)
        self._jaw_target = self._jaw_angle
        self._closing_on = None
        self._gripper = gripper
        self._stalled = False
        self.time_s = 0.0

    def _add_walls(self, space: pymunk.Space) -> None:
        w = self.cfg.walls
        hx, hy = self.cfg.work_area.size_m[0] / 2, self.cfg.work_area.size_m[1] / 2
        r = self.cfg.sim2d.wall_collision_thickness_m / 2  # すり抜け防止のため実物の壁より厚い（内側の面は同じ）
        # 線分の太さ r の内側の面が作業領域の境界に来るよう、r だけ外へずらす
        X, Y = hx + r, hy + r
        for a, b in (((-X, -Y), (X, -Y)), ((X, -Y), (X, Y)), ((X, Y), (-X, Y)), ((-X, Y), (-X, -Y))):
            seg = pymunk.Segment(space.static_body, a, b, r)
            seg.friction = w.friction
            seg.elasticity = 0.0
            space.add(seg)

    def _set_jaws(self, angle: float) -> None:
        assert self._space is not None and self._robot is not None
        for s in self._jaws:
            self._space.remove(s)
        self._jaws = []
        for side in SIDES:
            for poly in jaw_polygons(self.cfg, side, angle):
                s = pymunk.Poly(self._robot, poly)
                s.friction = self.cfg.contact.robot_object_friction
                s.elasticity = 0.0
                s.filter = pymunk.ShapeFilter(group=_GRASP_GROUP)
                self._space.add(s)
                self._jaws.append(s)
        self._jaw_angle = angle
        self._space.reindex_shapes_for_body(self._robot)

    # ---------- 時間発展 ----------
    def _n_substeps(self, duration: float) -> int:
        n = round(duration / self.dt)
        if n <= 0 or abs(n * self.dt - duration) > 1e-6:
            raise ValueError(f"duration {duration} s は physics_dt_s {self.dt} の整数倍でない")
        return n

    def drive(self, linear: float, angular: float, duration: float) -> None:
        """ロボットを (linear m/s, angular rad/s) の指令で duration 秒動かす。motor_lag_s = τ > 0 なら、車輪の速さは刻みごとに
        v += (v_cmd − v) × dt / τ で指令に追いつく（v36。左右の車輪の速さに掛けるのと線形なので同じ）。"""
        assert self._space is not None and self._robot is not None, "reset() が先"
        self._touched = set()
        tau = self.motor_lag_s
        for _ in range(self._n_substeps(duration)):
            self._move_jaws()
            if tau > 0:
                v, w = self._wheel_v
                self._wheel_v = (v + (linear - v) * self.dt / tau, w + (angular - w) * self.dt / tau)
            else:
                self._wheel_v = (linear, angular)
            v, w = self._wheel_v
            a = self._robot.angle
            self._control.velocity = (v * math.cos(a), v * math.sin(a))
            self._control.angular_velocity = w
            self._space.step(self.dt)
            self.time_s += self.dt
            self._robot.each_arbiter(self._note_chassis_contact)
        self._control.velocity = (0.0, 0.0)
        self._control.angular_velocity = 0.0

    def _jaw_depth(self) -> float:
        """今の爪の形が、物体・壁にめり込んでいる深さの最大（m。重なりが無ければ 0）。車体と挟んだ物体は同じ組なので数えない。
        爪が触れている物体が壁にめり込んでいれば（直前の刻みの接触）、その深さも入れる（壁に押し付けられて逃げ場の無い物体を、
        爪が少しずつ壁の中へ押し込まないように。2026-10-19、v27 s0 の例外）。"""
        d = 0.0
        for s in self._jaws:
            for info in self._space.shape_query(s):
                for p in info.contact_point_set.points:
                    if -p.distance > d:
                        d = -p.distance
                body = info.shape.body
                if body.body_type == pymunk.Body.DYNAMIC and body is not self._robot:
                    for w in self._wall_depths(body):
                        if w > d:
                            d = w
        return d

    @staticmethod
    def _wall_depths(body) -> list[float]:
        """物体と動かない形（壁）の接触のめり込みの深さ（直前の刻みの接触）。"""
        out = []
        def f(arb):
            a, b = arb.shapes
            if a.body.body_type == pymunk.Body.STATIC or b.body.body_type == pymunk.Body.STATIC:
                out.extend(-p.distance for p in arb.contact_point_set.points)
        body.each_arbiter(f)
        return out

    def _turn_jaws(self, angle: float) -> None:
        """爪を angle へ置き直す。sim2d.jaw_stall が有効なら、置き直した形が depth_m より深く、置き直す前より深くめり込むとき
        は戻す（その刻みは回さない。サーボの失速）。"""
        st = self.cfg.sim2d.jaw_stall
        if not st.enabled:
            self._set_jaws(angle)
            return
        old = self._jaw_angle
        before = self._jaw_depth()
        self._set_jaws(angle)
        after = self._jaw_depth()
        if after > st.depth_m and after > before:
            self._set_jaws(old)
            self._stalled = True

    def _move_jaws(self) -> None:
        """爪を 1 刻み分（jaw_speed_radps × dt、最後は目標ちょうど）目標の角度へ回す。space.step の前に呼ぶ。"""
        if self._jaw_angle == self._jaw_target and self._closing_on is None:
            return
        g = self.cfg.gripper
        step = g.jaw_speed_radps * self.dt
        d = self._jaw_target - self._jaw_angle
        if abs(d) > step:
            self._turn_jaws(self._jaw_angle + math.copysign(step, d))
            return
        # 着いた。対象に向かっていたなら、まだ爪の間にあればつなぐ（_attach がその時点の位置で角度を決め直す）
        name, self._closing_on = self._closing_on, None
        if name is not None and self._is_between(name, self._jaw_target):
            self._attach(name)
            self._jaw_target = self._jaw_angle
            return
        self._turn_jaws(self._jaw_target)
        if name is not None:
            self._jaw_target = g.jaw_closed_angle_rad   # 対象が爪の間から出た。つながずに閉じ切る

    def _note_chassis_contact(self, arb: pymunk.Arbiter) -> None:
        a, b = arb.shapes
        if a is self._chassis or b is self._chassis:
            name = self._name_of.get(b if a is self._chassis else a)
            if name is not None:
                self._touched.add(name)

    def chassis_contacts(self) -> set[str]:
        """直前の drive（actuate_gripper を含む）の間に、車体の形（アームは除く）に触れた物体の名前（v19）。"""
        return set(self._touched)

    def actuate_gripper(self, close: bool) -> None:
        """爪を開閉する。actuation_time_s の間ロボットは止まる。その間に爪が回り切らなければ例外。ただし sim2d.jaw_stall で
        ふさがれて止まった刻みがあれば例外にしない（その後の drive の刻みで、ふさがりが解ければ回り続ける）。"""
        self.set_gripper(close)
        self._stalled = False
        self.drive(0.0, 0.0, self.cfg.gripper.actuation_time_s)
        if not self._stalled and (self._jaw_angle != self._jaw_target or self._closing_on is not None):
            raise RuntimeError(f"爪が actuation_time_s {self.cfg.gripper.actuation_time_s} s の間に回り切らない"
                               f"（角度 {self._jaw_angle}、目標 {self._jaw_target}）。gripper.jaw_speed_radps を確かめる")

    def set_gripper(self, close: bool) -> None:
        """開閉の指令。目標の角度と把持の判定だけを決め、時間は進めない（角度は drive の刻みの中で回る）。

        閉じる: つかんでいる・対象へ閉じている途中なら何もしない。爪の間（今の角度で）に物体があれば、挟み損ねの判定を
        して、失敗なら弾いて閉じ切る。成功ならその物体に触れる角度へ回し、着いた刻みでつなぐ。なければ閉じ切る。
        開く: つないでいる物体を離し（向かっていた対象も取り消し）、開いた角度へ回す。
        実時間で動かすとき（toy_sweeper.field.sim.SimWorld）に、開閉の間ロボットを止めておく処理を外側で行うため。
        """
        assert self._space is not None and self._robot is not None, "reset() が先"
        g = self.cfg.gripper
        if close:
            if self._grasp is None and self._closing_on is None:
                target = self._object_between_jaws()
                if target is not None and self._grasp_fails(target):
                    self._kick(target)
                    target = None
                if target is not None:
                    self._closing_on = target
                    self._jaw_target = self._touching_angle(target)
                else:
                    self._jaw_target = g.jaw_closed_angle_rad
            self._gripper = GripperState.CLOSED
        else:
            self._detach()
            self._closing_on = None
            self._jaw_target = g.jaw_open_angle_rad
            self._gripper = GripperState.OPEN

    def _grasp_fails(self, name: str) -> bool:
        """挟み損ね（sim2d.grasp_fail、v21）。無効なら必ずつかむ（これまでどおり）。乱数は self.rng（env が渡す）。"""
        gf = self.cfg.sim2d.grasp_fail
        if not gf.enabled:
            return False
        if getattr(self, "rng", None) is None:
            raise RuntimeError("挟み損ねが有効なのに乱数（Sim2D.rng）が無い。env.reset が渡す")
        _, ly = self._local(name)
        p = gf.base_prob + (1.0 - gf.base_prob) * min(1.0, abs(ly) / gf.offset_tol_m) ** 2
        return bool(self.rng.random() < p)

    def _kick(self, name: str) -> None:
        """挟み損ねた物体を、斜め前（横のずれの側）に、床の摩擦で kick_m だけ滑る速さで弾く。閉じる爪にずれた物体が
        押し出されると前へ飛ぶ（Gazebo の動画）。横向きに弾くと閉じる爪に当たって止まる。"""
        gf = self.cfg.sim2d.grasp_fail
        ob, _, spec = self._objects[name]
        lx, ly = self._local(name)
        d = float(self.rng.uniform(*gf.kick_m))
        mu = self.cfg.objects[OBJECT_TYPE_NAMES[spec.type]].floor_friction_xy[0]
        speed = math.sqrt(2.0 * mu * GRAVITY * d)
        side = 1.0 if ly >= 0 else -1.0
        a = self._robot.angle + side * float(self.rng.uniform(0.2, 0.8))   # 前から 11〜46° 横へ
        ob.velocity = (speed * math.cos(a), speed * math.sin(a))
        ob.angular_velocity = float(self.rng.uniform(-10.0, 10.0))

    def _local(self, name: str) -> tuple[float, float]:
        ob = self._objects[name][0]
        rp = self.robot_pose()
        c, s = math.cos(rp.yaw), math.sin(rp.yaw)
        dx, dy = ob.position.x - rp.x, ob.position.y - rp.y
        return c * dx + s * dy, -s * dx + c * dy

    def _object_between_jaws(self) -> str | None:
        best, best_d = None, math.inf
        for name in self._objects:
            lx, ly = self._local(name)
            if abs(ly) < jaw_inner_half_gap(self.cfg, self._jaw_angle, lx) and abs(ly) < best_d:
                best, best_d = name, abs(ly)
        return best

    def _is_between(self, name: str, angle: float) -> bool:
        """物体の中心が、爪の角度 angle のときの爪の間の領域にあるか（_object_between_jaws と同じ判定）。"""
        lx, ly = self._local(name)
        return abs(ly) < jaw_inner_half_gap(self.cfg, angle, lx)

    def _touching_angle(self, name: str) -> float:
        """今の物体の位置で、梁の内側がその物体に触れる角度（閉じ切る角度より開いている場合。angle_touching）。"""
        spec = self._objects[name][2]
        rp = self.robot_pose()
        corners = box_corners(self.object_pose(name), self.cfg.objects[OBJECT_TYPE_NAMES[spec.type]].size_m[:2])
        c, s = math.cos(rp.yaw), math.sin(rp.yaw)
        ly = [-s * (x - rp.x) + c * (y - rp.y) for x, y in corners]
        lx, _ = self._local(name)
        half = (max(ly) - min(ly)) / 2
        return angle_touching(self.cfg, lx, half)

    def _attach(self, name: str) -> None:
        """爪をその時点の物体に触れる角度で止め、物体をロボットに固定する。"""
        ob, shape, spec = self._objects[name]
        self._set_jaws(self._touching_angle(name))
        shape.filter = pymunk.ShapeFilter(group=_GRASP_GROUP)
        local = self._robot.world_to_local(ob.position)
        pin = pymunk.PivotJoint(self._robot, ob, local, (0, 0))
        gear = pymunk.GearJoint(self._robot, ob, ob.angle - self._robot.angle, 1.0)
        # 保持力の上限。上限なしだと、挟んだ物体が他の物体を壁に押し付けたときに大きな力が出て弾き飛ばす
        hold = self.cfg.gripper.grip_holding_force_n
        sx, sy = self.cfg.objects[OBJECT_TYPE_NAMES[spec.type]].size_m[:2]
        pin.max_force = hold
        gear.max_force = hold * effective_friction_radius(sx, sy)
        self._space.add(pin, gear)
        self._grasp = (name, [pin, gear])

    def _detach(self) -> None:
        if self._grasp is None:
            return
        name, constraints = self._grasp
        for con in constraints:
            self._space.remove(con)
        self._objects[name][1].filter = pymunk.ShapeFilter()
        self._grasp = None

    # ---------- 状態 ----------
    def robot_pose(self) -> Pose2D:
        b = self._robot
        return Pose2D(float(b.position.x), float(b.position.y), float(b.angle))

    def object_pose(self, name: str) -> Pose2D:
        ob = self._objects[name][0]
        return Pose2D(float(ob.position.x), float(ob.position.y), float(ob.angle))

    def object_names(self) -> list[str]:
        return list(self._objects)

    def gripper_state(self) -> GripperState:
        """指令した開閉の状態（爪が回っている途中でも、最後の指令の OPEN / CLOSED）。"""
        return self._gripper

    def jaw_angle(self) -> float:
        """今の爪の角度（回っている途中ならその角度。形もこの角度）。"""
        return self._jaw_angle

    def grasped_object(self) -> str | None:
        """ロボットに固定している物体（閉じている途中は、着いてつなぐまで None）。"""
        return None if self._grasp is None else self._grasp[0]
