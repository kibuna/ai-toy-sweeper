"""Gazebo を ROS 2 経由で動かす物理バックエンド（ToySweeperEnv の Backend を実装）。

制御の経路は実機と同じ形にする: Policy → Action → /model/cutebot/cmd_vel (geometry_msgs/Twist, ROS 2)
→ ros_gz_bridge → Gazebo DiffDrive。姿勢も ROS 2 の Topic で受け取る。

Gazebo は一時停止状態で起動し、ControlWorld の multi_step で sim 時間を Action 1 回分だけ進める
（同期実行）。進め終わったかは /clock と姿勢のタイムスタンプで確かめる。
pixi 環境（rclpy がある）でのみ import できる。
"""
from __future__ import annotations

import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from std_msgs.msg import Float64

from ..config import AllCfg
from ..geometry import Pose2D
from ..sim2d import ObjectSpec
from ..spaces import OBJECT_TYPE_NAMES, GripperState
from .sdf_gen import (ECHO_PREFIX, JAW_JOINTS, OBJECT_KINDS, ROBOT_NAME, WORLD_NAME, all_model_names, jaw_cmd_topic, model_name, world_models,
                      object_names, parking_xy)

# ROS の応答を待つ上限（実時間）。超えたら Gazebo かブリッジが止まっているので例外にする
_TIMEOUT_S = 20.0


def _yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def _ns(stamp) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class GazeboBackend:
    def __init__(self, all_cfg: AllCfg, node_name: str = "toy_sweeper_gazebo_backend", models: dict[str, str] | None = None):
        """models: World にある物体のモデル（枠 → 種類。running_simulation の board と同じ盤面）。None なら全部の枠と種類。"""
        self.cfg = all_cfg.sim
        cfg = self.cfg
        self.names = object_names(all_cfg)
        self.models = world_models(all_cfg, models)
        if not rclpy.ok():
            rclpy.init()
        self.node: Node = rclpy.create_node(node_name)
        n = self.node
        self.cmd_pub = n.create_publisher(Twist, f"/model/{ROBOT_NAME}/cmd_vel", 10)
        self.jaw_pubs = [n.create_publisher(Float64, jaw_cmd_topic(j), 10) for j in JAW_JOINTS]
        # 折り返し（echo_bridge_args）で、Gazebo 側に出た最新の指令を受け取る
        self.echo: dict[str, tuple] = {}
        cmd_topic = f"/model/{ROBOT_NAME}/cmd_vel"
        n.create_subscription(Twist, ECHO_PREFIX + cmd_topic,
                              lambda m: self.echo.__setitem__(cmd_topic, (m.linear.x, m.angular.z)), 10)
        for j in JAW_JOINTS:
            t = jaw_cmd_topic(j)
            n.create_subscription(Float64, ECHO_PREFIX + t, lambda m, t=t: self.echo.__setitem__(t, (m.data,)), 10)
        self.clock_ns = -1
        self.poses: dict[str, tuple[int, Pose2D]] = {}
        n.create_subscription(Clock, "/clock", self._on_clock, 10)
        for name in (ROBOT_NAME, *(model_name(s, k) for _, s, _, k in self.models)):
            n.create_subscription(PoseStamped, f"/model/{name}/pose", self._pose_cb(name), 10)
        self.control = n.create_client(ControlWorld, f"/world/{WORLD_NAME}/control")
        self.set_pose = n.create_client(SetEntityPose, f"/world/{WORLD_NAME}/set_pose")
        for cli in (self.control, self.set_pose):
            if not cli.wait_for_service(timeout_sec=_TIMEOUT_S):
                raise RuntimeError(f"サービス {cli.srv_name} がない。scripts/start_simulation.sh で Gazebo を起動すること")
        self.dt = cfg.gazebo.physics_step_s
        self._gripper = GripperState.UNKNOWN
        self._active: list[str] = []
        self._wait(lambda: self.clock_ns >= 0, "最初の /clock")

    # ---------- ROS コールバック ----------
    def _on_clock(self, msg: Clock) -> None:
        self.clock_ns = _ns(msg.clock)

    def _pose_cb(self, name: str):
        def cb(msg: PoseStamped) -> None:
            p = msg.pose
            self.poses[name] = (_ns(msg.header.stamp), Pose2D(p.position.x, p.position.y, _yaw(p.orientation)))
        return cb

    def _wait(self, cond, what: str) -> None:
        t0 = time.time()
        while not cond():
            if time.time() - t0 > _TIMEOUT_S:
                raise TimeoutError(f"Gazebo 待ちがタイムアウト: {what}")
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def _call(self, cli, req):
        fut = cli.call_async(req)
        self._wait(fut.done, cli.srv_name)
        res = fut.result()
        if res is None or not res.success:
            raise RuntimeError(f"{cli.srv_name} が失敗した: {req}")
        return res

    # ---------- 時間発展 ----------
    def _n_steps(self, duration: float) -> int:
        n = round(duration / self.dt)
        if n <= 0 or abs(n * self.dt - duration) > 1e-6:
            raise ValueError(f"duration {duration} s は physics_step_s {self.dt} の整数倍でない")
        return n

    def _step(self, duration: float) -> None:
        start = self.clock_ns
        target = start + round(duration * 1e9)
        req = ControlWorld.Request()
        req.world_control.pause = True
        req.world_control.multi_step = self._n_steps(duration)
        self._call(self.control, req)
        self._wait(lambda: self.clock_ns >= target, f"/clock が {target} ns に達する")
        # 姿勢の出力周期は pose_publish_hz なので、その周期以内の古さの姿勢を待つ
        period_ns = round(1e9 / self.cfg.gazebo.pose_publish_hz)
        names = [ROBOT_NAME, *self._active]
        self._wait(lambda: all(n in self.poses and self.poses[n][0] >= target - period_ns for n in names),
                   "step 後の姿勢")

    def _publish_until_echoed(self, pub, msg, topic: str, value: tuple) -> None:
        """Gazebo 側に同じ値が出たのを折り返しで確かめるまで送り続ける（一時停止中なので sim 時間は進まない）。"""
        t0 = time.time()
        while self.echo.get(topic) != value:
            if time.time() - t0 > _TIMEOUT_S:
                raise TimeoutError(f"指令 {topic}={value} が Gazebo に届かない")
            pub.publish(msg)
            for _ in range(5):
                rclpy.spin_once(self.node, timeout_sec=0.002)
                if self.echo.get(topic) == value:
                    break

    def _publish_cmd(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._publish_until_echoed(self.cmd_pub, msg, f"/model/{ROBOT_NAME}/cmd_vel", (msg.linear.x, msg.angular.z))

    # ---------- Backend API ----------
    def reset(self, robot: Pose2D, objects: list[ObjectSpec], gripper: GripperState) -> None:
        if gripper not in (GripperState.OPEN, GripperState.CLOSED):
            raise ValueError(f"初期グリッパー状態は OPEN か CLOSED: {gripper}")
        for o in objects:
            if o.name not in self.names:
                raise ValueError(f"World にない物体名: {o.name}（ある名前: {self.names}）")
        self._publish_cmd(0.0, 0.0)
        self._set_jaws(gripper == GripperState.CLOSED)
        self._teleport(ROBOT_NAME, robot, 0.0)
        used = {o.name: OBJECT_TYPE_NAMES[o.type] for o in objects}
        have = {(s, k) for _, s, _, k in self.models}
        if missing := [(n, k) for n, k in used.items() if (n, k) not in have]:
            raise ValueError(f"World にないモデル: {missing}（盤面を変えたら Gazebo を起動し直す）")
        for i, slot, j, kind in self.models:
            if used.get(slot) != kind:
                self._teleport(model_name(slot, kind), Pose2D(*parking_xy(self.cfg, i, j), 0.0), 0.0)
        for o in objects:
            self._teleport(model_name(o.name, OBJECT_TYPE_NAMES[o.type]), o.pose, 0.0)
        self._model_of = {o.name: model_name(o.name, OBJECT_TYPE_NAMES[o.type]) for o in objects}
        self._active = list(self._model_of.values())
        self.poses.clear()
        # 指の開閉と接地を落ち着かせる（グリッパー 1 回分）。この間ロボットは止まっている
        self._step(self.cfg.gripper.actuation_time_s)
        self._gripper = gripper

    def _teleport(self, name: str, pose: Pose2D, z: float) -> None:
        req = SetEntityPose.Request()
        req.entity.name = name
        req.entity.type = Entity.MODEL
        req.pose.position.x, req.pose.position.y, req.pose.position.z = float(pose.x), float(pose.y), float(z)
        req.pose.orientation.z = math.sin(pose.yaw / 2)
        req.pose.orientation.w = math.cos(pose.yaw / 2)
        self._call(self.set_pose, req)

    def _set_jaws(self, close: bool) -> None:
        """爪の目標角を一度に送る（初期化用。開閉の動作は actuate_gripper がサーボの速さで送る）。"""
        g = self.cfg.gripper
        self._send_jaw_angle(float(g.jaw_closed_angle_rad if close else g.jaw_open_angle_rad))

    def _send_jaw_angle(self, q: float) -> None:
        for pub, j in zip(self.jaw_pubs, JAW_JOINTS):
            self._publish_until_echoed(pub, Float64(data=q), jaw_cmd_topic(j), (q,))
        self._jaw_q = q

    def drive(self, linear: float, angular: float, duration: float) -> None:
        self._publish_cmd(linear, angular)
        self._step(duration)
        self._publish_cmd(0.0, 0.0)

    def actuate_gripper(self, close: bool) -> None:
        """爪をサーボの速さ（gripper.jaw_speed_radps）で目標角まで回し、残りの時間は止まって待つ（合計 actuation_time_s）。"""
        g = self.cfg.gripper
        self._publish_cmd(0.0, 0.0)
        target = float(g.jaw_closed_angle_rad if close else g.jaw_open_angle_rad)
        q0 = getattr(self, "_jaw_q", target)
        inc = 5 * self.dt                                   # 5 ms ごとに目標角を進める
        n = max(1, math.ceil(abs(target - q0) / g.jaw_speed_radps / inc))
        if n * inc > g.actuation_time_s:
            raise ValueError(f"爪が {abs(target - q0):.3f} rad 回るのに {n * inc:.3f} s かかり、actuation_time_s を超える")
        for k in range(1, n + 1):
            self._send_jaw_angle(q0 + (target - q0) * k / n)
            self._step(inc)
        rest = g.actuation_time_s - n * inc
        if rest > 1e-9:
            self._step(round(rest / self.dt) * self.dt)
        self._gripper = GripperState.CLOSED if close else GripperState.OPEN

    def robot_pose(self) -> Pose2D:
        return self.poses[ROBOT_NAME][1]

    def object_pose(self, name: str) -> Pose2D:
        if name not in self._model_of:
            raise KeyError(f"このエピソードで使っていない物体: {name}")
        return self.poses[self._model_of[name]][1]

    def gripper_state(self) -> GripperState:
        return self._gripper

    def close(self) -> None:
        self.node.destroy_node()
