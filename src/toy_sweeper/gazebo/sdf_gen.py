"""config/sim.yaml から Gazebo の World SDF と ros_gz_bridge の引数を生成する。

寸法は 2D シミュレータと同じ設定から作るので、二重管理しない。
ロボットのモデル原点は車軸中点の床面上（2D の Pose2D と同じ点）。
"""
from __future__ import annotations

import math
from pathlib import Path

from ..config import AllCfg, SimCfg
from ..robot_model import SIDES, body_center_x, front_x, pivot_xy

WORLD_NAME = "toy_sweeper"
ROBOT_NAME = "cutebot"
JAW_JOINTS = ("jaw_left_joint", "jaw_right_joint")


OBJECT_KINDS = ("lego", "toy_car")


def object_names(cfg: AllCfg) -> tuple[str, ...]:
    """World に置く物体の枠。盤面の物体数の上限（task.max_objects、全 Stage）だけ用意する。名前は TaskSampler と同じ。"""
    from ..task import max_objects
    return tuple(f"obj{i}" for i in range(max_objects(cfg.task)))


def model_name(slot: str, kind: str) -> str:
    """枠ごとに LEGO とミニカーの両方のモデルを置き、使う方だけを作業領域へ動かす。"""
    return f"{slot}_{kind}"


def all_model_names(cfg: AllCfg) -> tuple[str, ...]:
    return tuple(model_name(s, k) for s in object_names(cfg) for k in OBJECT_KINDS)


def board_models(objects) -> dict[str, str]:
    """盤面の物体（ObjectSpec）→ World に置くモデル（枠 → 種類）。"""
    return {o.name: o.type.name.lower() for o in objects}


def world_models(cfg: AllCfg, only: dict[str, str] | None) -> list[tuple[int, str, int, str]]:
    """World に置く物体のモデル（枠の番号, 枠, 種類の番号, 種類）。only（枠 → 種類）を渡すと、その盤面の物体だけ
    （2026-10-18、使わない 30 個のモデルも計算し続けて、0.25 s を進めるのに 1.7 s かかった。5 枠では 0.76 s）。"""
    out = []
    for i, slot in enumerate(object_names(cfg)):
        for j, kind in enumerate(OBJECT_KINDS):
            if only is None or only.get(slot) == kind:
                out.append((i, slot, j, kind))
    if only is not None and len(out) != len(only):
        raise ValueError(f"World に置けない物体がある（枠 {object_names(cfg)}、種類 {OBJECT_KINDS}）: {only}")
    return out


CAMERA_TOPIC = "/overhead_camera/image"
# 物体ごとのマスクのカメラ（gazebo.segmentation_camera）。labels_map は画素ごとに (インスタンスの番号, ラベル) を持つ
SEG_TOPIC = "/overhead_camera/seg"
# 物体のモデルに付ける種類のラベル（0 は背景: 床・壁・ロボット）
SEG_LABELS = {"lego": 1, "toy_car": 2}


def camera_sdf(sim: SimCfg, rate_hz: float) -> str:
    """作業領域を撮るカメラ（姿勢は gazebo.camera_pose）。rate_hz は 1 Action に 1 枚になる周期（sim 時間）。"""
    gz = sim.gazebo
    w, h = gz.camera_image_px
    if len(gz.camera_pose) != 6:
        raise ValueError(f"gazebo.camera_pose は [x, y, z, roll, pitch, yaw]: {gz.camera_pose}")
    return f"""
  <model name="overhead_camera"><static>true</static>
    <pose>{" ".join(str(v) for v in gz.camera_pose)}</pose>
    <link name="link">
      <sensor name="overhead" type="camera"><always_on>true</always_on>
        <update_rate>{rate_hz}</update_rate><topic>{CAMERA_TOPIC}</topic>
        <camera><horizontal_fov>{gz.camera_hfov_rad}</horizontal_fov><image><width>{w}</width><height>{h}</height></image>
          <clip><near>0.05</near><far>5</far></clip></camera>
      </sensor>{_seg_sensor(gz, rate_hz) if gz.segmentation_camera else ""}
    </link>
  </model>"""


def _seg_sensor(gz, rate_hz: float) -> str:
    """RGB のカメラと同じ姿勢・画角の、物体ごとのマスクのカメラ（gz-sensors の segmentation、instance）。"""
    w, h = gz.camera_image_px
    return f"""
      <sensor name="overhead_seg" type="segmentation"><always_on>true</always_on>
        <update_rate>{rate_hz}</update_rate><topic>{SEG_TOPIC}</topic>
        <camera><segmentation_type>instance</segmentation_type>
          <horizontal_fov>{gz.camera_hfov_rad}</horizontal_fov><image><width>{w}</width><height>{h}</height></image>
          <clip><near>0.05</near><far>5</far></clip></camera>
      </sensor>"""


def _seg_label(sim: SimCfg, kind: str) -> str:
    if not sim.gazebo.segmentation_camera:
        return ""
    return f'<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{SEG_LABELS[kind]}</label></plugin>'


def jaw_cmd_topic(joint: str) -> str:
    return f"/model/{ROBOT_NAME}/joint/{joint}/cmd_pos"


def _box_inertia(m: float, x: float, y: float, z: float) -> str:
    return (f"<inertia><ixx>{m * (y * y + z * z) / 12}</ixx><iyy>{m * (x * x + z * z) / 12}</iyy>"
            f"<izz>{m * (x * x + y * y) / 12}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>")


def _cyl_inertia(m: float, r: float, h: float) -> str:
    # 軸が y 方向の円柱
    a = m * (3 * r * r + h * h) / 12
    return f"<inertia><ixx>{a}</ixx><iyy>{m * r * r / 2}</iyy><izz>{a}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>"


def _surface(mu: float) -> str:
    return f"<surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>"


def robot_sdf(sim: SimCfg) -> str:
    r, g, gz = sim.robot, sim.gripper, sim.gazebo
    bx, by, top = r.body_size_m
    wr = r.wheel_radius_m
    ch = top - r.chassis_bottom_z_m
    if ch <= 0:
        raise ValueError("chassis_bottom_z_m が車体高さ以上")
    base_mass = r.mass_kg - 2 * r.wheel_mass_kg - 2 * g.jaw_mass_kg
    if base_mass <= 0:
        raise ValueError("車輪と指の質量の和が車体質量を超えている")
    # base_link の原点は車軸中点（高さ wr）
    cz = (r.chassis_bottom_z_m + top) / 2 - wr
    # 衝突形状はスカート下端まで伸ばした箱（車体＋外周スカート）。見た目は車体のみ
    sh = top - r.skirt_bottom_z_m
    sz = (r.skirt_bottom_z_m + top) / 2 - wr
    cx = body_center_x(sim)
    fx = front_x(sim)
    caster_x = fx - r.caster_radius_m - 0.005
    caster_z = r.caster_radius_m - wr
    mu_ro = sim.contact.robot_object_friction
    L, jw, jh = g.jaw_length_m, g.jaw_width_m, g.jaw_height_m
    jz = g.jaw_bottom_z_m + jh / 2 - wr
    d0, e = g.jaw_beam_start_m, g.jaw_lateral_offset_m
    beam_len = L - d0
    arm_len = math.hypot(d0, e)

    jaws, controllers = [], []
    for name, joint, side in (("jaw_left", JAW_JOINTS[0], SIDES[0]), ("jaw_right", JAW_JOINTS[1], SIDES[1])):
        px, py = pivot_xy(sim, side)
        # リンク原点は爪の軸。関節角 0 で梁は前方に平行（閉）、正の角度で外向きに開く
        arm_pose = f"{d0 / 2} {side * e / 2} {jz} 0 0 {side * math.atan2(e, d0)}"
        beam_pose = f"{d0 + beam_len / 2} {side * e} {jz} 0 0 0"
        mat = "<material><ambient>0.1 0.1 0.4 1</ambient><diffuse>0.1 0.1 0.4 1</diffuse></material>"
        jaws.append(f"""
    <link name="{name}">
      <pose relative_to="base_link">{px} {py} 0 0 0 0</pose>
      <inertial><pose>{beam_pose}</pose><mass>{g.jaw_mass_kg}</mass>{_box_inertia(g.jaw_mass_kg, L, jw, jh)}</inertial>
      <collision name="arm"><pose>{arm_pose}</pose><geometry><box><size>{arm_len} {jw} {jh}</size></box></geometry>{_surface(mu_ro)}</collision>
      <collision name="beam"><pose>{beam_pose}</pose><geometry><box><size>{beam_len} {jw} {jh}</size></box></geometry>{_surface(mu_ro)}</collision>
      <visual name="arm"><pose>{arm_pose}</pose><geometry><box><size>{arm_len} {jw} {jh}</size></box></geometry>{mat}</visual>
      <visual name="beam"><pose>{beam_pose}</pose><geometry><box><size>{beam_len} {jw} {jh}</size></box></geometry>{mat}</visual>
    </link>
    <joint name="{joint}" type="revolute">
      <parent>base_link</parent><child>{name}</child>
      <axis><xyz>0 0 {side}</xyz><limit><lower>{g.jaw_closed_angle_rad - 0.05}</lower><upper>{g.jaw_open_angle_rad + 0.05}</upper></limit>
        <dynamics><damping>0.0001</damping></dynamics></axis>
    </joint>""")
        controllers.append(f"""
    <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
      <joint_name>{joint}</joint_name>
      <topic>{jaw_cmd_topic(joint)}</topic>
      <p_gain>{gz.jaw_p_gain}</p_gain><d_gain>{gz.jaw_d_gain}</d_gain><i_gain>0</i_gain>
      <cmd_max>{gz.jaw_max_torque_nm}</cmd_max><cmd_min>{-gz.jaw_max_torque_nm}</cmd_min>
    </plugin>""")

    wheels = []
    for name, side in (("left_wheel", 1.0), ("right_wheel", -1.0)):
        wheels.append(f"""
    <link name="{name}">
      <pose relative_to="base_link">0 {side * r.wheel_separation_m / 2} 0 {-math.pi / 2} 0 0</pose>
      <inertial><mass>{r.wheel_mass_kg}</mass>{_cyl_inertia(r.wheel_mass_kg, wr, r.wheel_width_m)}</inertial>
      <collision name="c"><geometry><cylinder><radius>{wr}</radius><length>{r.wheel_width_m}</length></cylinder></geometry>
        {_surface(r.wheel_floor_friction)}</collision>
      <visual name="v"><geometry><cylinder><radius>{wr}</radius><length>{r.wheel_width_m}</length></cylinder></geometry>
        <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material></visual>
    </link>
    <joint name="{name}_joint" type="revolute">
      <parent>base_link</parent><child>{name}</child>
      <axis><xyz expressed_in="base_link">0 1 0</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis>
    </joint>""")

    return f"""
  <model name="{ROBOT_NAME}">
    <pose>0 0 0 0 0 0</pose>
    <link name="base_link">
      <pose>0 0 {wr} 0 0 0</pose>
      <inertial><pose>{cx} 0 {cz} 0 0 0</pose><mass>{base_mass}</mass>{_box_inertia(base_mass, bx, by, ch)}</inertial>
      <collision name="chassis"><pose>{cx} 0 {sz} 0 0 0</pose><geometry><box><size>{bx} {by} {sh}</size></box></geometry>
        {_surface(mu_ro)}</collision>
      <collision name="caster"><pose>{caster_x} 0 {caster_z} 0 0 0</pose>
        <geometry><sphere><radius>{r.caster_radius_m}</radius></sphere></geometry>{_surface(0.0)}</collision>
      <visual name="chassis"><pose>{cx} 0 {cz} 0 0 0</pose><geometry><box><size>{bx} {by} {ch}</size></box></geometry>
        <material><ambient>0.25 0.35 0.8 1</ambient><diffuse>0.25 0.35 0.8 1</diffuse></material></visual>
    </link>{''.join(wheels)}{''.join(jaws)}
    <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
      <left_joint>left_wheel_joint</left_joint><right_joint>right_wheel_joint</right_joint>
      <wheel_separation>{r.wheel_separation_m}</wheel_separation><wheel_radius>{wr}</wheel_radius>
      <topic>/model/{ROBOT_NAME}/cmd_vel</topic>
    </plugin>{''.join(controllers)}
    <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">
      <publish_link_pose>false</publish_link_pose><publish_model_pose>true</publish_model_pose>
      <publish_nested_model_pose>false</publish_nested_model_pose><use_pose_vector_msg>false</use_pose_vector_msg>
      <update_frequency>{gz.pose_publish_hz}</update_frequency>
    </plugin>
  </model>"""


def _pose_publisher(sim: SimCfg) -> str:
    return f"""
    <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">
      <publish_link_pose>false</publish_link_pose><publish_model_pose>true</publish_model_pose>
      <use_pose_vector_msg>false</use_pose_vector_msg><update_frequency>{sim.gazebo.pose_publish_hz}</update_frequency>
    </plugin>"""


def _material(rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    return f"<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>"


def object_rgb(color: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """物体の色名 → Gazebo の色（0〜1）。2D の描画（render.object_rgb）と同じ色。None なら default（枠の既定の色）。"""
    if color is None:
        return default
    from ..render import object_rgb as rgb255
    return tuple(round(v / 255, 3) for v in rgb255(color))


def lego_sdf(sim: SimCfg, name: str, xy: tuple[float, float], color: str | None = None) -> str:
    """LEGO: 等方摩擦の箱。モデル原点は底面の中心（床の高さ）。"""
    o = sim.objects["lego"]
    sx, sy, sz = o.size_m
    if o.floor_friction_xy[0] != o.floor_friction_xy[1]:
        raise ValueError("Gazebo の LEGO は等方摩擦の箱。floor_friction_xy の 2 値を揃えること")
    return f"""
  <model name="{name}">
    <pose>{xy[0]} {xy[1]} 0 0 0 0</pose>
    <link name="link">
      <pose>0 0 {sz / 2} 0 0 0</pose>
      <inertial><mass>{o.mass_kg}</mass>{_box_inertia(o.mass_kg, sx, sy, sz)}</inertial>
      <collision name="c"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>{_surface(o.floor_friction_xy[0])}</collision>
      <visual name="v"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        {_material(object_rgb(color, (0.85, 0.15, 0.15)))}</visual>
    </link>{_pose_publisher(sim)}{_seg_label(sim, "lego")}
  </model>"""


def toy_car_sdf(sim: SimCfg, name: str, xy: tuple[float, float], color: str | None = None) -> str:
    """ミニカー: 車体と自由に回る 4 輪。モデル原点は床の高さの車体中心。車体の x が前後。"""
    o, gz = sim.objects["toy_car"], sim.gazebo
    sx, sy, sz = o.size_m
    wr, ww = gz.toy_car_wheel_radius_m, gz.toy_car_wheel_width_m
    body_bottom = wr * 0.6
    bh = sz - body_bottom
    body_mass = o.mass_kg - 4 * gz.toy_car_wheel_mass_kg
    if bh <= 0 or body_mass <= 0:
        raise ValueError("ミニカーの車輪寸法・質量が車体と矛盾している")
    ax = gz.toy_car_axle_x_frac * sx / 2
    wy = sy / 2 - ww / 2
    wheels = "".join(f"""
    <link name="wheel_{i}">
      <pose>{x} {y} {wr} {-math.pi / 2} 0 0</pose>
      <inertial><mass>{gz.toy_car_wheel_mass_kg}</mass>{_cyl_inertia(gz.toy_car_wheel_mass_kg, wr, ww)}</inertial>
      <collision name="c"><geometry><cylinder><radius>{wr}</radius><length>{ww}</length></cylinder></geometry>
        {_surface(o.floor_friction_xy[1])}</collision>
      <visual name="v"><geometry><cylinder><radius>{wr}</radius><length>{ww}</length></cylinder></geometry>
        <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material></visual>
    </link>
    <joint name="wheel_{i}_joint" type="revolute"><parent>body</parent><child>wheel_{i}</child>
      <axis><xyz expressed_in="__model__">0 1 0</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit>
        <dynamics><damping>{gz.toy_car_wheel_damping}</damping></dynamics></axis></joint>"""
                     for i, (x, y) in enumerate(((ax, wy), (ax, -wy), (-ax, wy), (-ax, -wy))))
    return f"""
  <model name="{name}">
    <pose>{xy[0]} {xy[1]} 0 0 0 0</pose>
    <link name="body">
      <pose>0 0 {body_bottom + bh / 2} 0 0 0</pose>
      <inertial><mass>{body_mass}</mass>{_box_inertia(body_mass, sx, sy, bh)}</inertial>
      <collision name="c"><geometry><box><size>{sx} {sy - 2 * ww} {bh}</size></box></geometry>{_surface(sim.contact.robot_object_friction)}</collision>
      <visual name="v"><geometry><box><size>{sx} {sy - 2 * ww} {bh}</size></box></geometry>
        {_material(object_rgb(color, (0.15, 0.3, 0.85)))}</visual>
    </link>{wheels}{_pose_publisher(sim)}{_seg_label(sim, "toy_car")}
  </model>"""


def walls_sdf(sim: SimCfg) -> str:
    """作業領域の外周の壁（静的な箱 4 枚）。内側の面が作業領域の境界に来る。"""
    w = sim.walls
    hx, hy = sim.work_area.size_m[0] / 2, sim.work_area.size_m[1] / 2
    t, h = w.thickness_m, w.height_m
    boxes = [((hx + t / 2, 0), (t, 2 * hy + 2 * t)), ((-hx - t / 2, 0), (t, 2 * hy + 2 * t)),
             ((0, hy + t / 2), (2 * hx, t)), ((0, -hy - t / 2), (2 * hx, t))]
    links = "".join(f"""
      <collision name="c{i}"><pose>{x} {y} {h / 2} 0 0 0</pose><geometry><box><size>{sx} {sy} {h}</size></box></geometry>
        {_surface(w.friction)}</collision>
      <visual name="v{i}"><pose>{x} {y} {h / 2} 0 0 0</pose><geometry><box><size>{sx} {sy} {h}</size></box></geometry>
        <material><ambient>0.45 0.35 0.25 1</ambient><diffuse>0.45 0.35 0.25 1</diffuse></material></visual>"""
                    for i, ((x, y), (sx, sy)) in enumerate(boxes))
    return f"""
  <model name="walls"><static>true</static><link name="link">{links}
  </link></model>"""


def parking_xy(sim: SimCfg, slot_index: int, kind_index: int) -> tuple[float, float]:
    px, py = sim.gazebo.parking_xy_m
    return (px + 0.15 * slot_index, py - 0.15 * kind_index)


def marker_visuals(sim: SimCfg, texture_dir: Path) -> str:
    """床の 4 隅の ArUco マーカー（config/markers.yaml の置き方。白い余白は印刷の切り取り線まで）。
    模様は texture_dir に PNG で書き、PBR の albedo_map で貼る。"""
    import cv2
    from ..markers import corner_marker_centers, load_markers_cfg, marker_image
    m = load_markers_cfg()
    s, mg = m.field.corner_marker_size_m, m.field.paper_margin_m
    side_px = 600   # 黒い四角 60 mm で 1 mm 10 px（カメラの画素よりずっと細かい）
    margin_px = round(side_px * mg / s)
    texture_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for mid, (x, y) in corner_marker_centers(m, sim, sim.walls.thickness_m).items():
        path = texture_dir / f"aruco_{mid}.png"
        cv2.imwrite(str(path), marker_image(m, mid, side_px, margin_px))
        w = s + 2 * mg
        out.append(f"""
      <visual name="aruco_{mid}"><pose>{x} {y} 0.0016 0 0 0</pose>
        <geometry><box><size>{w} {w} 0.0002</size></box></geometry>
        <material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse>
          <pbr><metal><albedo_map>{path.resolve()}</albedo_map><roughness>1</roughness><metalness>0</metalness></metal></pbr>
        </material></visual>""")
    return "".join(out)


def _rgb(v) -> str:
    """3 つの値を SDF の空白区切りに（設定の値をそのまま書く）。"""
    if len(v) != 3:
        raise ValueError(f"3 つの値が要る: {v}")
    return " ".join(str(float(x)) for x in v)


def world_sdf(cfg: AllCfg, camera: bool = False, colors: dict[str, str] | None = None,
              markers_dir: Path | None = None, camera_rate_hz: float | None = None,
              only: dict[str, str] | None = None) -> str:
    """colors: 枠（obj0〜）→ 色名（盤面の色に合わせる。無い枠は既定の色）。markers_dir: 床の ArUco の模様の置き場
    （None ならマーカーを描かない）。"""
    sim = cfg.sim
    g = sim.gazebo
    ax, ay = sim.work_area.size_m
    park = sim.gazebo.parking_xy_m
    colors = colors or {}
    if unknown := set(colors) - set(object_names(cfg)):
        raise ValueError(f"Gazebo の World に無い枠の色: {sorted(unknown)}")
    objects = "".join(
        (lego_sdf if kind == "lego" else toy_car_sdf)(sim, model_name(slot, kind), parking_xy(sim, i, j), colors.get(slot))
        for i, slot, j, kind in world_models(cfg, only))
    goals = "".join(f"""
      <visual name="goal_{k}"><pose>{z.center_m[0]} {z.center_m[1]} 0.0011 0 0 0</pose>
        <geometry><box><size>{z.size_m[0]} {z.size_m[1]} 0.001</size></box></geometry>
        <material><ambient>0.6 0.85 0.6 1</ambient><diffuse>0.6 0.85 0.6 1</diffuse></material></visual>"""
                    for k, z in sim.goal_zones.items())
    return f"""<?xml version="1.0"?>
<!-- 自動生成: toy_sweeper.gazebo.sdf_gen（config/sim.yaml から）。手で編集しない -->
<sdf version="1.9">
<world name="{WORLD_NAME}">
  <physics name="default" type="dart"><max_step_size>{sim.gazebo.physics_step_s}</max_step_size>
    <real_time_factor>0</real_time_factor></physics>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  {'<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>' if camera else ''}
  <gravity>0 0 -9.81</gravity>
  <scene><ambient>{_rgb(g.scene_ambient)} 1</ambient></scene>
  <light type="directional" name="sun"><cast_shadows>{'true' if g.light_cast_shadows else 'false'}</cast_shadows><pose>0 0 10 0 0 0</pose>
    <diffuse>{_rgb(g.light_diffuse)} 1</diffuse><direction>{_rgb(g.light_direction)}</direction></light>
  <model name="floor"><static>true</static>
    <link name="link">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>10 10</size></plane></geometry>
        {_surface(sim.gazebo.floor_friction)}</collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>10 10</size></plane></geometry>
        <material><ambient>{_rgb(g.floor_rgb)} 1</ambient><diffuse>{_rgb(g.floor_rgb)} 1</diffuse></material></visual>
      <visual name="work_area"><pose>0 0 0.0005 0 0 0</pose><geometry><box><size>{ax} {ay} 0.001</size></box></geometry>
        <material><ambient>{_rgb(g.work_area_rgb)} 1</ambient><diffuse>{_rgb(g.work_area_rgb)} 1</diffuse></material></visual>
{goals}{marker_visuals(sim, markers_dir) if markers_dir is not None else ""}
    </link>
  </model>
{walls_sdf(sim) if sim.walls.enabled else ""}
{robot_sdf(sim)}
{objects}
{camera_sdf(sim, camera_rate_hz or 1.0 / cfg.task.control.action_duration_s) if camera else ""}
</world>
</sdf>
"""


ECHO_PREFIX = "/gz_echo"


def cmd_topics() -> list[tuple[str, str, str]]:
    """ROS → Gazebo の指令トピック（名前, ROS 型, gz 型）。"""
    return [(f"/model/{ROBOT_NAME}/cmd_vel", "geometry_msgs/msg/Twist", "gz.msgs.Twist"),
            *((jaw_cmd_topic(j), "std_msgs/msg/Float64", "gz.msgs.Double") for j in JAW_JOINTS)]


def echo_bridge_args() -> tuple[list[str], list[tuple[str, str]]]:
    """指令が Gazebo 側に出たことを確かめるための折り返し（Gazebo → ROS、ECHO_PREFIX 付きの名前へ remap）。

    2026-10-16: 指令を送ってから一定時間待つだけの方式では、CPU が混むと最初の数 ms が古い指令のまま進み、
    test_gazebo.py が 5 回に 1 回失敗した。折り返しで同じ値が戻るのを待ってから sim 時間を進める。
    """
    args = [f"{t}@{r}[{g}" for t, r, g in cmd_topics()]
    remaps = [(t, ECHO_PREFIX + t) for t, _, _ in cmd_topics()]
    return args, remaps


def bridge_args(cfg: AllCfg, camera: bool = False) -> list[str]:
    """ros_gz_bridge parameter_bridge のコマンドライン引数（topic@ROS型[]@gz型）。"""
    w = WORLD_NAME
    args = [
        f"/model/{ROBOT_NAME}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        f"/model/{ROBOT_NAME}/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose",
        *(f"/model/{n}/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose" for n in all_model_names(cfg)),
        *(f"{jaw_cmd_topic(j)}@std_msgs/msg/Float64]gz.msgs.Double" for j in JAW_JOINTS),
        f"/world/{w}/control@ros_gz_interfaces/srv/ControlWorld",
        f"/world/{w}/set_pose@ros_gz_interfaces/srv/SetEntityPose",
        *([f"{CAMERA_TOPIC}@sensor_msgs/msg/Image[gz.msgs.Image"] if camera else []),
        *([f"{SEG_TOPIC}/labels_map@sensor_msgs/msg/Image[gz.msgs.Image"]
          if camera and cfg.sim.gazebo.segmentation_camera else []),
    ]
    return args


def write_world(cfg: AllCfg, out_dir: Path, camera: bool = False, colors: dict[str, str] | None = None,
                markers: bool = False, camera_rate_hz: float | None = None, only: dict[str, str] | None = None) -> Path:
    """camera_rate_hz: カメラの周期（sim 時間）。None なら 1 Action に 1 枚（動画の記録用）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{WORLD_NAME}{'_camera' if camera else ''}{'_markers' if markers else ''}{'_colors' if colors else ''}.sdf"
    path.write_text(world_sdf(cfg, camera, colors, out_dir / "markers" if markers else None, camera_rate_hz, only))
    return path
