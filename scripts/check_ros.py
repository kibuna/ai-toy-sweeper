"""ROS 2 のプロセス間 pub/sub を確認する（scripts/check_env.sh から呼ぶ）。

使い方（pixi 環境内）:
    python scripts/check_ros.py talker   # 12 秒間 /env_check に送信
    python scripts/check_ros.py listener # 10 秒以内に受信できなければ AssertionError
"""
import sys
import time

import rclpy
from std_msgs.msg import String

TOPIC = "/env_check"


def talker() -> None:
    node = rclpy.create_node("env_check_talker")
    pub = node.create_publisher(String, TOPIC, 10)
    t0 = time.time()
    while time.time() - t0 < 12:
        pub.publish(String(data="hello"))
        rclpy.spin_once(node, timeout_sec=0.2)


def listener() -> None:
    node = rclpy.create_node("env_check_listener")
    got: list[str] = []
    node.create_subscription(String, TOPIC, lambda m: got.append(m.data), 10)
    t0 = time.time()
    while not got and time.time() - t0 < 10:
        rclpy.spin_once(node, timeout_sec=0.1)
    assert got, f"他プロセスからの {TOPIC} を 10 秒以内に受信できなかった"
    print(f"OK: ROS 2 pub/sub ({time.time() - t0:.2f} s)")


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in ("talker", "listener"), __doc__
    rclpy.init()
    {"talker": talker, "listener": listener}[sys.argv[1]]()
