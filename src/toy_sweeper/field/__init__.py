"""現場側（MacBook）の非同期ランタイム。ROS を使わない素の Python asyncio。

- latest.py: 最新の値だけを持つ入れ物 Latest
- interfaces.py: 機器の約束（CameraSource、RobotLink、姿勢の推定、物体の同定）と指令・停止の型
- runtime.py: 検出・追跡、制御、安全監視、指令の門、全体の流れ（FieldRuntime）
- sim.py: 実時間シミュレーション（SimWorld、SimCamera、SimRobotLink）
- hardware.py: 実機（OpenCVCamera、BleRobotLink。未検証）
- rig.py: シミュレーションの機器一式とランタイムをつなぐ（SimRig）
- voice_bridge.py: 音声クライアント（toy_sweeper.voice）の指示・停止語をランタイムへ渡す
"""
