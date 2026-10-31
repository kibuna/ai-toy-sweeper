"""音声入力。

- MacBook（現場、CPU）: マイク → VAD → ウェイクワード・停止語の検出 → サーバーに送る → 確認の読み上げ（client.py）
- RTX 3090 の PC: Whisper による書き起こしと、VLM による指示の構造化を HTTP で返す（server.py）

このパッケージの __init__ は何も import しない（クライアントが GPU 用の依存を読まないように）。
"""
