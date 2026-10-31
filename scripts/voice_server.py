"""音声入力のサーバー（RTX 3090 の PC）。pixi の vlm 環境で実行する。

    ~/.pixi/bin/pixi run -e vlm python scripts/voice_server.py --host 127.0.0.1 --port 8765   # MacBook からは ssh のトンネルで使う

起動時に Whisper（config/voice.yaml の server.asr_model_id）と VLM（config/vlm.yaml）を 1 回だけ読み込み、
POST /transcribe（WAV → 書き起こし）と POST /instruction（文 → 検証済みの指示）を返す。
ログは dev_log/voice/server_<日時>.log にも書く。
"""
import argparse
import logging
import time
from pathlib import Path

import uvicorn

from toy_sweeper.config import load_all
from toy_sweeper.task_manager import load_vocabulary
from toy_sweeper.vlm import QwenVL
from toy_sweeper.vlm_config import load_vlm_cfg
from toy_sweeper.voice.config import load_voice_cfg
from toy_sweeper.voice.server import WhisperASR, create_app

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="待ち受けるアドレス（127.0.0.1 にして、MacBook からは ssh のトンネルで使う。LAN に開けるなら 0.0.0.0）")
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    log_dir = ROOT / "dev_log" / "voice"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_dir / f"server_{time.strftime('%Y%m%d_%H%M%S')}.log")])
    log = logging.getLogger("voice_server")
    vcfg, cfg, vlm_cfg = load_voice_cfg(), load_all(), load_vlm_cfg()
    t0 = time.perf_counter()
    asr = WhisperASR(vcfg.server, vcfg.language, vcfg.audio.sample_rate_hz)
    log.info("Whisper（%s）を読み込んだ（%.1f s）", vcfg.server.asr_model_id, time.perf_counter() - t0)
    t0 = time.perf_counter()
    vlm = QwenVL(vlm_cfg)
    log.info("VLM（%s）を読み込んだ（%.1f s）", vlm_cfg.model_id, time.perf_counter() - t0)
    app = create_app(asr, vlm, vcfg, cfg, load_vocabulary(), vlm_cfg.max_new_tokens_instruction, vlm_cfg)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
