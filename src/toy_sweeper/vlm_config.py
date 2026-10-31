"""config/vlm.yaml の読み込み（VLM 環境と学習環境の両方から使う）。"""
from __future__ import annotations

from dataclasses import dataclass

from .config import CONFIG_DIR, load_yaml, strict_dataclass_from_dict


@dataclass(frozen=True)
class VLMCfg:
    model_id: str
    dtype: str
    max_new_tokens_instruction: int
    max_new_tokens_object: int
    do_sample: bool
    crop_margin_px: int
    crop_size_px: int
    identify_mode: str
    batch_size: int
    grid_max: int
    max_new_tokens_grid: int


def load_vlm_cfg(config_dir=CONFIG_DIR) -> VLMCfg:
    return strict_dataclass_from_dict(VLMCfg, load_yaml("vlm.yaml", config_dir), "vlm")
