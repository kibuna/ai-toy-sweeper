"""Rust 版の学習環境（toy-sweeper-native）は Linux だけで入る（pyproject.toml の依存の印）。MacBook（現場側）では
test_native_*.py を集めない。Linux では集めるので、ビルドが壊れていれば import で落ちる（importorskip で黙って飛ばさない）。"""
import sys

collect_ignore_glob = [] if sys.platform == "linux" else ["test_native_*.py"]
