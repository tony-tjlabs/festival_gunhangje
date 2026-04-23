"""Festival_Gunhangje — 배포용 — 캐시 로드 전용.

이 모듈은 사전에 생성된 Parquet 캐시를 로드하는 함수만 포함한다.
Raw CSV 파싱/빌드 로직(build_cache, _process_one_day 등)은 제외되었다.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import (
    CACHE_DAILY, CACHE_DAILY_AST, CACHE_DIR_NAME,
    CACHE_FINE_5MIN, CACHE_FLOW, CACHE_HOURLY, CACHE_HOURLY_AST,
    CACHE_INFLOW, CACHE_SWARD_HOURLY, CACHE_ZONE_HOURLY,
)

logger = logging.getLogger(__name__)


# ── 클라우드 배포용 스텁 ───────────────────────────────────────────────────────
# 실제 빌드 로직은 로컬 SandBox에서만 실행. 배포 환경에서는 호출되지 않음.

def build_cache(base_dir: Path, progress_callback=None) -> bool:
    """배포 환경에서는 사용 불가 (캐시 사전 생성 필요)."""
    raise NotImplementedError("Cloud mode: caches are pre-built locally.")


def discover_dates(base_dir: Path) -> list:
    """배포 환경에서는 Data/ 폴더 없음 → 빈 리스트 반환."""
    return []


# ── 경로 ────────────────────────────────────────────────────────────────────

def get_cache_dir(base: Path) -> Path:
    d = base / CACHE_DIR_NAME
    d.mkdir(exist_ok=True)
    return d


# ── 캐시 존재 확인 ────────────────────────────────────────────────────────────

def cache_exists(base: Path) -> bool:
    cache_dir = get_cache_dir(base)
    return all(
        (cache_dir / fname).exists()
        for fname in [
            CACHE_DAILY, CACHE_HOURLY, CACHE_ZONE_HOURLY, CACHE_INFLOW,
            CACHE_FINE_5MIN, CACHE_DAILY_AST, CACHE_HOURLY_AST,
            CACHE_SWARD_HOURLY, CACHE_FLOW,
        ]
    )


# ── 캐시 로드 ─────────────────────────────────────────────────────────────────

def load_cache(base: Path) -> dict[str, pd.DataFrame]:
    """9개 Parquet 캐시 로드 후 dict 반환."""
    cache_dir = get_cache_dir(base)
    return {
        "daily":         pd.read_parquet(cache_dir / CACHE_DAILY),
        "hourly":        pd.read_parquet(cache_dir / CACHE_HOURLY),
        "zone_hourly":   pd.read_parquet(cache_dir / CACHE_ZONE_HOURLY),
        "inflow":        pd.read_parquet(cache_dir / CACHE_INFLOW),
        "fine_5min":     pd.read_parquet(cache_dir / CACHE_FINE_5MIN),
        "daily_ast":     pd.read_parquet(cache_dir / CACHE_DAILY_AST),
        "hourly_ast":    pd.read_parquet(cache_dir / CACHE_HOURLY_AST),
        "sward_hourly":  pd.read_parquet(cache_dir / CACHE_SWARD_HOURLY),
        "flow":          pd.read_parquet(cache_dir / CACHE_FLOW),
    }


# ── 캐시 메타데이터 ───────────────────────────────────────────────────────────

def cache_info(base: Path) -> dict:
    """캐시 파일 크기 및 수정 시각 정보 반환."""
    cache_dir = get_cache_dir(base)
    info = {}
    for key, fname in [
        ("daily",         CACHE_DAILY),
        ("hourly",        CACHE_HOURLY),
        ("zone_hourly",   CACHE_ZONE_HOURLY),
        ("inflow",        CACHE_INFLOW),
        ("fine_5min",     CACHE_FINE_5MIN),
        ("daily_ast",     CACHE_DAILY_AST),
        ("hourly_ast",    CACHE_HOURLY_AST),
        ("sward_hourly",  CACHE_SWARD_HOURLY),
        ("flow",          CACHE_FLOW),
    ]:
        p = cache_dir / fname
        if p.exists():
            stat = p.stat()
            info[key] = {
                "size_mb": round(stat.st_size / 1_048_576, 2),
                "mtime":   pd.Timestamp(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
            }
    return info
