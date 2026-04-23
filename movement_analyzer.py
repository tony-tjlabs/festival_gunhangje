"""Festival_Gunhangje — 배포용 — 캐시 로드 전용 (compute 함수 제외).

이 모듈은 사전에 생성된 Parquet 캐시를 로드하는 함수만 포함한다.
Raw CSV 계산 로직(compute_and_cache, _compute_mobility 등)은 제외되었다.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 캐시 버전 ──────────────────────────────────────────────────────────────────
MOVE_CACHE_V      = "v9"
MOBILITY_FNAME    = f"mobility_{MOVE_CACHE_V}.parquet"      # S-Ward × hour 이동성
DWELL_FNAME       = f"zone_dwell_{MOVE_CACHE_V}.parquet"    # zone별 체류시간 세션
MAC_MOBILITY_FNAME= f"mac_mobility_{MOVE_CACHE_V}.parquet"  # MAC × hour 이동성 (속도 분포용)
ZONE_CUM_FNAME    = f"zone_cumulative_{MOVE_CACHE_V}.parquet"  # 분단위 구역별 MAC수

# ── 파라미터 ──────────────────────────────────────────────────────────────────
WIN_MINUTES       = 5     # 이동성 집계 윈도우 (분). 1=10초×6, 2=10초×12, 5=10초×30
WIN_TI            = WIN_MINUTES * 6   # time_index 단위 윈도우 크기 (5분=30)


# ════════════════════════════════════════════════════════════════════════════
# 캐시 경로
# ════════════════════════════════════════════════════════════════════════════

def _movement_dir(base_dir: Path) -> Path:
    d = base_dir / "cache" / "movement"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mobility_cache_path(base_dir: Path, date_str: str) -> Path:
    return _movement_dir(base_dir) / f"{date_str}_{MOBILITY_FNAME}"


def dwell_cache_path(base_dir: Path, date_str: str) -> Path:
    return _movement_dir(base_dir) / f"{date_str}_{DWELL_FNAME}"


def mac_mobility_cache_path(base_dir: Path, date_str: str) -> Path:
    return _movement_dir(base_dir) / f"{date_str}_{MAC_MOBILITY_FNAME}"


def zone_cumulative_cache_path(base_dir: Path, date_str: str) -> Path:
    return _movement_dir(base_dir) / f"{date_str}_{ZONE_CUM_FNAME}"


def cache_exists(base_dir: Path, date_str: str) -> bool:
    return (
        mobility_cache_path(base_dir, date_str).exists()
        and dwell_cache_path(base_dir, date_str).exists()
        and mac_mobility_cache_path(base_dir, date_str).exists()
        and zone_cumulative_cache_path(base_dir, date_str).exists()
    )


def load_mobility_cache(base_dir: Path, date_str: str) -> pd.DataFrame:
    p = mobility_cache_path(base_dir, date_str)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def load_dwell_cache(base_dir: Path, date_str: str) -> pd.DataFrame:
    p = dwell_cache_path(base_dir, date_str)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def load_mac_mobility_cache(base_dir: Path, date_str: str) -> pd.DataFrame:
    p = mac_mobility_cache_path(base_dir, date_str)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def load_zone_cumulative_cache(base_dir: Path, date_str: str) -> pd.DataFrame:
    p = zone_cumulative_cache_path(base_dir, date_str)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# 속도 환산 유틸
# ════════════════════════════════════════════════════════════════════════════

def _mobility_to_speed(mob: "np.ndarray | float") -> "np.ndarray | float":
    """이동성 지수(0~100) → 추정 보행 속도(km/h).

    군항제 축제 환경 보정 (Weidmann 보행자 기본 다이어그램 기반):
    - 극혼잡 피크(0~20)  : 0.1~0.5 km/h  — 떠밀리는 수준
    - 혼잡(20~40)        : 0.5~1.2 km/h  — 매우 느린 보행
    - 보통(40~60)        : 1.2~2.5 km/h  — 느린 보행
    - 여유(60~80)        : 2.5~3.8 km/h  — 정상 보행
    - 자유(80~100)       : 3.8~5.0 km/h  — 빠른 보행
    """
    bp  = [0,   20,  40,  60,  80, 100]
    spd = [0.1, 0.5, 1.2, 2.5, 3.8, 5.0]
    return np.interp(mob, bp, spd)


def _speed_level(speed_kmh: float) -> tuple[str, str]:
    """추정 속도(km/h) → (혼잡 레벨 이름, 색상 hex)"""
    if speed_kmh < 0.5:
        return "극혼잡 · 거의 정지", "#cc2222"
    elif speed_kmh < 1.2:
        return "혼잡 · 매우 느린 보행", "#e86020"
    elif speed_kmh < 2.5:
        return "보통 · 느린 보행", "#e8c020"
    elif speed_kmh < 3.8:
        return "여유 · 정상 보행", "#60c840"
    else:
        return "자유 · 빠른 보행", "#4080e0"


# ════════════════════════════════════════════════════════════════════════════
# 전체 날짜 집계 헬퍼
# ════════════════════════════════════════════════════════════════════════════

def aggregate_mobility(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """여러 날짜의 이동성 DataFrame을 합산 집계."""
    if not frames:
        return pd.DataFrame(columns=["sward", "hour", "avg_mobility", "mac_count"])
    combined = pd.concat(frames, ignore_index=True)
    agg = (
        combined.groupby(["sward", "hour"])
        .apply(lambda g: pd.Series({
            "avg_mobility": np.average(g["avg_mobility"], weights=g["mac_count"]),
            "mac_count":    g["mac_count"].sum(),
        }))
        .reset_index()
    )
    # 재정규화 0~100
    mx = agg["avg_mobility"].max()
    if mx > 0:
        agg["avg_mobility"] = (agg["avg_mobility"] / mx * 100).round(1)
    return agg


def aggregate_dwell(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """여러 날짜의 체류시간 DataFrame을 합산."""
    if not frames:
        return pd.DataFrame(columns=["mac_address", "zone", "dwell_s", "hour_start"])
    return pd.concat(frames, ignore_index=True)
