"""Festival_Gunhangje — 날씨 모듈.

Open-Meteo Archive API로 진해 군항제 기간 날씨 데이터를 조회한다.
API 키 불필요, 24시간 캐시.
"""
from __future__ import annotations

import json
import logging
import ssl
import urllib.parse
import urllib.request

import pandas as pd
import streamlit as st

from config import (
    JINHAE_LAT, JINHAE_LON,
    WEATHER_EMOJI, WEATHER_COLORS,
    DOW_KO,
)

logger = logging.getLogger(__name__)


def _ssl_ctx() -> ssl.SSLContext:
    """SSL 컨텍스트 생성 — certifi 우선, 없으면 시스템 기본값 (CERT_NONE 금지)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        # certifi 미설치 시 시스템 기본 CA 사용 (검증 비활성화 절대 금지)
        return ssl.create_default_context()


@st.cache_data(show_spinner=False, ttl=86_400)
def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Open-Meteo Archive API로 일별 날씨 조회 (진해).

    Returns
    -------
    DataFrame with columns:
        date, precipitation, snowfall, temp_max, temp_min,
        weather, weather_emoji, weather_color
    """
    empty = pd.DataFrame(columns=[
        "date", "precipitation", "snowfall", "temp_max", "temp_min",
        "weather", "weather_emoji", "weather_color",
    ])

    params = {
        "latitude":   JINHAE_LAT,
        "longitude":  JINHAE_LON,
        "start_date": start_date,
        "end_date":   end_date,
        "daily": ",".join([
            "precipitation_sum",
            "snowfall_sum",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_hours",
            "windspeed_10m_max",
        ]),
        "timezone": "Asia/Seoul",
    }

    for base_url in [
        "https://archive-api.open-meteo.com/v1/archive",
        "https://api.open-meteo.com/v1/forecast",
    ]:
        try:
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(url, timeout=10, context=_ssl_ctx()) as resp:
                data = json.loads(resp.read().decode())
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            if not dates:
                continue

            df = pd.DataFrame({
                "date":          dates,
                "precipitation": daily.get("precipitation_sum"),
                "snowfall":      daily.get("snowfall_sum"),
                "temp_max":      daily.get("temperature_2m_max"),
                "temp_min":      daily.get("temperature_2m_min"),
                "precip_hours":  daily.get("precipitation_hours"),
                "wind_max":      daily.get("windspeed_10m_max"),
            })

            def classify(row) -> str:
                if not pd.isna(row["snowfall"]) and row["snowfall"] > 0:
                    return "Snow"
                if not pd.isna(row["precipitation"]) and row["precipitation"] > 0:
                    return "Rain"
                return "Sunny"

            df["weather"]       = df.apply(classify, axis=1)
            df["weather_emoji"] = df["weather"].map(WEATHER_EMOJI).fillna("❓")
            df["weather_color"] = df["weather"].map(WEATHER_COLORS).fillna("#888888")

            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)
            return df

        except Exception as exc:
            logger.warning("날씨 API 실패 (%s): %s", base_url, exc)
            continue

    logger.error("날씨 데이터 조회 실패 (모든 엔드포인트)")
    return empty


def make_date_label(date_str: str, weather_df: pd.DataFrame) -> str:
    """
    날짜 문자열을 '03/27(금) ☀️ 15°C' 형식으로 포맷팅.
    weather_df가 비어 있으면 요일만 포함.
    """
    try:
        dt  = pd.Timestamp(date_str)
        dow = DOW_KO[dt.dayofweek]
        base = f"{dt.month:02d}/{dt.day:02d}({dow})"
    except Exception:
        return date_str

    if weather_df.empty:
        return base

    row = weather_df[weather_df["date"] == date_str]
    if row.empty:
        return base

    r = row.iloc[0]
    emoji = r.get("weather_emoji", "")
    t_max = r.get("temp_max", float("nan"))
    t_min = r.get("temp_min", float("nan"))

    if not pd.isna(t_max) and not pd.isna(t_min):
        return f"{base} {emoji} {t_min:.0f}~{t_max:.0f}°C"
    return f"{base} {emoji}"


def weather_summary_text(weather_df: pd.DataFrame) -> str:
    """날씨 분포 요약 문자열 (AI 프롬프트용)."""
    if weather_df.empty:
        return "날씨 데이터 없음"
    counts = weather_df["weather"].value_counts().to_dict()
    parts = []
    for w, cnt in counts.items():
        emoji = WEATHER_EMOJI.get(w, "")
        parts.append(f"{emoji}{w} {cnt}일")
    avg_max = weather_df["temp_max"].mean()
    avg_min = weather_df["temp_min"].mean()
    return (
        f"{', '.join(parts)} / "
        f"평균기온 {avg_min:.1f}~{avg_max:.1f}°C"
    )
