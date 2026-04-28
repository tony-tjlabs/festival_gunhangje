"""Festival_Gunhangje — 설정 및 상수 모듈.

진해 군항제 BLE 트래픽 분석 대시보드의 전역 상수, S-Ward 구역 정의,
RSSI 임계값 등을 관리한다.
"""
from __future__ import annotations

# ── 위치 (Open-Meteo) ─────────────────────────────────────────────────────────
JINHAE_LAT = 35.1497
JINHAE_LON = 128.6847

# ── 축제 기간 ─────────────────────────────────────────────────────────────────
FESTIVAL_START = "2026-03-27"
FESTIVAL_END   = "2026-04-05"

# ── RSSI 필터 임계값 ──────────────────────────────────────────────────────────
# 야외 대규모 축제 환경: 실내 건설현장 대비 약간 완화
IOS_RSSI_THRESH     = -75   # dBm (iPhone 광범위 — 기본 필터)
ANDROID_RSSI_THRESH = -85   # dBm (Android 광범위 — 기본 필터)
# 강신호 임계값 — Hermes Android 보정 기준 (iOS/Android 모두 잘 잡히는 범위)
IOS_RSSI_TIGHT      = -65   # dBm (iPhone 강신호)
ANDROID_RSSI_TIGHT  = -75   # dBm (Android 강신호)
TYPE_IOS     = 1
TYPE_ANDROID = 10

# ── 시간 인덱스 ───────────────────────────────────────────────────────────────
# time_index: 10초 단위 (1 = 00:00:10, 8640 = 24:00:00)
TI_PER_HOUR = 360   # 3600초 / 10초
TI_PER_DAY  = 8640

# ── S-Ward 구역 정의 ──────────────────────────────────────────────────────────
# Ground.png 좌표(3319×6599) 기반 군집 분석 결과
# Y가 작을수록 지도 상단(북쪽), X가 클수록 동쪽
# 속천항 와드(0,0): 27041757, 27040823, 27064920, 27064290 → 속천항 구역

ZONE_SWARDS: dict[str, list[str]] = {
    "A구역": [   # 북부 (x≈1405, y=2186~3000)
        "27041786", "27045056", "27065020", "27041684", "27041711",
        "27064668", "27041780", "27065019", "27064136", "27046091",
        "27065344", "27064710", "27064737", "27064974", "27064755",
    ],
    "B구역": [   # 중부 (x=1405~1611, y=3372~4090)
        "27041847", "27043194", "27042210", "27064845", "27064532",
        "27043099", "27064968", "27046875", "27041169", "27042767",
        "27065367",
    ],
    "C구역": [   # 남서부 (y>4090, x<1450)
        "27064841", "27041857", "27041299", "27064296",
        "27065446", "27041770", "27041745", "27042913",
    ],
    "D구역": [   # 남중부 (y>4090, 1450≤x≤1620)
        "27041737", "27043231", "27065013", "27043299",
        "27041753", "27047549", "27064481", "27064137",
        "27065480", "27064423",
    ],
    "E구역": [   # 남동부 (y>4090, x>1620)
        "27064660", "27044628", "27042538", "27065056",
        "27042283", "27064949", "27042956", "27065036",
        "27065448", "27065143", "27045314", "27041744",
    ],
    "속천항": [  # 속천항 (좌표 미등록 — swards.csv x=0,y=0)
        "27041757", "27040823", "27064920", "27064290",
    ],
}

ZONE_COLORS: dict[str, str] = {
    "A구역": "#4e79a7",
    "B구역": "#59a14f",
    "C구역": "#f28e2b",
    "D구역": "#e15759",
    "E구역": "#76b7b2",
    "속천항": "#9c755f",
    "미분류": "#b07aa1",
}

ZONE_LABELS: dict[str, str] = {
    "A구역": "A구역 (북부)",
    "B구역": "B구역 (중부)",
    "C구역": "C구역 (남서)",
    "D구역": "D구역 (남중)",
    "E구역": "E구역 (남동)",
    "속천항": "속천항",
    "미분류": "미분류",
}

# sward_name(str) → zone(str) 역매핑 (빠른 lookup)
SWARD_TO_ZONE: dict[str, str] = {
    sw: zone
    for zone, swards in ZONE_SWARDS.items()
    for sw in swards
}

ALL_ZONES = list(ZONE_SWARDS.keys())

# ── 날씨 ─────────────────────────────────────────────────────────────────────
WEATHER_EMOJI: dict[str, str] = {
    "Sunny": "☀️",
    "Rain":  "🌧️",
    "Snow":  "❄️",
    "Cloud": "☁️",
}
WEATHER_COLORS: dict[str, str] = {
    "Sunny": "#f0c040",
    "Rain":  "#4a90d9",
    "Snow":  "#a0d8ef",
    "Cloud": "#aaaaaa",
}

# ── 요일 ─────────────────────────────────────────────────────────────────────
DOW_KO = ["월", "화", "수", "목", "금", "토", "일"]
WEEKEND_DAYS = {5, 6}   # Saturday=5, Sunday=6 (Python weekday())

# ── 전처리 ────────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 500_000
CACHE_DIR_NAME  = "cache"
CACHE_VERSION   = "v4"

# ── 5분 윈도우 / AST 상수 ──────────────────────────────────────────────────────
TI_PER_5MIN            = 30     # 30 × 10초 = 5분
ANDROID_MAC_CORRECTION = 3.5   # Android UDC 보정 제수 (MAC 랜덤화 주기 보정)

# ── 캐시 파일명 ───────────────────────────────────────────────────────────────
CACHE_DAILY        = f"daily_summary_{CACHE_VERSION}.parquet"
CACHE_HOURLY       = f"hourly_summary_{CACHE_VERSION}.parquet"
CACHE_ZONE_HOURLY  = f"zone_hourly_{CACHE_VERSION}.parquet"
CACHE_INFLOW       = f"inflow_outflow_{CACHE_VERSION}.parquet"
CACHE_FINE_5MIN    = f"fine_5min_{CACHE_VERSION}.parquet"    # 5분 단위 보정 인원
CACHE_DAILY_AST    = f"daily_ast_{CACHE_VERSION}.parquet"    # 일별 AST
CACHE_HOURLY_AST   = f"hourly_ast_{CACHE_VERSION}.parquet"   # 시간별 AST
CACHE_SWARD_HOURLY = f"sward_hourly_{CACHE_VERSION}.parquet"  # S-Ward 시간별 DC
CACHE_FLOW         = f"flow_transitions_{CACHE_VERSION}.parquet"

# ── Streamlit ─────────────────────────────────────────────────────────────────
APP_TITLE   = "진해 군항제 트래픽 분석 대시보드"
APP_ICON    = "🌸"
STREAMLIT_PORT = 8560
