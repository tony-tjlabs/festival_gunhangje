"""Festival_Gunhangje — LLM 분석 모듈 v2.

Claude API를 통해 트래픽 패턴 인사이트를 생성한다.
비식별 데이터(MAC 주소 랜덤화 환경)이므로 최대한 많은 데이터를 컨텍스트에 포함하여
정밀한 분석을 제공한다.

.env 또는 환경변수에서 ANTHROPIC_API_KEY를 읽는다.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    FESTIVAL_START, FESTIVAL_END, DOW_KO,
    ZONE_LABELS, ALL_ZONES, ANDROID_MAC_CORRECTION,
    WEEKEND_DAYS,
)
from weather import weather_summary_text

logger = logging.getLogger(__name__)

# ── API 키 로드 ──────────────────────────────────────────────────────────────

def _load_api_key() -> str | None:
    # 1. 환경변수
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    # 2. .env 파일 (프로젝트 루트)
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    # 3. st.secrets (Streamlit Cloud)
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return None


def is_llm_ready() -> bool:
    try:
        import anthropic  # noqa: F401
        return _load_api_key() is not None
    except ImportError:
        return False


# ── 시스템 프롬프트 ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 대규모 야외 축제의 유동인구 및 트래픽 패턴을 분석하는 공간 AI 전문가입니다.

분석 대상: 진해 군항제 (대한민국 경남 창원시 진해구)
- 매년 3~4월 개최되는 한국 최대 규모 벚꽃 축제
- 전국에서 약 200~300만 명 방문, 10일간 개최
- 주요 장소: 여좌천(로망스 다리), 경화역(철길 벚꽃 터널), 중원로터리, 제황산공원, 해군사관학교 등
- BLE/WiFi S-Ward 센서 약 56개 설치, 방문객 모바일 기기 신호 수집
- 센서 분포: 균일하지 않음, 음영 지역 존재 → 절대 인원 추정보다 상대적 패턴 분석에 집중

구역 정의:
- A구역(북부): 경화역, 제황산공원 방면 (S-Ward 15개)
- B구역(중부): 중원로터리, 도심 (S-Ward 11개)
- C구역(남서): 여좌천, 로망스 다리 (S-Ward 8개)
- D구역(남중): 해군사관학교 방면 (S-Ward 10개)
- E구역(남동): 해안도로 방면 (S-Ward 12개)

핵심 지표 정의:
- Device Count (DC): 하루 동안 감지된 고유 MAC 주소 수 (실제 방문자의 근사치)
- AST (Accumulated Staying Time): 체류시간 합산 (signal_count × 10초). 방문 "질" 지표
- 유입: 하루 중 처음 감지된 기기 수 (방문 개시 시각 분포)
- 유출: 하루 중 마지막 감지된 기기 수 (방문 종료 시각 분포)
- MAC 랜덤화: 모바일 기기는 일정 주기로 MAC을 변경하므로 Device Count는 실제 방문자의 근사값임

인사이트 작성 원칙:
- 데이터에서 관찰된 사실을 수치와 함께 서술 (추측은 "가능성" 표현)
- 축제 운영·안전·방문객 경험 개선에 실용적인 제언 포함
- 한국어로 작성, 전문적이지만 이해하기 쉬운 문체 사용
- 구체적인 수치 비교와 퍼센트 변화율을 적극 활용
"""


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _festival_day_tag(date_str: str) -> str:
    """날짜 → 축제 일차 태그 (Day N / 사전 / 비축제)."""
    if not (FESTIVAL_START <= date_str <= FESTIVAL_END):
        return "사전"
    start = pd.Timestamp(FESTIVAL_START)
    delta = (pd.Timestamp(date_str) - start).days + 1
    return f"Day{delta}"


def _pct_change(new_val: float, old_val: float) -> str:
    """증감률 포맷 문자열 반환. old_val=0이면 'N/A'."""
    if old_val == 0:
        return "N/A"
    pct = (new_val - old_val) / old_val * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"




# ════════════════════════════════════════════════════════════════════════════
# _build_daily_context  (대폭 강화)
# ════════════════════════════════════════════════════════════════════════════

def _build_daily_context(
    date_str: str,
    daily_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    zone_hourly_df: pd.DataFrame,
    inflow_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    daily_ast_df: pd.DataFrame | None = None,
    hourly_ast_df: pd.DataFrame | None = None,
    fine_5min_df: pd.DataFrame | None = None,
) -> str:
    """단일 날짜 분석용 풍부한 컨텍스트 텍스트 구성."""
    dt  = pd.Timestamp(date_str)
    dow = DOW_KO[dt.dayofweek]
    is_weekend = dt.dayofweek in WEEKEND_DAYS
    festival_tag = _festival_day_tag(date_str)

    parts: list[str] = []
    parts.append(
        f"## 분석 대상: {date_str}({dow}) [{festival_tag}] "
        f"{'[주말]' if is_weekend else '[평일]'}"
    )

    # ── 날씨 ──────────────────────────────────────────────────────────────────
    weather_row = {}
    if not weather_df.empty:
        w = weather_df[weather_df["date"] == date_str]
        if not w.empty:
            r = w.iloc[0]
            weather_row = r.to_dict()
            emoji = r.get("weather_emoji", "")
            precip = r.get("precipitation", 0.0) or 0.0
            wind   = r.get("windspeed_max", r.get("wind_max", 0.0)) or 0.0
            parts.append(
                f"날씨: {emoji} {r.get('weather', 'N/A')} "
                f"| 기온 {r.get('temp_min', '?'):.0f}~{r.get('temp_max', '?'):.0f}°C "
                f"| 강수량 {precip:.1f}mm "
                f"| 바람 최대 {wind:.0f}km/h"
            )

    # ── 일별 요약 ─────────────────────────────────────────────────────────────
    row_d = daily_df[daily_df["date"] == date_str]
    if not row_d.empty:
        r = row_d.iloc[0]
        ios_udc     = int(r["ios_udc"])
        android_udc = int(r["android_udc"])
        dc          = int(r["udc"])
        total_ios   = ios_udc + android_udc
        ios_pct     = ios_udc / total_ios * 100 if total_ios > 0 else 0

        # 전일 비교
        daily_sorted = daily_df.sort_values("date").reset_index(drop=True)
        idx = daily_sorted[daily_sorted["date"] == date_str].index
        prev_dc   = 0
        prev_date = ""
        if len(idx) > 0 and idx[0] > 0:
            prev_row  = daily_sorted.iloc[idx[0] - 1]
            prev_dc   = int(prev_row["udc"])
            prev_date = str(prev_row["date"])

        parts.append("\n### 일별 요약")
        parts.append(
            f"- Device Count: {dc:,}명"
        )
        parts.append(
            f"- iOS: {ios_udc:,} | Android: {android_udc:,} "
            f"| iOS 비율: {ios_pct:.0f}:{100 - ios_pct:.0f}"
        )
        parts.append(
            f"- 총 레코드: {int(r['total_records']):,}건 "
            f"| 평균 RSSI: {r['avg_rssi']:.1f} dBm"
        )
        if prev_dc > 0:
            parts.append(
                f"- 전일({prev_date}) 대비 Device Count: {_pct_change(dc, prev_dc)} "
                f"({prev_dc:,} → {dc:,})"
            )

        # 같은 요일 과거 비교
        daily_df2 = daily_df.copy()
        daily_df2["dow_num"] = pd.to_datetime(daily_df2["date"]).dt.dayofweek
        same_dow = daily_df2[
            (daily_df2["dow_num"] == dt.dayofweek) & (daily_df2["date"] != date_str)
        ]
        if not same_dow.empty:
            avg_same_dow = same_dow["udc"].mean()
            parts.append(
                f"- 같은 요일({dow}) 평균 대비: {_pct_change(dc, avg_same_dow)} "
                f"(평균 {int(avg_same_dow):,})"
            )

        # 축제 전체 평균 대비
        festival_avg = daily_df[
            daily_df["date"].between(FESTIVAL_START, FESTIVAL_END)
        ]["udc"].mean()
        if festival_avg > 0 and FESTIVAL_START <= date_str <= FESTIVAL_END:
            parts.append(
                f"- 축제 기간 평균 대비: {_pct_change(dc, festival_avg)} "
                f"(평균 {int(festival_avg):,})"
            )

    # ── AST (일별) ────────────────────────────────────────────────────────────
    if daily_ast_df is not None and not daily_ast_df.empty:
        row_ast = daily_ast_df[daily_ast_df["date"] == date_str]
        if not row_ast.empty:
            ra = row_ast.iloc[0]
            total_ast_h = float(ra["ast_hours"])
            ios_ast_s   = float(ra.get("ios_ast_seconds", 0))
            and_ast_s   = float(ra.get("and_ast_seconds", 0))
            ios_ast_h   = ios_ast_s / 3600
            and_ast_h   = and_ast_s / 3600

            # 기기당 평균 체류시간 (분)
            udc_for_ast = row_d.iloc[0]["udc"] if not row_d.empty else 1
            ios_udc_a   = row_d.iloc[0]["ios_udc"] if not row_d.empty else 1
            and_udc_a   = row_d.iloc[0]["android_udc"] if not row_d.empty else 1
            avg_stay_min = (total_ast_h * 60) / max(udc_for_ast, 1)
            ios_stay_min = (ios_ast_h * 60) / max(ios_udc_a, 1)
            and_stay_min = (and_ast_h * 60) / max(and_udc_a, 1)

            parts.append("\n### 체류시간 (AST)")
            parts.append(
                f"- 총 AST: {total_ast_h:,.1f}시간 "
                f"(iOS: {ios_ast_h:,.1f}h | Android: {and_ast_h:,.1f}h)"
            )
            parts.append(
                f"- 기기당 평균 체류: {avg_stay_min:.0f}분/기기 "
                f"(iOS: {ios_stay_min:.0f}분 | Android: {and_stay_min:.0f}분)"
            )

    # ── 시간별 DC (전체 테이블) ──────────────────────────────────────────────
    row_h = hourly_df[hourly_df["date"] == date_str].sort_values("hour")
    if not row_h.empty:
        max_dc  = row_h["dc"].max()
        peak_h  = int(row_h.loc[row_h["dc"].idxmax(), "hour"])
        parts.append("\n### 시간별 Device Count")
        parts.append(f"[피크: {peak_h:02d}시 Device Count={int(max_dc):,}]")
        parts.append(f"{'시간':>4} | {'Device Count':>12} | {'iOS':>6} | {'Android':>7} | 막대그래프")
        parts.append("-" * 55)
        for _, hr in row_h.iterrows():
            h     = int(hr["hour"])
            dc_h  = int(hr["dc"])
            ios_d = int(hr.get("ios_dc", 0))
            and_d = int(hr.get("android_dc", 0))
            bar   = "█" * min(int(dc_h / max(max_dc / 20, 1)), 20)
            parts.append(f"{h:02d}시  | {dc_h:>12,} | {ios_d:>6,} | {and_d:>7,} | {bar}")

    # ── 시간별 AST ────────────────────────────────────────────────────────────
    if hourly_ast_df is not None and not hourly_ast_df.empty:
        row_hast = hourly_ast_df[hourly_ast_df["date"] == date_str].sort_values("hour")
        if not row_hast.empty:
            max_ast_min = row_hast["ast_minutes"].max()
            peak_hast   = int(row_hast.loc[row_hast["ast_minutes"].idxmax(), "hour"])
            parts.append(f"\n### 시간별 AST [피크: {peak_hast:02d}시]")
            parts.append(f"{'시간':>4} | {'AST(분)':>9} | 막대")
            parts.append("-" * 35)
            for _, har in row_hast.iterrows():
                h       = int(har["hour"])
                ast_min = float(har["ast_minutes"])
                bar     = "█" * min(int(ast_min / max(max_ast_min / 20, 1)), 20)
                parts.append(f"{h:02d}시  | {ast_min:>9,.0f} | {bar}")

    # ── 5분 단위 Device Count (피크 구간 + 시간별 최대값) ───────────────────
    if fine_5min_df is not None and not fine_5min_df.empty:
        row_f = fine_5min_df[fine_5min_df["date"] == date_str].copy()
        if not row_f.empty:
            max_fine = row_f["corrected_dc"].max()
            peak_fine_row = row_f.loc[row_f["corrected_dc"].idxmax()]
            ph = int(peak_fine_row["hour"])
            pm = int(peak_fine_row["minute"])
            parts.append(
                f"\n### 5분 단위 Device Count "
                f"[피크: {ph:02d}:{pm:02d} | DC={int(max_fine):,}]"
            )
            # 시간별 최대값 요약
            hour_max = row_f.groupby("hour")["corrected_dc"].max()
            parts.append(f"{'시간':>4} | {'5분 피크 DC':>12}")
            parts.append("-" * 25)
            for h, val in hour_max.items():
                parts.append(f"{int(h):02d}시  | {int(val):>12,}")

    # ── 유입/유출 ─────────────────────────────────────────────────────────────
    row_io = inflow_df[inflow_df["date"] == date_str].sort_values("hour")
    if not row_io.empty:
        total_in  = int(row_io["inflow"].sum())
        total_out = int(row_io["outflow"].sum())
        peak_in_r = row_io.loc[row_io["inflow"].idxmax()]
        peak_out_r = row_io.loc[row_io["outflow"].idxmax()]
        peak_in_h  = int(peak_in_r["hour"])
        peak_out_h = int(peak_out_r["hour"])

        # 시간대 구간 집계
        am_in   = int(row_io[row_io["hour"].between(6, 11)]["inflow"].sum())
        pm_in   = int(row_io[row_io["hour"].between(12, 17)]["inflow"].sum())
        eve_in  = int(row_io[row_io["hour"].between(18, 23)]["inflow"].sum())
        early_in = int(row_io[row_io["hour"].between(0, 5)]["inflow"].sum())

        # 누적 재실 피크
        cum_in  = row_io["inflow"].cumsum()
        cum_out = row_io["outflow"].cumsum()
        occupancy = cum_in - cum_out
        peak_occ_h = int(row_io.iloc[occupancy.idxmax()]["hour"])
        peak_occ   = int(occupancy.max())

        parts.append("\n### 유입/유출 패턴")
        parts.append(
            f"- 총 유입: {total_in:,} | 총 유출: {total_out:,}"
        )
        parts.append(
            f"- 최대 유입: {peak_in_h:02d}시 ({int(peak_in_r['inflow']):,}명) "
            f"| 최대 유출: {peak_out_h:02d}시 ({int(peak_out_r['outflow']):,}명)"
        )
        parts.append(
            f"- 새벽(00~05시): {early_in:,} | 오전(06~11시): {am_in:,} "
            f"| 오후(12~17시): {pm_in:,} | 저녁(18~23시): {eve_in:,}"
        )
        parts.append(
            f"- 누적 재실 피크: {peak_occ_h:02d}시 (추정 {peak_occ:,}명)"
        )
        parts.append(f"\n{'시간':>4} | {'유입':>7} | {'유출':>7} | {'순유입':>7}")
        parts.append("-" * 38)
        for _, ior in row_io.iterrows():
            h   = int(ior["hour"])
            inf = int(ior["inflow"])
            out = int(ior["outflow"])
            net = inf - out
            parts.append(
                f"{h:02d}시  | {inf:>7,} | {out:>7,} | "
                f"{('+' if net >= 0 else '')}{net:>6,}"
            )

    # ── 구역별 트래픽 (전일 비교 포함) ──────────────────────────────────────
    row_z = zone_hourly_df[zone_hourly_df["date"] == date_str]
    if not row_z.empty:
        zone_total = row_z.groupby("zone")["dc"].sum().sort_values(ascending=False)
        grand_dc   = zone_total.sum()

        # 전일 구역 데이터
        prev_zone_dc: dict[str, int] = {}
        if prev_date:
            prev_z = zone_hourly_df[zone_hourly_df["date"] == prev_date]
            if not prev_z.empty:
                prev_zone_dc = prev_z.groupby("zone")["dc"].sum().to_dict()

        parts.append("\n### 구역별 트래픽")
        parts.append(
            f"{'구역':^10} | {'총DC':>8} | {'비율':>6} | {'전일비교':>10}"
        )
        parts.append("-" * 48)
        for zone, dc in zone_total.items():
            label    = ZONE_LABELS.get(zone, zone)
            ratio    = dc / grand_dc * 100 if grand_dc > 0 else 0
            prev_dc  = prev_zone_dc.get(zone, 0)
            cmp_str  = _pct_change(dc, prev_dc) if prev_dc > 0 else "N/A"
            parts.append(
                f"{label:^10} | {int(dc):>8,} | {ratio:>5.1f}% | {cmp_str:>10}"
            )

        # 시간별 구역 DC 요약 (피크 시간대만)
        peak_hours = row_h.nlargest(6, "dc")["hour"].tolist() if not row_h.empty else []
        if peak_hours:
            parts.append("\n### 구역별 피크 시간대 DC")
            zone_pivot = (
                row_z[row_z["hour"].isin(peak_hours)]
                .groupby(["hour", "zone"])["dc"]
                .sum()
                .unstack(fill_value=0)
                .sort_index()
            )
            header = f"{'시간':>4} | " + " | ".join(f"{ZONE_LABELS.get(z, z):^10}" for z in zone_pivot.columns)
            parts.append(header)
            parts.append("-" * len(header))
            for h, zrow in zone_pivot.iterrows():
                row_str = " | ".join(f"{int(v):>10,}" for v in zrow.values)
                parts.append(f"{int(h):02d}시  | {row_str}")

    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# _build_period_context  (대폭 강화)
# ════════════════════════════════════════════════════════════════════════════

def _build_period_context(
    daily_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    zone_hourly_df: pd.DataFrame,
    inflow_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    daily_ast_df: pd.DataFrame | None = None,
    hourly_ast_df: pd.DataFrame | None = None,
) -> str:
    """전체 기간 분석용 풍부한 컨텍스트 텍스트 구성."""
    daily_sorted = daily_df.sort_values("date").reset_index(drop=True)
    date_min = daily_sorted["date"].min()
    date_max = daily_sorted["date"].max()
    n_days   = len(daily_sorted)

    parts: list[str] = [
        f"## 분석 기간: {date_min} ~ {date_max} ({n_days}일)",
        f"축제 기간: {FESTIVAL_START}(금) ~ {FESTIVAL_END}(일) — 10일간",
    ]

    # ── 날씨 요약 ──────────────────────────────────────────────────────────
    if not weather_df.empty:
        parts.append(f"\n### 날씨 요약\n{weather_summary_text(weather_df)}")
        # 날씨 유형별 집계
        wm = weather_df.set_index("date")["weather"].to_dict()
        from collections import Counter
        wcount = Counter(wm.values())
        parts.append(
            "날씨 분포: " + " / ".join(f"{k} {v}일" for k, v in sorted(wcount.items()))
        )
        # 기온 요약
        temp_max_col = weather_df["temp_max"].dropna()
        temp_min_col = weather_df["temp_min"].dropna()
        if not temp_max_col.empty:
            parts.append(
                f"기온 범위: 최저 {temp_min_col.min():.0f}°C ~ 최고 {temp_max_col.max():.0f}°C "
                f"| 평균 {temp_min_col.mean():.1f}~{temp_max_col.mean():.1f}°C"
            )
        # 강수일
        rain_days = weather_df[weather_df["precipitation"] > 0][["date", "precipitation"]]
        if not rain_days.empty:
            rain_list = ", ".join(
                f"{r['date']}({r['precipitation']:.1f}mm)" for _, r in rain_days.iterrows()
            )
            parts.append(f"강수일: {rain_list}")

    # ── 일별 완전 데이터 테이블 ──────────────────────────────────────────────
    parts.append("\n### 일별 완전 데이터")
    # AST 병합
    ast_map: dict[str, float] = {}
    ios_ast_map: dict[str, float] = {}
    and_ast_map: dict[str, float] = {}
    if daily_ast_df is not None and not daily_ast_df.empty:
        for _, ar in daily_ast_df.iterrows():
            ast_map[str(ar["date"])] = float(ar["ast_hours"])
            ios_ast_map[str(ar["date"])] = float(ar.get("ios_ast_seconds", 0)) / 3600
            and_ast_map[str(ar["date"])] = float(ar.get("and_ast_seconds", 0)) / 3600

    # inflow 피크 시간
    peak_in_map: dict[str, int] = {}
    peak_out_map: dict[str, int] = {}
    for d, grp in inflow_df.groupby("date"):
        if not grp.empty:
            peak_in_map[d]  = int(grp.loc[grp["inflow"].idxmax(), "hour"])
            peak_out_map[d] = int(grp.loc[grp["outflow"].idxmax(), "hour"])

    # hourly 피크 시간
    peak_dc_map: dict[str, tuple[int, int]] = {}
    for d, grp in hourly_df.groupby("date"):
        if not grp.empty:
            r = grp.loc[grp["dc"].idxmax()]
            peak_dc_map[d] = (int(r["hour"]), int(r["dc"]))

    # 날씨 맵
    wrow_map: dict[str, dict] = {}
    if not weather_df.empty:
        for _, wr in weather_df.iterrows():
            wrow_map[str(wr["date"])] = wr.to_dict()

    header = (
        "날짜      | 요일 | 축제일 | Device Count | iOS     | Android | "
        "AST(h)  | 피크DC시 | 피크DC  | 유입피크 | 유출피크 | 날씨    | 기온"
    )
    parts.append(header)
    parts.append("-" * len(header))

    for _, r in daily_sorted.iterrows():
        d        = str(r["date"])
        dt       = pd.Timestamp(d)
        dow      = DOW_KO[dt.dayofweek]
        ftag     = _festival_day_tag(d)
        dc       = int(r["udc"])
        ios_u    = int(r["ios_udc"])
        and_u    = int(r["android_udc"])
        ast_h    = ast_map.get(d, 0.0)
        pk_h, pk_dc = peak_dc_map.get(d, (0, 0))
        pi       = peak_in_map.get(d, 0)
        po       = peak_out_map.get(d, 0)
        wr       = wrow_map.get(d, {})
        weather  = wr.get("weather", "N/A")
        t_max    = wr.get("temp_max")
        t_min    = wr.get("temp_min")
        temp_str = f"{t_min:.0f}~{t_max:.0f}°C" if t_max is not None else "N/A"
        parts.append(
            f"{d} | {dow:^4} | {ftag:^6} | {dc:>12,} | "
            f"{ios_u:>7,} | {and_u:>7,} | {ast_h:>7.1f} | "
            f"{pk_h:02d}시     | {pk_dc:>7,} | {pi:02d}시     | "
            f"{po:02d}시     | {weather:^7} | {temp_str}"
        )

    # ── 최근 추이 (3일) ──────────────────────────────────────────────────────
    last3 = daily_sorted.tail(3)
    dc_trend = " → ".join(f"{int(r['udc']):,}({DOW_KO[pd.Timestamp(r['date']).dayofweek]})" for _, r in last3.iterrows())
    parts.append(f"\n최근 3일 Device Count 추이: {dc_trend}")

    # ── 시간별 패턴 (전기간 평균) ────────────────────────────────────────────
    parts.append("\n### 시간별 패턴 (전기간 평균)")
    hourly_avg = hourly_df.groupby("hour")[["dc", "ios_dc", "android_dc"]].mean().sort_index()
    io_avg = inflow_df.groupby("hour")[["inflow", "outflow"]].mean().sort_index()
    has_ast = hourly_ast_df is not None and not hourly_ast_df.empty
    ast_avg = hourly_ast_df.groupby("hour")["ast_minutes"].mean().sort_index() if has_ast else None

    peak_avg_h = int(hourly_avg["dc"].idxmax())
    parts.append(f"[전기간 평균 피크 시간: {peak_avg_h:02d}시]")

    if has_ast:
        header2 = f"{'시간':>4} | {'평균DC':>7} | {'평균유입':>8} | {'평균유출':>8} | {'평균AST(분)':>11}"
    else:
        header2 = f"{'시간':>4} | {'평균DC':>7} | {'평균유입':>8} | {'평균유출':>8}"
    parts.append(header2)
    parts.append("-" * len(header2))

    for h in range(24):
        dc_val = int(hourly_avg.loc[h, "dc"]) if h in hourly_avg.index else 0
        inf_val = int(io_avg.loc[h, "inflow"]) if h in io_avg.index else 0
        out_val = int(io_avg.loc[h, "outflow"]) if h in io_avg.index else 0
        if has_ast and h in ast_avg.index:
            ast_val = float(ast_avg.loc[h])
            parts.append(
                f"{h:02d}시  | {dc_val:>7,} | {inf_val:>8,} | {out_val:>8,} | {ast_val:>11,.0f}"
            )
        else:
            parts.append(
                f"{h:02d}시  | {dc_val:>7,} | {inf_val:>8,} | {out_val:>8,}"
            )

    # ── 구역별 전기간 통계 ───────────────────────────────────────────────────
    parts.append("\n### 구역별 전기간 통계")
    zone_by_date = zone_hourly_df.groupby(["date", "zone"])["dc"].sum().reset_index()
    zone_total = zone_by_date.groupby("zone")["dc"].sum().sort_values(ascending=False)
    grand_total_zone = zone_total.sum()
    zone_daily_avg   = zone_by_date.groupby("zone")["dc"].mean()

    # 각 구역 피크일
    zone_peak_day = zone_by_date.loc[zone_by_date.groupby("zone")["dc"].idxmax()][["zone", "date", "dc"]]
    zone_peak_map = {str(r["zone"]): (str(r["date"]), int(r["dc"])) for _, r in zone_peak_day.iterrows()}

    parts.append(
        f"{'구역':^10} | {'총DC':>9} | {'비율':>6} | {'일평균DC':>9} | {'피크일':>10} | {'피크DC':>9}"
    )
    parts.append("-" * 65)
    for zone, dc in zone_total.items():
        label    = ZONE_LABELS.get(zone, zone)
        ratio    = dc / grand_total_zone * 100 if grand_total_zone > 0 else 0
        d_avg    = int(zone_daily_avg.get(zone, 0))
        pk_date, pk_dc = zone_peak_map.get(zone, ("N/A", 0))
        parts.append(
            f"{label:^10} | {int(dc):>9,} | {ratio:>5.1f}% | {d_avg:>9,} | {pk_date:>10} | {pk_dc:>9,}"
        )

    # ── 요일별 패턴 ──────────────────────────────────────────────────────────
    parts.append("\n### 요일별 패턴")
    daily_df2 = daily_df.copy()
    daily_df2["dow_num"] = pd.to_datetime(daily_df2["date"]).dt.dayofweek
    dow_stats = daily_df2.groupby("dow_num").agg(
        n=("udc", "count"),
        avg_udc=("udc", "mean"),
        max_udc=("udc", "max"),
    ).sort_index()

    # 평균 유입 피크 시간 (요일별)
    dow_map = {str(r["date"]): pd.Timestamp(r["date"]).dayofweek for _, r in daily_df.iterrows()}
    inflow_df2 = inflow_df.copy()
    inflow_df2["dow_num"] = inflow_df2["date"].map(dow_map)
    dow_peak_in = (
        inflow_df2.groupby(["dow_num", "hour"])["inflow"]
        .mean()
        .groupby(level=0)
        .idxmax()
        .apply(lambda x: x[1] if isinstance(x, tuple) else x)
    )

    ast_dow_map: dict[int, float] = {}
    if daily_ast_df is not None and not daily_ast_df.empty:
        daily_ast_df2 = daily_ast_df.copy()
        daily_ast_df2["dow_num"] = pd.to_datetime(daily_ast_df2["date"]).dt.dayofweek
        ast_dow_map = daily_ast_df2.groupby("dow_num")["ast_hours"].mean().to_dict()

    parts.append(
        f"{'요일':>4} | {'날짜수':>6} | {'평균DC':>8} | {'최대DC':>8} | {'평균유입피크':>12} | {'평균AST(h)':>11}"
    )
    parts.append("-" * 65)
    for dow_idx, st in dow_stats.iterrows():
        pk_in_h = int(dow_peak_in.get(dow_idx, 0)) if dow_idx in dow_peak_in.index else 0
        ast_avg_h = ast_dow_map.get(dow_idx, 0.0)
        parts.append(
            f"{DOW_KO[dow_idx]:>4} | {int(st['n']):>6} | {int(st['avg_udc']):>8,} | "
            f"{int(st['max_udc']):>8,} | {pk_in_h:>11d}시 | {ast_avg_h:>11.1f}"
        )

    # ── 날씨별 통계 ───────────────────────────────────────────────────────────
    if not weather_df.empty:
        parts.append("\n### 날씨별 트래픽 통계")
        merged = daily_df.merge(
            weather_df[["date", "weather", "temp_max", "temp_min", "precipitation"]],
            on="date", how="left"
        )
        w_stats = merged.groupby("weather").agg(
            n=("udc", "count"),
            avg_udc=("udc", "mean"),
            max_udc=("udc", "max"),
            min_udc=("udc", "min"),
        )
        # AST 병합
        if daily_ast_df is not None and not daily_ast_df.empty:
            merged_ast = merged.merge(
                daily_ast_df[["date", "ast_hours"]], on="date", how="left"
            )
            ast_w = merged_ast.groupby("weather")["ast_hours"].mean()
        else:
            ast_w = None

        parts.append(
            f"{'날씨유형':^8} | {'일수':>4} | {'평균DC':>8} | {'최대DC':>8} | {'최소DC':>8} | {'평균AST(h)':>11}"
        )
        parts.append("-" * 65)
        for w_type, ws in w_stats.iterrows():
            ast_h_val = float(ast_w.get(w_type, 0)) if ast_w is not None else 0.0
            parts.append(
                f"{str(w_type):^8} | {int(ws['n']):>4} | {int(ws['avg_udc']):>8,} | "
                f"{int(ws['max_udc']):>8,} | {int(ws['min_udc']):>8,} | {ast_h_val:>11.1f}"
            )

    # ── iOS vs Android 분석 ──────────────────────────────────────────────────
    parts.append("\n### iOS vs Android 분석")
    parts.append(
        f"{'날짜':>10} | {'iOS DC':>8} | {'Android DC':>10} | {'iOS비율':>7}"
    )
    parts.append("-" * 45)
    for _, r in daily_sorted.iterrows():
        d     = str(r["date"])
        ios_u = int(r["ios_udc"])
        and_u = int(r["android_udc"])
        total = ios_u + and_u
        ios_p = ios_u / total * 100 if total > 0 else 0
        parts.append(
            f"{d} | {ios_u:>8,} | {and_u:>10,} | {ios_p:>6.0f}%"
        )

    # 전체 합계행
    tot_ios = int(daily_sorted["ios_udc"].sum())
    tot_and = int(daily_sorted["android_udc"].sum())
    tot_all = tot_ios + tot_and
    tot_ios_p = tot_ios / tot_all * 100 if tot_all > 0 else 0
    parts.append(
        f"{'합계':>10} | {tot_ios:>8,} | {tot_and:>10,} | {tot_ios_p:>6.0f}%"
    )

    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# _build_weather_context  (강화)
# ════════════════════════════════════════════════════════════════════════════

def _build_weather_context(
    daily_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    inflow_df: pd.DataFrame | None = None,
) -> str:
    """날씨 영향 분석용 컨텍스트 — 요일 교란 통제 포함."""
    parts: list[str] = ["## 날씨-트래픽 분석 컨텍스트"]

    merged = daily_df.merge(
        weather_df[["date", "weather", "temp_max", "temp_min", "precipitation"]],
        on="date", how="left"
    )
    merged["dow_num"]   = pd.to_datetime(merged["date"]).dt.dayofweek
    merged["is_weekend"] = merged["dow_num"].isin(WEEKEND_DAYS)
    merged["dow_label"]  = merged["dow_num"].map(lambda x: DOW_KO[x])

    # 전체 날씨별 데이터 (풍부하게)
    parts.append("\n### 날짜별 날씨-트래픽 전체 데이터")
    parts.append(
        f"{'날짜':>10} | {'요일':>4} | {'주말':>4} | {'날씨':^8} | "
        f"{'기온':>12} | {'강수':>8} | {'Device Count':>12}"
    )
    parts.append("-" * 78)
    for _, r in merged.sort_values("date").iterrows():
        is_we  = "주말" if r["is_weekend"] else "평일"
        t_min  = r.get("temp_min")
        t_max  = r.get("temp_max")
        t_str  = f"{t_min:.0f}~{t_max:.0f}°C" if t_max is not None else "N/A"
        precip = r.get("precipitation", 0.0) or 0.0
        parts.append(
            f"{r['date']:>10} | {r['dow_label']:>4} | {is_we:>4} | {str(r.get('weather', 'N/A')):^8} | "
            f"{t_str:>12} | {precip:>6.1f}mm | {int(r['udc']):>12,}"
        )

    # 날씨 유형별 집계
    parts.append("\n### 날씨 유형별 집계")
    w_stats = merged.groupby("weather").agg(
        n=("udc", "count"),
        avg_udc=("udc", "mean"),
        max_udc=("udc", "max"),
        min_udc=("udc", "min"),
        avg_temp=("temp_max", "mean"),
    )
    parts.append(
        f"{'날씨':^8} | {'일수':>4} | {'평균DC':>9} | {'최대DC':>9} | {'최소DC':>9} | {'평균최고기온':>12}"
    )
    parts.append("-" * 65)
    for w, ws in w_stats.iterrows():
        parts.append(
            f"{str(w):^8} | {int(ws['n']):>4} | {int(ws['avg_udc']):>9,} | "
            f"{int(ws['max_udc']):>9,} | {int(ws['min_udc']):>9,} | {ws['avg_temp']:>11.1f}°C"
        )

    # ── 요일 통제 후 날씨 효과 ────────────────────────────────────────────────
    parts.append("\n### 요일 통제 후 날씨 효과")
    for is_we, label in [(True, "주말"), (False, "평일")]:
        sub = merged[merged["is_weekend"] == is_we]
        if sub.empty:
            continue
        parts.append(f"\n[{label}]")
        w_we = sub.groupby("weather").agg(
            n=("udc", "count"),
            avg_udc=("udc", "mean"),
        )
        sunny_avg = float(w_we.loc["Sunny", "avg_udc"]) if "Sunny" in w_we.index else None
        parts.append(f"{'날씨':^8} | {'일수':>4} | {'평균DC':>9} | {'Sunny대비':>10}")
        parts.append("-" * 42)
        for w, ws in w_we.iterrows():
            vs_str = (
                _pct_change(float(ws["avg_udc"]), sunny_avg)
                if sunny_avg and str(w) != "Sunny" else "기준"
            )
            parts.append(
                f"{str(w):^8} | {int(ws['n']):>4} | {int(ws['avg_udc']):>9,} | {vs_str:>10}"
            )

    # 기온 구간별 집계
    if merged["temp_max"].notna().any():
        parts.append("\n### 기온 구간별 평균 Device Count")
        bins   = [0, 10, 15, 18, 21, 30]
        labels = ["~10°C", "10~15°C", "15~18°C", "18~21°C", "21°C+"]
        merged["temp_bin"] = pd.cut(merged["temp_max"], bins=bins, labels=labels, right=False)
        temp_stats = merged.groupby("temp_bin", observed=False).agg(
            n=("udc", "count"),
            avg_udc=("udc", "mean"),
        )
        for tb, ts in temp_stats.iterrows():
            if ts["n"] > 0:
                parts.append(f"  {tb}: 평균 Device Count {int(ts['avg_udc']):,} ({int(ts['n'])}일)")

    # 유입 패턴 날씨별 (있을 경우)
    if inflow_df is not None and not inflow_df.empty:
        parts.append("\n### 날씨별 최대 유입 시간")
        wdate_map = weather_df.set_index("date")["weather"].to_dict()
        inflow2 = inflow_df.copy()
        inflow2["weather"] = inflow2["date"].map(wdate_map)
        for w_type, wgrp in inflow2.groupby("weather"):
            peak_h = int(wgrp.groupby("hour")["inflow"].mean().idxmax())
            parts.append(f"  {w_type}: 평균 최대 유입 {peak_h:02d}시")

    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# _build_zone_deep_context
# ════════════════════════════════════════════════════════════════════════════

def _build_zone_deep_context(
    date_str: str,
    zone_hourly_df: pd.DataFrame,
    hourly_ast_df: pd.DataFrame | None,
    weather_df: pd.DataFrame,
) -> str:
    """구역별 심층 분석 컨텍스트."""
    dt  = pd.Timestamp(date_str)
    dow = DOW_KO[dt.dayofweek]
    ftag = _festival_day_tag(date_str)

    parts: list[str] = [
        f"## 구역 심층 분석: {date_str}({dow}) [{ftag}]"
    ]

    # 날씨
    if not weather_df.empty:
        w = weather_df[weather_df["date"] == date_str]
        if not w.empty:
            r = w.iloc[0]
            emoji = r.get("weather_emoji", "")
            parts.append(
                f"날씨: {emoji} {r.get('weather', 'N/A')} "
                f"| 기온 {r.get('temp_min', '?'):.0f}~{r.get('temp_max', '?'):.0f}°C"
            )

    row_z = zone_hourly_df[zone_hourly_df["date"] == date_str].copy()
    if row_z.empty:
        parts.append("구역 데이터 없음")
        return "\n".join(parts)

    # 구역별 총 DC
    zone_total = row_z.groupby("zone")["dc"].sum()
    grand_dc   = zone_total.sum()

    parts.append("\n### 구역별 총 DC 및 점유율")
    for zone, dc in zone_total.sort_values(ascending=False).items():
        label = ZONE_LABELS.get(zone, zone)
        ratio = dc / grand_dc * 100 if grand_dc > 0 else 0
        parts.append(f"  {label}: {int(dc):,} ({ratio:.1f}%)")

    # 시간별 구역 DC 전체 테이블
    zone_pivot = (
        row_z.groupby(["hour", "zone"])["dc"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )
    zones_order = zone_total.sort_values(ascending=False).index.tolist()
    zone_pivot  = zone_pivot[[z for z in zones_order if z in zone_pivot.columns]]

    parts.append("\n### 시간대별 구역 DC 전체 (시간 × 구역)")
    header = "시간 | " + " | ".join(f"{ZONE_LABELS.get(z, z):^10}" for z in zone_pivot.columns)
    parts.append(header)
    parts.append("-" * len(header))
    for h, zrow in zone_pivot.iterrows():
        row_str = " | ".join(f"{int(v):>10,}" for v in zrow.values)
        parts.append(f"{int(h):02d}시 | {row_str}")

    # 구역별 피크 시간
    parts.append("\n### 구역별 피크 시간")
    for zone in zones_order:
        if zone not in row_z["zone"].values:
            continue
        z_grp = row_z[row_z["zone"] == zone]
        if z_grp.empty:
            continue
        peak_h  = int(z_grp.loc[z_grp["dc"].idxmax(), "hour"])
        peak_dc = int(z_grp["dc"].max())
        label   = ZONE_LABELS.get(zone, zone)
        parts.append(f"  {label}: 피크 {peak_h:02d}시 (DC={peak_dc:,})")

    # 전기간 구역 패턴 (비교 맥락)
    all_zone_daily = (
        zone_hourly_df.groupby(["date", "zone"])["dc"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )
    if date_str in all_zone_daily.index:
        parts.append("\n### 구역별 시계열 DC (전기간 — 구역 간 비교)")
        zones_cols = [z for z in zones_order if z in all_zone_daily.columns]
        header2 = f"{'날짜':>10} | " + " | ".join(f"{ZONE_LABELS.get(z, z):^10}" for z in zones_cols)
        parts.append(header2)
        parts.append("-" * len(header2))
        for d, zrow in all_zone_daily[zones_cols].iterrows():
            highlight = " ←" if d == date_str else ""
            row_str = " | ".join(f"{int(v):>10,}" for v in zrow.values)
            parts.append(f"{d} | {row_str}{highlight}")

    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# _build_ast_pattern_context
# ════════════════════════════════════════════════════════════════════════════

def _build_ast_pattern_context(
    daily_ast_df: pd.DataFrame,
    hourly_ast_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    daily_df: pd.DataFrame | None = None,
) -> str:
    """AST 패턴 분석 컨텍스트."""
    parts: list[str] = ["## AST(누적 체류시간) 패턴 분석"]

    if daily_ast_df is None or daily_ast_df.empty:
        parts.append("AST 데이터 없음")
        return "\n".join(parts)

    # 일별 AST 전체 테이블
    daily_ast_sorted = daily_ast_df.sort_values("date").reset_index(drop=True)

    # Device Count 병합 (기기당 평균 체류시간 계산)
    if daily_df is not None and not daily_df.empty:
        ast_merged = daily_ast_sorted.merge(
            daily_df[["date", "udc", "ios_udc", "android_udc"]],
            on="date", how="left"
        )
    else:
        ast_merged = daily_ast_sorted.copy()
        for col in ["udc", "ios_udc", "android_udc"]:
            ast_merged[col] = 0

    # 날씨 병합
    if not weather_df.empty:
        ast_merged = ast_merged.merge(
            weather_df[["date", "weather", "temp_max"]],
            on="date", how="left"
        )
    else:
        ast_merged["weather"] = "N/A"
        ast_merged["temp_max"] = float("nan")

    parts.append("\n### 일별 AST 전체 데이터")
    parts.append(
        f"{'날짜':>10} | {'요일':>4} | {'AST(시간)':>10} | {'기기당(분)':>11} | "
        f"{'iOS_AST(h)':>11} | {'And_AST(h)':>11} | {'날씨':^8}"
    )
    parts.append("-" * 85)

    for _, r in ast_merged.iterrows():
        d      = str(r["date"])
        dt     = pd.Timestamp(d)
        dow    = DOW_KO[dt.dayofweek]
        ast_h  = float(r["ast_hours"])
        udc    = max(int(r.get("udc", 1)), 1)
        avg_m  = ast_h * 60 / udc
        ios_h  = float(r.get("ios_ast_seconds", 0)) / 3600
        and_h  = float(r.get("and_ast_seconds", 0)) / 3600
        weather = str(r.get("weather", "N/A"))
        parts.append(
            f"{d} | {dow:>4} | {ast_h:>10.1f} | {avg_m:>11.0f} | "
            f"{ios_h:>11.1f} | {and_h:>11.1f} | {weather:^8}"
        )

    # AST 통계 요약
    parts.append("\n### AST 요약 통계")
    ast_h_series = daily_ast_sorted["ast_hours"]
    parts.append(f"  총 AST: {ast_h_series.sum():,.1f}시간")
    parts.append(f"  일 평균 AST: {ast_h_series.mean():,.1f}시간")
    parts.append(f"  최대 AST일: {daily_ast_sorted.loc[ast_h_series.idxmax(), 'date']} ({ast_h_series.max():,.1f}시간)")
    parts.append(f"  최소 AST일: {daily_ast_sorted.loc[ast_h_series.idxmin(), 'date']} ({ast_h_series.min():,.1f}시간)")

    # 날씨별 평균 AST
    if "weather" in ast_merged.columns:
        w_ast = ast_merged.groupby("weather")["ast_hours"].agg(["mean", "count"])
        parts.append("\n### 날씨별 평균 AST")
        for w, ws in w_ast.iterrows():
            parts.append(f"  {w}: 평균 {ws['mean']:.1f}시간 ({int(ws['count'])}일)")

    # 시간별 AST 전체 테이블 (전기간 평균)
    if hourly_ast_df is not None and not hourly_ast_df.empty:
        parts.append("\n### 시간별 평균 AST (전기간)")
        hast_avg = hourly_ast_df.groupby("hour")["ast_minutes"].mean().sort_index()
        peak_hast = int(hast_avg.idxmax())
        parts.append(f"[피크 시간: {peak_hast:02d}시 ({hast_avg[peak_hast]:,.0f}분)]")
        parts.append(f"{'시간':>4} | {'평균AST(분)':>11} | 막대")
        parts.append("-" * 40)
        max_val = hast_avg.max()
        for h, val in hast_avg.items():
            bar = "█" * min(int(val / max(max_val / 20, 1)), 20)
            parts.append(f"{int(h):02d}시  | {val:>11,.0f} | {bar}")

        # 날짜별 시간별 AST 피크 테이블
        parts.append("\n### 날짜별 시간별 AST 피크")
        for d, grp in hourly_ast_df.sort_values("date").groupby("date"):
            if grp.empty:
                continue
            pk_h   = int(grp.loc[grp["ast_minutes"].idxmax(), "hour"])
            pk_min = float(grp["ast_minutes"].max())
            dt2    = pd.Timestamp(str(d))
            dow2   = DOW_KO[dt2.dayofweek]
            parts.append(
                f"  {d}({dow2}): 피크 {pk_h:02d}시 (AST={pk_min:,.0f}분)"
            )

    return "\n".join(parts)


# ── Claude API 호출 ──────────────────────────────────────────────────────────

def _call_claude(system: str, user: str, max_tokens: int = 2000) -> str:
    api_key = _load_api_key()
    if not api_key:
        return "ANTHROPIC_API_KEY가 설정되지 않았습니다. .env 파일에 키를 추가해주세요."
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = max_tokens,
            system     = system,
            messages   = [{"role": "user", "content": user}],
        )
        return msg.content[0].text
    except Exception as e:
        logger.error("Claude API 오류: %s", e)
        return f"AI 분석 오류: {e}"


# ════════════════════════════════════════════════════════════════════════════
# 공개 분석 함수
# ════════════════════════════════════════════════════════════════════════════

def analyze_single_day(
    date_str: str,
    daily_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    zone_hourly_df: pd.DataFrame,
    inflow_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    daily_ast_df: pd.DataFrame | None = None,
    hourly_ast_df: pd.DataFrame | None = None,
    fine_5min_df: pd.DataFrame | None = None,
) -> str:
    """단일 날짜 AI 분석 (풍부한 컨텍스트 포함)."""
    context = _build_daily_context(
        date_str, daily_df, hourly_df, zone_hourly_df, inflow_df, weather_df,
        daily_ast_df=daily_ast_df,
        hourly_ast_df=hourly_ast_df,
        fine_5min_df=fine_5min_df,
    )
    prompt = f"""{context}

위 데이터를 바탕으로 다음을 분석해주세요:

1. **트래픽 특성**: 이 날의 방문 패턴 (피크 시간, 전반적 흐름, 전일 대비 특이점)
2. **체류 품질**: AST 데이터를 통해 방문자들이 얼마나 오래, 어느 시간대에 머물렀는지
3. **유입/유출 패턴**: 방문객이 언제 주로 도착하고 떠났는지, 누적 재실 피크와 의미
4. **구역별 특성**: 어느 구역이 활성화되었고, 구역 간 시간대별 이동 패턴은?
5. **날씨 영향**: 기상 조건이 이날 트래픽에 미친 영향 (다른 날과 비교)
6. **운영 인사이트**: 혼잡 관리 및 방문객 경험 개선을 위한 구체적 제언 2가지

각 항목은 데이터 수치를 근거로 2~4문장으로 작성해주세요."""
    return _call_claude(SYSTEM_PROMPT, prompt, max_tokens=2000)


def analyze_full_period(
    daily_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    zone_hourly_df: pd.DataFrame,
    inflow_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    daily_ast_df: pd.DataFrame | None = None,
    hourly_ast_df: pd.DataFrame | None = None,
) -> str:
    """전체 기간 종합 AI 분석."""
    context = _build_period_context(
        daily_df, hourly_df, zone_hourly_df, inflow_df, weather_df,
        daily_ast_df=daily_ast_df,
        hourly_ast_df=hourly_ast_df,
    )
    prompt = f"""{context}

위 전체 기간 데이터를 바탕으로 종합 분석을 작성해주세요:

1. **전체 트래픽 동향**: 축제 기간 동안의 방문자 흐름 (상승/하락/피크일과 그 원인)
2. **요일 패턴**: 주말 vs 평일 트래픽 차이 (수치 포함), 특이 요일
3. **시간대 패턴**: 하루 중 방문 집중 시간대와 구조 (출발-피크-해산)
4. **체류시간(AST) 분석**: 방문 "질" 지표 — 어느 날, 어느 시간대에 방문자가 오래 머물렀는가?
5. **날씨 영향**: 맑은 날과 비 오는 날의 트래픽 차이 (요일 효과 분리)
6. **구역 분석**: 가장 활성화된 구역과 조용한 구역, 시간대별 구역 이동 패턴
7. **유입/유출 패턴**: 방문객 도착·출발 시간대 분포 (오전/오후/저녁 비중)
8. **iOS vs Android 비율 분석**: 날씨/요일에 따른 기기 유형 비율 변화 의미
9. **핵심 발견**: 이번 군항제에서 발견된 가장 중요한 트래픽 패턴 3가지 (수치 포함)
10. **운영 개선 제언**: 내년 축제를 위한 실질적 제언 3가지

전문적이지만 이해하기 쉽게, 구체적 수치와 함께 작성해주세요."""
    return _call_claude(SYSTEM_PROMPT, prompt, max_tokens=2500)


def analyze_weather_impact(
    daily_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    inflow_df: pd.DataFrame | None = None,
) -> str:
    """날씨와 트래픽 상관관계 AI 분석 (요일 통제 포함)."""
    if weather_df.empty:
        return "날씨 데이터를 불러올 수 없습니다."
    context = _build_weather_context(daily_df, weather_df, inflow_df)
    prompt = f"""{context}

위 데이터에서 날씨가 군항제 방문객 트래픽에 미치는 영향을 분석해주세요:

1. **비 vs 맑음**: 강수 시 Device Count 변화율 및 패턴 차이 (구체적 수치 포함)
2. **기온 영향**: 기온 구간별 방문자 수 분포 — 최적 기온대는?
3. **요일 교란 제거**: 날씨 효과를 주말/평일로 분리하여 순수 날씨 영향만 추출
4. **방문 시간대 변화**: 날씨에 따른 방문 피크 시간 이동 여부
5. **체감 날씨 vs 트래픽**: 기온, 강수량, 날씨 유형 중 어느 요소가 가장 큰 영향?
6. **종합 평가**: 이번 군항제에서 날씨가 방문객 수에 미친 영향의 크기와 방향

데이터에 근거하여 구체적 수치와 함께 작성해주세요."""
    return _call_claude(SYSTEM_PROMPT, prompt, max_tokens=2000)


def analyze_zone_deep(
    date_str: str,
    zone_hourly_df: pd.DataFrame,
    hourly_ast_df: pd.DataFrame | None,
    weather_df: pd.DataFrame,
) -> str:
    """구역별 심층 분석 — 구역 간 트래픽 이동 패턴, 시간대별 구역 집중도."""
    context = _build_zone_deep_context(date_str, zone_hourly_df, hourly_ast_df, weather_df)
    prompt = f"""{context}

위 구역별 데이터를 바탕으로 다음을 분석해주세요:

1. **구역 점유율**: 어느 구역에 방문자가 가장 많이/적게 집중되었는가? 그 이유는?
2. **시간대별 구역 이동 패턴**: 오전→오후→저녁에 따라 어느 구역이 부상하고 쇠퇴하는가?
3. **구역 간 상관관계**: 동시에 붐비는 구역 vs 보완적으로 움직이는 구역은?
4. **전기간 대비 이날의 특이점**: 평균적 패턴과 다른 점이 있다면?
5. **구역별 혼잡 위험**: 어느 구역, 어느 시간대에 집중 관리가 필요한가?
6. **운영 제언**: 구역별 안내·동선·자원 배치 최적화 방안

수치를 근거로 구체적으로 작성해주세요."""
    return _call_claude(SYSTEM_PROMPT, prompt, max_tokens=2000)


def analyze_ast_pattern(
    daily_ast_df: pd.DataFrame,
    hourly_ast_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    daily_df: pd.DataFrame | None = None,
) -> str:
    """AST 패턴 분석 — 체류시간이 긴 날/시간대, 방문 질 분석."""
    context = _build_ast_pattern_context(
        daily_ast_df, hourly_ast_df, weather_df, daily_df=daily_df
    )
    prompt = f"""{context}

위 AST(누적 체류시간) 데이터를 바탕으로 방문 "질" 관점에서 분석해주세요:

1. **체류시간 패턴**: 방문자가 가장 오래 머문 날과 그 이유 (날씨·요일·축제일차)
2. **시간대별 체류 강도**: 어느 시간대에 방문자들이 가장 오래 머무는가?
3. **기기 유형별 차이**: iOS vs Android 체류시간 차이 — 방문자 프로파일 시사점
4. **날씨와 체류시간**: 맑은 날과 비 오는 날에 체류시간이 어떻게 달라지는가?
5. **Device Count vs AST 괴리**: 방문자 수는 많지만 체류시간이 짧은 날, 반대의 날은?
6. **방문 경험 개선**: AST 분석을 바탕으로 방문자 체류를 늘리기 위한 운영 제언

구체적 수치와 함께 한국어로 작성해주세요."""
    return _call_claude(SYSTEM_PROMPT, prompt, max_tokens=2000)
