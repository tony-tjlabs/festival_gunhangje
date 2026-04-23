"""Festival_Gunhangje — 진해 군항제 트래픽 분석 대시보드.

탭 구성
-------
1. 개요        — 일별 Device Count 트렌드, 요일별 평균, 날씨별 분포
2. 시간대 분석  — 날짜별 오버레이, 요일×시간(30분) 히트맵, iOS/Android 비율, 피크 테이블
3. 인원 변화   — 순유입 바+Device Count 라인 (10분/30분), 전기간 히트맵
4. 구역별 분석  — 파이/스택 바, 일별 추이, S-Ward DC 지도
5. 날씨 영향   — 날씨별 박스플롯, 기온/강수 산점도, 날씨별 시간 패턴
6. AI 인사이트  — 전체기간 / 단일날짜 / 날씨영향 Claude API 분석

실행: streamlit run app.py --server.port 8560
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── 경로 설정 ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    ALL_ZONES, APP_ICON, APP_TITLE,
    DOW_KO, FESTIVAL_END, FESTIVAL_START,
    ZONE_LABELS,
)
from preprocessor import (
    build_cache, cache_exists, cache_info, discover_dates, load_cache,
)
from weather import fetch_weather, make_date_label
from charts import (
    # 탭 1
    chart_daily_trend, chart_dow_avg, chart_weather_udc_box,
    # 탭 1 — 누적 체류시간
    chart_daily_ast,
    # 탭 2
    chart_hourly_overlay, chart_dow_hour_heatmap, chart_peak_hour_table,
    chart_ios_android_30min_bar,
    # 탭 2 — 5분/누적 체류시간
    chart_fine_5min, chart_hourly_ast, chart_cumulative_ast_overlay,
    # 탭 3
    chart_net_inflow_fine, chart_inflow_heatmap_fine,
    # 탭 4
    chart_zone_pie, chart_zone_bar_daily, chart_zone_hourly_stacked,
    chart_zone_map, chart_zone_map_with_slider, chart_sward_heatmap_slider,
    # 탭 5 (동선)
    chart_flow_sankey, chart_flow_zone_map, chart_flow_sward_map,
    # 탭 6
    chart_weather_scatter_temp, chart_weather_scatter_precip, chart_weather_hourly_pattern,
    # 탭 8 — 이동성
    chart_mobility_map, chart_mobility_hourly, chart_speed_distribution,
    # 탭 9 — 구역 체류시간
    chart_zone_dwell_bar, chart_zone_dwell_box, chart_zone_dwell_heatmap,
    # 탭 9 — 누적 체류시간
    chart_cumulative_daily, chart_cumulative_hourly, chart_cumulative_30min,
    # 공통 — 구역 위치 안내
    chart_zone_highlight,
)
from movement_analyzer import (
    compute_and_cache, cache_exists as movement_cache_exists,
    load_mobility_cache, load_dwell_cache, load_mac_mobility_cache,
    load_zone_cumulative_cache,
    aggregate_mobility, aggregate_dwell,
)
from config import ZONE_COLORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Streamlit 설정 ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title            = APP_TITLE,
    page_icon             = APP_ICON,
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ── 전역 CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
.stTabs [data-baseweb="tab"] { font-size: 0.95rem; padding: 6px 18px; }
div[data-testid="stSidebarContent"] h2 { color: #e5604a; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 헬퍼
# ════════════════════════════════════════════════════════════════════════════

def _is_festival(date_str: str) -> bool:
    return FESTIVAL_START <= date_str <= FESTIVAL_END


@st.cache_data(show_spinner=False)
def _load_swards() -> pd.DataFrame:
    swards_path = BASE_DIR / "Data" / "swards.csv"
    if swards_path.exists():
        return pd.read_csv(swards_path)
    return pd.DataFrame(columns=["name", "description", "x", "y"])


# ════════════════════════════════════════════════════════════════════════════
# 데이터 로드
# ════════════════════════════════════════════════════════════════════════════

def _load_all() -> dict:
    """캐시 로드 + 날씨 데이터 포함 통합 dict 반환.

    일별/시간별 Device Count는 raw 값 그대로 사용 (연인원 근사, 요일 비교용).
    Device Count는 daily_df의 udc(일별 unique MAC 수)를 그대로 사용.
    """
    data = load_cache(BASE_DIR)
    daily_df = data["daily"]

    # ── 날씨 데이터 ──────────────────────────────────────────────────────────
    if not daily_df.empty:
        start = daily_df["date"].min()
        end   = daily_df["date"].max()
        weather_df = fetch_weather(start, end)
    else:
        weather_df = pd.DataFrame()

    data["weather"] = weather_df
    return data


# ════════════════════════════════════════════════════════════════════════════
# 사이드바
# ════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> dict | None:
    with st.sidebar:
        st.title(APP_ICON + " 군항제 대시보드")
        st.markdown("---")

        # 축제 정보 (보정 수치 동적 표시)
        st.markdown("### 진해 군항제 2026")
        st.markdown("""
- **사전기간**: 3/25(수) ~ 3/26(목)
- **축제기간**: 3/27(금) ~ 4/5(일)  10일간
- **분석 데이터**: 12일, BLE S-Ward 56개
""")
        st.markdown("---")

        # 캐시 상태
        st.markdown("### 캐시 상태")
        if cache_exists(BASE_DIR):
            info = cache_info(BASE_DIR)
            for key, meta in info.items():
                st.markdown(f"- **{key}**: {meta['size_mb']} MB `{meta['mtime']}`")
            st.success("캐시 정상")

            if st.button("캐시 재빌드", help="raw CSV 재처리"):
                progress_bar = st.progress(0.0, text="준비 중...")
                def _cb(ratio: float, msg: str) -> None:
                    progress_bar.progress(ratio, text=msg)
                ok = build_cache(BASE_DIR, progress_callback=_cb)
                if ok:
                    st.success("재빌드 완료 — 새로고침 필요")
                    st.cache_data.clear()
                else:
                    st.error("재빌드 실패 — 로그 확인")
        else:
            st.warning("캐시 없음")
            dates = discover_dates(BASE_DIR)
            st.markdown(f"발견된 CSV: {len(dates)}개")
            if st.button("캐시 빌드"):
                progress_bar = st.progress(0.0, text="준비 중...")
                def _cb(ratio: float, msg: str) -> None:
                    progress_bar.progress(ratio, text=msg)
                ok = build_cache(BASE_DIR, progress_callback=_cb)
                if ok:
                    st.success("빌드 완료")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("빌드 실패")
            return None

        st.markdown("---")
        st.caption("TJLABS · Festival Analytics v1.0")

    if not cache_exists(BASE_DIR):
        return None
    return _load_all()


def _render_ratio_comparison(
    fest_ios: float, fest_and: float,
    nfest_ios: float, nfest_and: float,
) -> None:
    """iOS/Android 비율 비교 — 두 기간 나란히 프로그레스 바 스타일."""
    def _bar_html(ios_pct: float, and_pct: float, period: str, emoji: str) -> str:
        return f"""
        <div style="background:#1e2130;border-radius:8px;padding:12px 16px;margin:4px 0;">
          <div style="font-size:0.82rem;color:#aaa;margin-bottom:6px;">{emoji} {period}</div>
          <div style="display:flex;height:44px;border-radius:5px;overflow:hidden;">
            <div style="width:{ios_pct:.1f}%;background:#4e9ff5;
                        display:flex;align-items:center;justify-content:center;
                        font-size:11px;color:white;font-weight:600;">
              iOS {ios_pct:.1f}%
            </div>
            <div style="width:{and_pct:.1f}%;background:#59a14f;
                        display:flex;align-items:center;justify-content:center;
                        font-size:11px;color:white;font-weight:600;">
              Android {and_pct:.1f}%
            </div>
          </div>
        </div>"""

    html = f"""
    <div style="display:flex;gap:10px;">
      <div style="flex:1">{_bar_html(fest_ios, fest_and, "축제기간", "🌸")}</div>
      <div style="flex:1">{_bar_html(nfest_ios, nfest_and, "비축제기간", "📅")}</div>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 탭 1 — 개요
# ════════════════════════════════════════════════════════════════════════════

def render_overview(data: dict) -> None:
    daily_df     = data["daily"]
    daily_ast_df = data["daily_ast"]
    weather_df   = data["weather"]

    st.header("개요")

    # ── KPI 계산 ──────────────────────────────────────────────────────────────
    festival_df   = daily_df[daily_df["date"].between(FESTIVAL_START, FESTIVAL_END)]
    non_fest_df   = daily_df[~daily_df["date"].between(FESTIVAL_START, FESTIVAL_END)]

    avg_festival  = int(festival_df["udc"].mean()) if not festival_df.empty else 0
    avg_non_fest  = int(non_fest_df["udc"].mean()) if not non_fest_df.empty else 0
    total_udc     = int(daily_df["udc"].sum())
    peak_day      = daily_df.loc[daily_df["udc"].idxmax()]
    peak_label    = make_date_label(peak_day["date"], weather_df)

    total_ast_h   = round(daily_ast_df["ast_hours"].sum(), 1) if not daily_ast_df.empty else 0.0
    peak_ast_row  = (daily_ast_df.loc[daily_ast_df["ast_hours"].idxmax()]
                     if not daily_ast_df.empty else None)
    peak_ast_h    = round(float(peak_ast_row["ast_hours"]), 1) if peak_ast_row is not None else 0.0
    peak_ast_lbl  = (make_date_label(str(peak_ast_row["date"]), weather_df)
                     if peak_ast_row is not None else "")

    # iOS/Android 비율 계산 (축제기간 / 비축제기간 분리)
    def _ratio(df: pd.DataFrame) -> tuple[float, float]:
        if df.empty:
            return 0.0, 0.0
        ios_s = int(df["ios_udc"].sum())
        and_s = int(df["android_udc"].sum())
        tot   = ios_s + and_s
        if tot == 0:
            return 0.0, 0.0
        return round(ios_s / tot * 100, 1), round(and_s / tot * 100, 1)

    fest_ios_pct,     fest_and_pct     = _ratio(festival_df)
    non_fest_ios_pct, non_fest_and_pct = _ratio(non_fest_df)

    # ── Row 1: 축제/비축제 일평균 DC, 전체 누적 체류시간, 피크일 누적 체류시간 ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("축제기간 일평균 DC", f"{avg_festival:,}",
              help=f"{FESTIVAL_START} ~ {FESTIVAL_END} 기간 일평균 Device Count")
    c2.metric("비축제기간 일평균 DC", f"{avg_non_fest:,}",
              help="축제 외 날짜 일평균 Device Count")
    c3.metric("전체 누적 체류시간", f"{total_ast_h:,.1f}시간",
              help="모든 기기의 체류시간 합산 (MAC × signal_count × 10초)")
    c4.metric("피크일 누적 체류시간", f"{peak_ast_h:,.1f}시간", delta=peak_ast_lbl)

    # ── Row 2: 피크 일 DC, 전체 DC 합계 ─────────────────────────────────────
    c5, c6 = st.columns(2)
    c5.metric("피크 일 DC", f"{int(peak_day['udc']):,}", delta=peak_label)
    c6.metric("전체 DC 합계", f"{total_udc:,}",
              help="전 기간 일별 Device Count 합산 (연인원 근사값)")

    # ── Row 3: iOS/Android 비율 — 축제 vs 비축제 비교 ────────────────────────
    st.caption("**기기 유형 비율** — 축제기간 vs 비축제기간 비교")
    _render_ratio_comparison(fest_ios_pct, fest_and_pct, non_fest_ios_pct, non_fest_and_pct)

    st.markdown("---")

    st.markdown("---")

    # ── 일별 누적 체류시간 트렌드 ────────────────────────────────────────────
    st.subheader("일별 누적 체류시간 트렌드")
    st.plotly_chart(
        chart_daily_ast(daily_ast_df, weather_df),
        use_container_width=True,
        key="overview_daily_ast",
    )

    # ── 일별 Device Count 트렌드 (접힌 패널) ─────────────────────────────────
    with st.expander("일별 Device Count 트렌드"):
        st.plotly_chart(
            chart_daily_trend(daily_df, weather_df),
            use_container_width=True,
            key="overview_daily_trend",
        )

    # ── 요일별 평균 + 날씨별 분포 나란히 ────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("요일별 평균 Device Count")
        st.plotly_chart(chart_dow_avg(daily_df), use_container_width=True, key="overview_dow_avg")
    with col_b:
        st.subheader("날씨별 Device Count 분포")
        if weather_df.empty:
            st.info("날씨 데이터를 불러올 수 없습니다.")
        else:
            st.plotly_chart(chart_weather_udc_box(daily_df, weather_df), use_container_width=True, key="overview_weather_udc_box")

    # ── 상세 일별 테이블 ────────────────────────────────────────────────────
    with st.expander("일별 상세 데이터 보기"):
        display_df = daily_df.copy().sort_values("date")
        display_df["날짜"] = display_df["date"].apply(
            lambda d: make_date_label(d, weather_df)
        )
        display_df["Device Count"] = display_df["udc"].map("{:,}".format)
        display_df["총 레코드"]      = display_df["total_records"].map("{:,}".format)
        display_df["평균 RSSI"]      = display_df["avg_rssi"].map("{:.1f}".format)
        display_df["축제기간"]        = display_df["date"].apply(
            lambda d: "축제" if _is_festival(d) else "사전"
        )
        st.dataframe(
            display_df[["날짜", "축제기간", "Device Count", "총 레코드", "평균 RSSI"]],
            use_container_width=True,
            hide_index=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# 탭 2 — 시간대 분석
# ════════════════════════════════════════════════════════════════════════════

def render_hourly(data: dict) -> None:
    hourly_df     = data["hourly"]
    fine_5min_df  = data["fine_5min"]
    hourly_ast_df = data["hourly_ast"]
    weather_df    = data["weather"]
    all_dates     = sorted(hourly_df["date"].unique().tolist())

    st.header("시간대 분석")

    # 날짜 멀티셀렉트 (기본값: 전체 or 최대 7일)
    default_dates = all_dates[-7:] if len(all_dates) > 7 else all_dates
    selected = st.multiselect(
        "비교할 날짜 선택 (최대 7일)",
        options=all_dates,
        default=default_dates,
        format_func=lambda d: make_date_label(d, weather_df),
        key="hourly_dates",
    )
    if len(selected) > 7:
        st.warning("최대 7일까지 선택 가능합니다. 앞 7일만 표시합니다.")
        selected = selected[:7]

    if not selected:
        st.info("날짜를 하나 이상 선택해주세요.")
        return

    # ── 시간대별 DC 오버레이 (5분 단위) ─────────────────────────────────────
    st.subheader("시간대별 DC 오버레이")
    st.plotly_chart(
        chart_hourly_overlay(fine_5min_df, weather_df, selected),
        use_container_width=True,
        key=f"hourly_overlay_{'_'.join(sorted(selected))}",
    )

    # ── 날짜별 누적 체류시간 비교 ─────────────────────────────────────────────
    st.subheader("날짜별 누적 체류시간 비교")
    st.caption(
        "0시부터 시간이 지날수록 쌓이는 누적 체류시간. "
        "실선=축제기간, 점선=비축제기간. 오른쪽 끝 숫자=하루 총 체류시간(분)."
    )
    st.plotly_chart(
        chart_cumulative_ast_overlay(hourly_ast_df, weather_df, selected),
        use_container_width=True,
        key=f"cumulative_ast_{'_'.join(sorted(selected))}",
    )

    # ── 요일×시간(30분) 히트맵 + 피크 테이블 나란히 ─────────────────────────
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.subheader("요일 × 시간(30분) 히트맵")
        st.plotly_chart(
            chart_dow_hour_heatmap(fine_5min_df),
            use_container_width=True,
            key="hourly_dow_heatmap",
        )
    with col_b:
        st.subheader("날짜별 피크 시간")
        peak_tbl = chart_peak_hour_table(hourly_df, weather_df)
        st.dataframe(peak_tbl, use_container_width=True, hide_index=True)

    # ── 5분 단위 / 시간별 누적 체류시간 단일 날짜 선택 ─────────────────────
    st.markdown("---")
    st.subheader("단일 날짜 심층 분석")
    detail_date = st.selectbox(
        "날짜 선택 (5분 Device Count & 시간별 누적 체류시간)",
        options=all_dates,
        index=len(all_dates) - 1,
        format_func=lambda d: make_date_label(d, weather_df),
        key="hourly_detail_date",
    )

    # 5분 단위 Device Count
    st.subheader("5분 단위 Device Count")
    st.caption("5분 윈도우 내 unique MAC 기반 Device Count")
    st.plotly_chart(
        chart_fine_5min(fine_5min_df, detail_date, weather_df),
        use_container_width=True,
        key="hourly_fine_5min",
    )

    # iOS/Android 30분 누적 막대
    st.subheader("30분 단위 iOS/Android Device Count")
    st.caption("막대 아래: Android (초록) / 위: iOS (파랑). 각 막대 내 iOS 비율(%) 표시.")
    st.plotly_chart(
        chart_ios_android_30min_bar(fine_5min_df, detail_date, weather_df),
        use_container_width=True,
        key=f"hourly_ios_android_30min_{detail_date}",
    )

    # 시간별 누적 체류시간
    st.subheader("시간별 누적 체류시간")
    st.caption(
        "누적 체류시간 = 각 MAC의 signal_count × 10초 합산. "
        "인원수 대신 체류 강도를 측정하므로 MAC 랜덤화 영향 최소화."
    )
    st.plotly_chart(
        chart_hourly_ast(hourly_ast_df, detail_date),
        use_container_width=True,
        key="hourly_ast",
    )


# ════════════════════════════════════════════════════════════════════════════
# 탭 3 — 유입/유출
# ════════════════════════════════════════════════════════════════════════════

def render_inflow(data: dict) -> None:
    fine_5min_df = data["fine_5min"]
    weather_df   = data["weather"]
    all_dates    = sorted(fine_5min_df["date"].unique().tolist())

    st.header("인원 변화 분석")

    # ── 컨트롤 행 ──────────────────────────────────────────────────────────
    ctrl_a, ctrl_b = st.columns([3, 1])
    with ctrl_a:
        selected_date = st.selectbox(
            "날짜 선택",
            options=all_dates,
            index=len(all_dates) - 1,
            format_func=lambda d: make_date_label(d, weather_df),
            key="inflow_date",
        )
    with ctrl_b:
        resolution = st.radio(
            "시간 해상도",
            options=[10, 30],
            format_func=lambda v: f"{v}분",
            horizontal=True,
            index=1,
            key="inflow_resolution",
        )

    # ── KPI ────────────────────────────────────────────────────────────────
    sub = fine_5min_df[fine_5min_df["date"] == selected_date].sort_values("bin_5min")
    if not sub.empty:
        import numpy as np
        bins_pw    = resolution // 5
        sub        = sub.copy()
        sub["win"] = sub["bin_5min"] // bins_pw
        agg        = sub.groupby("win")["corrected_dc"].mean()
        delta      = agg.diff().dropna()

        peak_occ_val  = int(agg.max())
        peak_occ_time = int(agg.idxmax()) * resolution
        peak_in_val   = int(delta[delta > 0].max()) if (delta > 0).any() else 0
        peak_in_time  = int(delta.idxmax()) * resolution if not delta.empty else 0
        peak_out_val  = int(delta[delta < 0].min()) if (delta < 0).any() else 0
        peak_out_time = int(delta.idxmin()) * resolution if not delta.empty else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("피크 Device Count",
                  f"{peak_occ_val:,}",
                  help=f"최대 Device Count ({peak_occ_time//60:02d}:{peak_occ_time%60:02d})")
        c2.metric("최대 순증가 DC",
                  f"+{peak_in_val:,}",
                  help=f"가장 빠르게 DC 증가한 {resolution}분 구간 ({peak_in_time//60:02d}:{peak_in_time%60:02d})")
        c3.metric("최대 순감소 DC",
                  f"{peak_out_val:,}",
                  help=f"가장 빠르게 DC 감소한 {resolution}분 구간 ({peak_out_time//60:02d}:{peak_out_time%60:02d})")
        c4.metric("분석 해상도", f"{resolution}분 단위")

    # ── 순유입 + Device Count 콤보 차트 ─────────────────────────────────────
    st.plotly_chart(
        chart_net_inflow_fine(fine_5min_df, selected_date, weather_df, resolution),
        use_container_width=True,
        key="inflow_net_fine",
    )

    st.caption(
        "* 순유입(바) = 해당 구간 Device Count 변화량. 초록=증가, 빨강=감소. "
        "Device Count(선) = 5분 단위 Device Count의 구간 평균."
    )

    # ── 전기간 히트맵 ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("전 기간 Device Count 히트맵")
    st.plotly_chart(
        chart_inflow_heatmap_fine(fine_5min_df, weather_df, resolution),
        use_container_width=True,
        key="inflow_heatmap_fine",
    )


# ════════════════════════════════════════════════════════════════════════════
# 탭 4 — 구역별 분석
# ════════════════════════════════════════════════════════════════════════════

def render_zone(data: dict) -> None:
    zone_df        = data["zone_hourly"]
    sward_hourly_df = data.get("sward_hourly", pd.DataFrame())
    weather_df     = data["weather"]
    swards_df      = _load_swards()
    all_dates      = sorted(zone_df["date"].unique().tolist())

    st.header("구역별 분석")

    selected_date = st.selectbox(
        "날짜 선택",
        options=all_dates,
        index=min(len(all_dates) - 1, 3),   # 기본: 3/28(토) 피크일 근방
        format_func=lambda d: make_date_label(d, weather_df),
        key="zone_date",
    )

    # ── 파이 + 지도 나란히 ───────────────────────────────────────────────────
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.subheader("구역별 DC 비율")
        st.plotly_chart(
            chart_zone_pie(zone_df, selected_date),
            use_container_width=True,
            key="zone_pie",
        )
    with col_b:
        st.subheader("S-Ward 구역 지도")
        st.plotly_chart(
            chart_zone_map(swards_df, zone_df, selected_date),
            use_container_width=True,
            key="zone_map",
        )

    # ── 시간대별 히트맵 — 지도(좌) + 컨트롤·정보(우) ────────────────────────
    st.markdown("---")
    st.subheader("시간대별 Heatmap")

    col_map, col_ctrl = st.columns([3, 1])

    # ── 우측 컨트롤 패널 ─────────────────────────────────────────────────────
    with col_ctrl:
        st.markdown("**모드**")
        heatmap_mode = st.radio(
            "히트맵 모드",
            options=["시간별", "누적 (0시~)"],
            index=0,
            key="heatmap_mode",
            label_visibility="collapsed",
            help="시간별: 선택 시간대 단독 혼잡도 / 누적: 0시부터 선택 시간까지 합산",
        )
        is_cumulative = (heatmap_mode == "누적 (0시~)")

        st.markdown("**날짜 선택**")
        map_date = st.selectbox(
            "날짜",
            options=all_dates,
            index=min(len(all_dates) - 1, 3),
            format_func=lambda d: make_date_label(d, weather_df),
            key="zone_map_slider_date",
            label_visibility="collapsed",
        )

        st.markdown("**시간 선택**")
        map_hour = st.select_slider(
            "시간",
            options=list(range(24)),
            value=12,
            format_func=lambda h: f"{h:02d}시",
            key="zone_map_slider_hour",
            label_visibility="collapsed",
        )

        # ── 해당 시간 구역별 DC 요약 ─────────────────────────────────────────
        st.markdown("---")
        dc_label = f"00~{map_hour:02d}시 누적 DC" if is_cumulative else f"{map_hour:02d}시 구역별 DC"
        st.markdown(f"**{dc_label}**")
        if not sward_hourly_df.empty and not zone_df.empty:
            from config import ZONE_LABELS, ZONE_COLORS
            if is_cumulative:
                zh_raw = zone_df[
                    (zone_df["date"] == map_date) &
                    (zone_df["hour"] <= map_hour)
                ].groupby("zone")["dc"].sum().reset_index().sort_values("dc", ascending=False)
            else:
                zh_raw = zone_df[
                    (zone_df["date"] == map_date) &
                    (zone_df["hour"] == map_hour)
                ].sort_values("dc", ascending=False)

            if not zh_raw.empty:
                for _, zrow in zh_raw.iterrows():
                    zone  = zrow["zone"]
                    label = ZONE_LABELS.get(zone, zone)
                    color = ZONE_COLORS.get(zone, "#888888")
                    dc_v  = int(zrow["dc"])
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:6px;'
                        f'margin:3px 0;">'
                        f'<div style="width:10px;height:10px;border-radius:50%;'
                        f'background:{color};flex-shrink:0;"></div>'
                        f'<span style="font-size:0.82rem;">{label}</span>'
                        f'<span style="margin-left:auto;font-weight:600;'
                        f'font-size:0.85rem;">{dc_v:,}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("데이터 없음")

    # ── 좌측 지도 ─────────────────────────────────────────────────────────────
    with col_map:
        if not sward_hourly_df.empty:
            st.plotly_chart(
                chart_sward_heatmap_slider(
                    sward_hourly_df, swards_df, map_date, map_hour, BASE_DIR,
                    cumulative=is_cumulative,
                ),
                use_container_width=True,
                key="zone_sward_heatmap",
            )
        else:
            st.plotly_chart(
                chart_zone_map_with_slider(zone_df, swards_df, map_date, map_hour, BASE_DIR),
                use_container_width=True,
                key="zone_map_slider",
            )

    # ── 시간별 누적 바 ───────────────────────────────────────────────────────
    st.subheader("시간별 구역 DC 누적")
    st.plotly_chart(
        chart_zone_hourly_stacked(zone_df, selected_date),
        use_container_width=True,
        key="zone_hourly_stacked",
    )

    # ── 일별 구역 추이 ───────────────────────────────────────────────────────
    st.subheader("구역별 일별 DC 추이 (전 기간)")
    st.plotly_chart(
        chart_zone_bar_daily(zone_df, weather_df),
        use_container_width=True,
        key="zone_bar_daily",
    )

    # ── 구역 요약 테이블 ─────────────────────────────────────────────────────
    with st.expander("구역 정의 보기"):
        rows = [
            {"구역": ZONE_LABELS.get(z, z), "설명": desc, "S-Ward 수": cnt}
            for z, desc, cnt in [
                ("A구역", "북부 (경화역, 제황산 방면)", 15),
                ("B구역", "중부 (중원로터리, 도심)", 11),
                ("C구역", "남서부 (여좌천, 로망스 다리)", 8),
                ("D구역", "남중부 (해군사관학교 방면)", 10),
                ("E구역", "남동부 (해안도로 방면)", 12),
            ]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# 탭 5 — 동선 분석
# ════════════════════════════════════════════════════════════════════════════

def render_flow_analysis(data: dict) -> None:
    """구역 간 이동 흐름 분석 탭 (Sankey + 구역 지도 화살표)."""
    import plotly.graph_objects as go

    flow_df    = data.get("flow", pd.DataFrame())
    swards_df  = _load_swards()
    weather_df = data["weather"]
    zone_df    = data["zone_hourly"]
    all_dates  = sorted(zone_df["date"].unique().tolist())

    st.header("동선 분석")

    st.info(
        "**동선 분석 원리** — 동일 MAC이 10분 이내에 다른 S-Ward에서 감지되면 이동 전환(Transition)으로 기록합니다. "
        "S-Ward를 5개 구역(A~E)으로 집계하여 **구역 간 이동 흐름**을 시각화합니다.\n\n"
        "- **Sankey 다이어그램**: 구역 간 이동량을 흐름 두께로 표현. 어느 구역에서 어느 구역으로 이동하는지 한눈에 파악.\n"
        "- **구역 이동 지도**: 지도 위 5개 구역 중심점을 화살표로 연결. 굵기/색상 = 이동 빈도."
    )

    if flow_df.empty:
        st.warning("동선 데이터 캐시가 없습니다. 사이드바에서 캐시를 재빌드해주세요.")
        return

    # ── 컨트롤 ────────────────────────────────────────────────────────────────
    ctrl_a, ctrl_b = st.columns([4, 1])
    with ctrl_a:
        selected_date = st.selectbox(
            "날짜 선택",
            options=all_dates,
            index=min(len(all_dates) - 1, 3),
            format_func=lambda d: make_date_label(d, weather_df),
            key="flow_date",
        )
    with ctrl_b:
        st.markdown("<br>", unsafe_allow_html=True)

    flow_hour = st.slider(
        "시간 선택",
        min_value=0, max_value=23, value=12, step=1, format="%d시",
        key="flow_hour_slider",
    )

    # ── KPI ──────────────────────────────────────────────────────────────────
    day_flow = flow_df[flow_df["date"] == selected_date]
    if not day_flow.empty:
        from config import SWARD_TO_ZONE
        zone_day = day_flow.copy()
        zone_day["from_zone"] = zone_day["from_sward"].map(SWARD_TO_ZONE).fillna("미분류")
        zone_day["to_zone"]   = zone_day["to_sward"].map(SWARD_TO_ZONE).fillna("미분류")
        inter = zone_day[
            (zone_day["from_zone"] != "미분류") &
            (zone_day["to_zone"]   != "미분류") &
            (zone_day["from_zone"] != zone_day["to_zone"])
        ]
        total_trans = int(day_flow["count"].sum())
        inter_trans = int(inter["count"].sum()) if not inter.empty else 0

        if not inter.empty:
            top_pair = inter.groupby(["from_zone", "to_zone"])["count"].sum().idxmax()
            top_cnt  = int(inter.groupby(["from_zone", "to_zone"])["count"].sum().max())
            top_label = f"{top_pair[0]} → {top_pair[1]}"
        else:
            top_label, top_cnt = "—", 0

        c1, c2, c3 = st.columns(3)
        c1.metric("총 S-Ward 전환 횟수", f"{total_trans:,}", help="당일 전체 전환 합계")
        c2.metric("구역 간 이동 횟수",    f"{inter_trans:,}", help="서로 다른 구역 간 이동 합계")
        c3.metric("최다 구역 이동",       top_label, delta=f"{top_cnt:,}회")

    # ── Sankey ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"구역 간 이동 흐름 — Sankey ({flow_hour:02d}시)")
    st.caption("흐름 두께 = 이동 횟수. 노드 색상 = 구역 색상.")
    st.plotly_chart(
        chart_flow_sankey(flow_df, selected_date, flow_hour),
        use_container_width=True,
        key=f"flow_sankey_{selected_date}_{flow_hour}",
    )

    # ── S-Ward 레벨 동선 흐름 지도 ───────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"S-Ward 이동 흐름 지도 ({flow_hour:02d}시)")
    st.caption("S-Ward 위치 간 실제 이동 경로. 굵기/색상 = 이동 빈도(노랑→빨강). 상위 50개 전환 표시.")
    st.plotly_chart(
        chart_flow_sward_map(flow_df, swards_df, selected_date, flow_hour, BASE_DIR),
        use_container_width=True,
        key=f"flow_sward_map_{selected_date}_{flow_hour}",
    )

    # ── 시간대별 전환 추이 ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("시간대별 구역 간 이동 추이")
    if not day_flow.empty:
        from config import SWARD_TO_ZONE as _STZ
        zd = day_flow.copy()
        zd["from_zone"] = zd["from_sward"].map(_STZ).fillna("미분류")
        zd["to_zone"]   = zd["to_sward"].map(_STZ).fillna("미분류")
        inter_h = zd[
            (zd["from_zone"] != "미분류") &
            (zd["to_zone"]   != "미분류") &
            (zd["from_zone"] != zd["to_zone"])
        ].groupby("hour")["count"].sum().reset_index()

        fig_h = go.Figure(go.Bar(
            x=inter_h["hour"].tolist(),
            y=inter_h["count"].tolist(),
            marker_color="#f28e2b",
            hovertemplate="%{x:02d}시 구역 간 이동: %{y:,}회<extra></extra>",
        ))
        fig_h.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#ccd6f6"),
            margin=dict(l=55, r=20, t=45, b=50),
            height=300,
            title=f"{selected_date} — 시간대별 구역 간 이동 횟수",
            xaxis=dict(tickmode="linear", dtick=1, range=[-0.5, 23.5]),
            xaxis_title="시간", yaxis_title="이동 횟수",
        )
        st.plotly_chart(fig_h, use_container_width=True, key="flow_hourly_bar")

    # ── 구역 간 이동 매트릭스 ────────────────────────────────────────────────
    with st.expander("구역 간 이동 매트릭스 (당일 전체)"):
        if not day_flow.empty:
            from config import SWARD_TO_ZONE as _STZ2, ALL_ZONES as _AZ, ZONE_LABELS as _ZL
            zd2 = day_flow.copy()
            zd2["from_zone"] = zd2["from_sward"].map(_STZ2).fillna("미분류")
            zd2["to_zone"]   = zd2["to_sward"].map(_STZ2).fillna("미분류")
            mat = (
                zd2[(zd2["from_zone"].isin(_AZ)) & (zd2["to_zone"].isin(_AZ)) &
                    (zd2["from_zone"] != zd2["to_zone"])]
                .groupby(["from_zone", "to_zone"])["count"].sum()
                .unstack(fill_value=0)
                .reindex(index=_AZ, columns=_AZ, fill_value=0)
            )
            mat.index   = [_ZL.get(z, z) for z in mat.index]
            mat.columns = [_ZL.get(z, z) for z in mat.columns]
            st.dataframe(mat.style.background_gradient(cmap="YlOrRd"),
                         use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# 탭 6 — 날씨 영향
# ════════════════════════════════════════════════════════════════════════════

def render_weather(data: dict) -> None:
    daily_df   = data["daily"]
    hourly_df  = data["hourly"]
    weather_df = data["weather"]

    st.header("날씨 영향 분석")

    if weather_df.empty:
        st.warning("날씨 데이터를 불러올 수 없습니다. 인터넷 연결을 확인해주세요.")
        return

    st.info(
        "**날씨 영향 분석** — BLE 센서로 수집한 일별 Device Count(DC)와 날씨 데이터를 교차 분석합니다.\n\n"
        "- **날씨별 DC 분포**: 맑음/흐림/비 등 날씨 유형별로 방문 규모가 어떻게 달라지는지 박스플롯으로 비교\n"
        "- **기온 vs DC**: 최고기온과 당일 DC의 상관관계. 기온이 높을수록 방문자가 늘어나는지 확인\n"
        "- **강수량 vs DC**: 강수량이 증가할수록 DC가 감소하는 경향 파악\n"
        "- **날씨별 시간대 패턴**: 날씨 유형에 따라 피크 시간대가 달라지는지 시각화 "
        "(예: 맑은 날은 오후 집중, 흐린 날은 분산)"
    )

    # ── 날씨 현황 테이블 ─────────────────────────────────────────────────────
    with st.expander("날씨 데이터 원본 보기"):
        display_w = weather_df.copy()
        display_w["날짜"] = display_w["date"].apply(
            lambda d: make_date_label(d, weather_df)
        )
        display_w["기온"] = display_w.apply(
            lambda r: f"{r.get('temp_min', 0):.0f}~{r.get('temp_max', 0):.0f}°C", axis=1
        )
        display_w["강수량"] = display_w["precipitation"].map("{:.1f} mm".format)
        display_w["날씨"] = display_w["weather_emoji"] + " " + display_w["weather"]
        st.dataframe(
            display_w[["날짜", "날씨", "기온", "강수량"]],
            use_container_width=True, hide_index=True,
        )

    # ── 날씨별 Device Count 박스플롯 ────────────────────────────────────────
    st.subheader("날씨 유형별 Device Count 분포")
    st.caption("각 점은 하루의 DC. 박스 중앙선=중앙값, 박스 범위=IQR")
    st.plotly_chart(
        chart_weather_udc_box(daily_df, weather_df),
        use_container_width=True,
        key="weather_udc_box",
    )

    # ── 산점도 2개 나란히 ────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("기온 vs Device Count")
        st.caption("우상향 추세일수록 기온↑ → DC↑ 양의 상관관계")
        st.plotly_chart(
            chart_weather_scatter_temp(daily_df, weather_df),
            use_container_width=True,
            key="weather_scatter_temp",
        )
    with col_b:
        st.subheader("강수량 vs Device Count")
        st.caption("우하향 추세일수록 강수↑ → DC↓ 음의 상관관계")
        st.plotly_chart(
            chart_weather_scatter_precip(daily_df, weather_df),
            use_container_width=True,
            key="weather_scatter_precip",
        )

    # ── 날씨별 시간대 패턴 ───────────────────────────────────────────────────
    st.subheader("날씨 유형별 시간대 평균 DC")
    st.caption("날씨별로 피크 시간대와 방문 분산 패턴이 달라지는지 확인")
    st.plotly_chart(
        chart_weather_hourly_pattern(hourly_df, weather_df),
        use_container_width=True,
        key="weather_hourly_pattern",
    )


# ════════════════════════════════════════════════════════════════════════════
# 탭 6 — AI 인사이트
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_ai_insights() -> dict:
    """사전 생성된 ai_insights.json 로드."""
    p = BASE_DIR / "cache" / "ai_insights.json"
    if p.exists():
        import json
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def render_ai(data: dict) -> None:
    daily_df  = data["daily"]
    weather_df = data["weather"]
    all_dates = sorted(daily_df["date"].unique().tolist())

    st.header("AI 인사이트")

    insights = _load_ai_insights()
    if not insights:
        st.warning("AI 인사이트 파일을 찾을 수 없습니다. (cache/ai_insights.json)")
        return

    subtab_period, subtab_single, subtab_weather, subtab_zone, subtab_ast = st.tabs(
        ["전체기간 종합", "단일 날짜 분석", "날씨 영향 분석", "구역 심층 분석", "체류시간 패턴"]
    )

    # ── 전체기간 종합 ────────────────────────────────────────────────────────
    with subtab_period:
        st.subheader("전체 축제 기간 종합 분석")
        st.markdown(insights.get("full_period", "분석 결과가 없습니다."))

    # ── 단일 날짜 분석 ───────────────────────────────────────────────────────
    with subtab_single:
        st.subheader("단일 날짜 상세 분석")
        selected_date = st.selectbox(
            "날짜 선택",
            options=all_dates,
            format_func=lambda d: make_date_label(d, weather_df),
            key="ai_single_date",
        )
        single_day = insights.get("single_day", {})
        result = single_day.get(selected_date)
        if result:
            st.markdown(result)
        else:
            st.info(f"{selected_date} 분석 결과가 없습니다.")

    # ── 날씨 영향 분석 ───────────────────────────────────────────────────────
    with subtab_weather:
        st.subheader("날씨와 트래픽 상관 분석")
        st.markdown(insights.get("weather_impact", "분석 결과가 없습니다."))

    # ── 구역 심층 분석 ───────────────────────────────────────────────────────
    with subtab_zone:
        st.subheader("구역별 심층 분석")
        zone_date = st.selectbox(
            "날짜 선택",
            options=all_dates,
            format_func=lambda d: make_date_label(d, weather_df),
            key="ai_zone_date",
        )
        zone_deep = insights.get("zone_deep", {})
        result = zone_deep.get(zone_date)
        if result:
            st.markdown(result)
        else:
            st.info(f"{zone_date} 구역 분석 결과가 없습니다.")

    # ── 체류시간 패턴 분석 ──────────────────────────────────────────────────
    with subtab_ast:
        st.subheader("누적 체류시간 패턴 분석")
        st.markdown(insights.get("ast_pattern", "분석 결과가 없습니다."))

    st.markdown("---")
    st.caption("AI 분석은 사전 생성된 결과를 표시합니다. (claude-haiku-4-5, 2026-04-15 생성)")


# ════════════════════════════════════════════════════════════════════════════
# 탭 8 — 이동 속도 (RSSI 변화율 기반)
# ════════════════════════════════════════════════════════════════════════════

def render_movement(data: dict) -> None:
    daily_df  = data["daily"]
    all_dates = sorted(daily_df["date"].unique().tolist())
    swards_df = _load_swards()

    st.header("이동 속도 분석")

    # ── 캐시 로드 ────────────────────────────────────────────────────────────
    sel_date = st.selectbox(
        "날짜 선택",
        options=["전체 기간"] + all_dates,
        key="mv_date",
    )
    targets = all_dates if sel_date == "전체 기간" else [sel_date]
    frames  = [load_mobility_cache(BASE_DIR, d) for d in targets]
    frames  = [f for f in frames if not f.empty]

    if not frames:
        st.info("데이터를 준비 중입니다. `generate_movement_cache.py`를 먼저 실행해주세요.")
        return

    mobility_df = aggregate_mobility(frames)

    # MAC 레벨 이동성 캐시 (속도 분포용)
    mac_frames = [load_mac_mobility_cache(BASE_DIR, d) for d in targets]
    mac_frames = [f for f in mac_frames if not f.empty]
    mac_mobility_df = pd.concat(mac_frames, ignore_index=True) if mac_frames else pd.DataFrame()

    # ── 하단 섹션: 구역별 이동 속도 분포 ────────────────────────────────────
    st.markdown("---")
    st.subheader("구역별 추정 보행속도 분포")
    st.caption(
        "RSSI 이동성 지수를 군항제 축제 환경 기준으로 보행속도(km/h)로 환산합니다.  \n"
        "피크 시간에는 극혼잡(0.1~0.5 km/h)에 가까울 것으로 예측됩니다.  \n"
        "※ BLE 신호 기반 추정값이며 실제 속도와 차이가 있을 수 있습니다."
    )

    from config import ALL_ZONES, ZONE_LABELS, SWARD_TO_ZONE

    ctrl_a, ctrl_b = st.columns(2)
    with ctrl_a:
        sel_zone = st.selectbox(
            "구역 선택",
            options=ALL_ZONES,
            format_func=lambda z: ZONE_LABELS.get(z, z),
            key="mv_speed_zone",
        )
    with ctrl_b:
        sel_speed_hour = st.select_slider(
            "시간 선택",
            options=list(range(24)),
            value=12,
            format_func=lambda h: f"{h:02d}시",
            key="mv_speed_hour",
        )

    # ── KPI: 해당 구역·시간 MAC 기반 속도 ───────────────────────────────────
    from movement_analyzer import _speed_level as _slvl
    from charts import _rssi_range_to_speed as _r2spd

    if not mac_mobility_df.empty:
        mac_filtered = mac_mobility_df[
            (mac_mobility_df["hour"] == sel_speed_hour) &
            (mac_mobility_df["zone"] == sel_zone)
        ]
    else:
        mac_filtered = pd.DataFrame()

    if not mac_filtered.empty:
        col    = "rssi_range_mean" if "rssi_range_mean" in mac_filtered.columns else "rssi_std_mean"
        speeds = mac_filtered[col].apply(_r2spd)
        mean_speed = float(speeds.mean())
        min_speed  = float(speeds.min())
        max_speed  = float(speeds.max())
        n_mac      = len(mac_filtered)
        level_name, _ = _slvl(mean_speed)

        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("평균 추정 속도", f"{mean_speed:.2f} km/h")
        kc2.metric("최저 속도", f"{min_speed:.2f} km/h")
        kc3.metric("최고 속도", f"{max_speed:.2f} km/h")
        kc4.metric("혼잡 수준", f"{level_name} ({n_mac:,}명)")

        st.plotly_chart(
            chart_speed_distribution(mac_mobility_df, sel_zone, sel_speed_hour),
            use_container_width=True,
        )
    else:
        st.info(f"{ZONE_LABELS.get(sel_zone, sel_zone)} — {sel_speed_hour:02d}시 MAC 데이터가 없습니다. 캐시를 재생성해 주세요.")


# ════════════════════════════════════════════════════════════════════════════
# 탭 9 — 구역 체류시간
# ════════════════════════════════════════════════════════════════════════════

def render_zone_dwell(data: dict) -> None:
    daily_df   = data["daily"]
    weather_df = data["weather"]
    all_dates  = sorted(daily_df["date"].unique().tolist())

    st.header("체류시간 분석")

    from config import ZONE_LABELS, ALL_ZONES

    # 날짜 선택
    sel_date = st.selectbox(
        "날짜 선택",
        options=["전체 기간"] + all_dates,
        key="dw_date",
    )
    targets = all_dates if sel_date == "전체 기간" else [sel_date]

    # ── 누적 체류시간 캐시 로드 ───────────────────────────────────────────────
    cum_frames = [load_zone_cumulative_cache(BASE_DIR, d) for d in targets]
    cum_frames = [f for f in cum_frames if not f.empty]
    cum_df = pd.concat(cum_frames, ignore_index=True) if cum_frames else pd.DataFrame()

    # ── 평균 체류시간 캐시 로드 ───────────────────────────────────────────────
    dwell_frames = [load_dwell_cache(BASE_DIR, d) for d in targets]
    dwell_frames = [f for f in dwell_frames if not f.empty]
    dwell_df = aggregate_dwell(dwell_frames)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1: 누적 체류시간
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("누적 체류시간")
    st.caption(
        "각 시간대에 구역 내 존재한 기기 수 × 시간 = 총 이용량(person-hours).  \n"
        "예: 5명이 1분간 체류 → 5 person-min = 1/12 person-hour.  \n"
        "※ 시간 해상도: 하루 누적 / 1시간 / 30분 (30분 해상도는 해당 구간 내 누적값 — 전 구간 독립)"
    )

    if cum_df.empty:
        st.info("누적 체류시간 데이터를 준비 중입니다. `generate_movement_cache.py`를 먼저 실행해주세요.")
    else:
        # 시간해상도 선택
        res = st.radio(
            "시간 해상도",
            options=["하루 누적", "1시간", "30분"],
            horizontal=True,
            key="dw_resolution",
        )

        if res == "하루 누적":
            # KPI: 구역별 총 person-hours
            daily_agg = cum_df.groupby("zone")["mac_count"].sum().reset_index()
            daily_agg["person_hour"] = (daily_agg["mac_count"] / 60).round(1)
            cols = st.columns(len(daily_agg))
            for col, (_, row) in zip(cols, daily_agg.sort_values("zone").iterrows()):
                col.metric(ZONE_LABELS.get(row["zone"], row["zone"]), f"{row['person_hour']:,.0f} ph")
            st.plotly_chart(chart_cumulative_daily(cum_df, ZONE_LABELS), use_container_width=True)

        elif res == "1시간":
            st.plotly_chart(chart_cumulative_hourly(cum_df, ZONE_LABELS), use_container_width=True)

        else:  # 30분
            z_col, map_col = st.columns([2, 1])
            with z_col:
                sel_zone_cum = st.selectbox(
                    "구역 선택",
                    options=ALL_ZONES,
                    format_func=lambda z: ZONE_LABELS.get(z, z),
                    key="dw_cum_zone",
                )
            with map_col:
                show_map_cum = st.toggle("구역 위치 보기", key="dw_cum_map_toggle")

            if show_map_cum:
                swards_df = _load_swards()
                st.plotly_chart(
                    chart_zone_highlight(swards_df, sel_zone_cum),
                    use_container_width=True,
                )
            st.plotly_chart(
                chart_cumulative_30min(cum_df, sel_zone_cum, ZONE_LABELS),
                use_container_width=True,
            )

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2: 평균 체류시간 (기존 분석)
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("개인별 평균 체류시간")
    st.caption(
        "MAC별 지배 S-Ward를 추적하여 A~E 구역에서 **실제 머문 시간**을 계산합니다.  \n"
        "※ MAC 랜덤화(iOS ~15분, Android ~5분) 특성상 2분~2시간 범위 세션만 집계합니다."
    )

    if dwell_df.empty:
        st.info("체류시간 데이터를 준비 중입니다.")
        return

    # 날짜/구역 필터
    dw_col_a, dw_col_b, dw_col_c = st.columns([2, 2, 1])
    with dw_col_a:
        dw_zone = st.selectbox(
            "구역 선택",
            options=["전체"] + ALL_ZONES,
            format_func=lambda z: "전체" if z == "전체" else ZONE_LABELS.get(z, z),
            key="dw_zone",
        )
    with dw_col_b:
        dw_hour = st.selectbox(
            "시간대 필터",
            options=["전체"] + [f"{h:02d}시" for h in range(24)],
            key="dw_hour",
        )
    with dw_col_c:
        show_map_avg = st.toggle("구역 위치 보기", key="dw_avg_map_toggle")

    if show_map_avg:
        swards_df = _load_swards()
        hl_zone = dw_zone if dw_zone != "전체" else None
        st.plotly_chart(
            chart_zone_highlight(swards_df, hl_zone),
            use_container_width=True,
        )

    filt = dwell_df.copy()
    if dw_zone != "전체":
        filt = filt[filt["zone"] == dw_zone]
    if dw_hour != "전체":
        h = int(dw_hour[:2])
        filt = filt[filt["hour_start"] == h]

    if filt.empty:
        st.info("선택한 조건에 데이터가 없습니다.")
        return

    # KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("총 세션", f"{len(filt):,}건")
    k2.metric("평균 체류", f"{filt['dwell_s'].mean()/60:.1f}분")
    k3.metric("중앙값", f"{filt['dwell_s'].median()/60:.1f}분")
    k4.metric("최장 체류", f"{filt['dwell_s'].max()/60:.0f}분")

    dw_c1, dw_c2 = st.columns(2)
    with dw_c1:
        st.plotly_chart(chart_zone_dwell_bar(filt, ZONE_COLORS), use_container_width=True)
    with dw_c2:
        st.plotly_chart(chart_zone_dwell_box(filt, ZONE_COLORS), use_container_width=True)

    st.plotly_chart(chart_zone_dwell_heatmap(dwell_df, ZONE_COLORS), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    data = render_sidebar()

    if data is None:
        st.title(APP_ICON + " " + APP_TITLE)
        st.info(
            "사이드바에서 캐시를 빌드하거나, "
            "이미 캐시가 있다면 새로고침 후 분석을 시작하세요."
        )
        return

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "개요", "시간대 분석", "유입/유출", "구역별 분석",
        "동선 분석", "날씨 영향", "AI 인사이트",
        "🏃 이동 속도", "🕐 구역 체류시간",
    ])

    with tab1:
        render_overview(data)
    with tab2:
        render_hourly(data)
    with tab3:
        render_inflow(data)
    with tab4:
        render_zone(data)
    with tab5:
        render_flow_analysis(data)
    with tab6:
        render_weather(data)
    with tab7:
        render_ai(data)
    with tab8:
        render_movement(data)
    with tab9:
        render_zone_dwell(data)


# ── 비밀번호 게이트 ───────────────────────────────────────────────────────────

def _check_password() -> bool:
    """Streamlit Secrets의 password와 사용자 입력을 비교.

    secrets.toml (로컬) 또는 Streamlit Cloud Secrets에
        password = "YOUR_PASSWORD"
    를 설정해야 한다.
    """
    # secrets에 password 키가 없으면 비밀번호 보호 비활성화 (로컬 개발 편의)
    try:
        correct_pw = st.secrets["password"]
    except (KeyError, FileNotFoundError):
        return True  # secrets 미설정 시 바이패스

    # 이미 인증된 세션이면 즉시 통과
    if st.session_state.get("authenticated"):
        return True

    # 비밀번호 입력 UI
    st.title(APP_ICON + " " + APP_TITLE)
    st.markdown("### 🔒 접속 비밀번호를 입력하세요")
    pw = st.text_input("비밀번호", type="password", key="_pw_input")
    if st.button("확인", key="_pw_btn"):
        if pw == correct_pw:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("비밀번호가 틀렸습니다.")
    return False


# Streamlit은 __name__ == "__main__" 블록을 실행하지 않음 → 모듈 레벨에서 직접 호출
if _check_password():
    main()
