"""Festival_Gunhangje — Plotly 차트 빌더 모듈.

모든 차트 함수는 go.Figure를 반환한다.
Entrance_Analysis_Y1/src/charts.py 패턴 차용.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from pathlib import Path

from config import (
    ALL_ZONES, DOW_KO, FESTIVAL_END, FESTIVAL_START,
    SWARD_TO_ZONE, ZONE_COLORS, ZONE_LABELS, WEATHER_COLORS, WEEKEND_DAYS,
)

# ── 공통 테마 ────────────────────────────────────────────────────────────────

BG_COLOR   = "#0e1117"
GRID_COLOR = "#1e2130"

CHART_LAYOUT = dict(
    template     = "plotly_dark",
    paper_bgcolor = BG_COLOR,
    plot_bgcolor  = BG_COLOR,
    font          = dict(family="sans-serif", color="#ccd6f6", size=12),
    margin        = dict(l=55, r=20, t=50, b=50),
    height        = 420,
)

# 날짜별 오버레이 팔레트 — 시인성 높은 7색 (밝기·색조 모두 다양)
FESTIVAL_PALETTE = [
    "#e5604a",  # 붉은 오렌지
    "#4e9ff5",  # 밝은 파랑
    "#f0c040",  # 노랑
    "#59a14f",  # 초록
    "#b07aa1",  # 보라
    "#76b7b2",  # 청록
    "#f28e2b",  # 주황
]
# 비축제 날짜용 회색 계열 (구분 강조)
PRE_PALETTE = ["#aaaaaa", "#777777"]

def _base(height: int = 420, **kw) -> dict:
    """기본 레이아웃 딕셔너리 + 오버라이드."""
    layout = {**CHART_LAYOUT, "height": height}
    layout.update(kw)
    return layout


def _is_festival(date_str: str) -> bool:
    return FESTIVAL_START <= date_str <= FESTIVAL_END


def _date_color_map(dates: list[str]) -> dict[str, str]:
    """날짜 리스트 → 색상 딕셔너리 (축제/비축제 팔레트 분리)."""
    result = {}
    fi = 0
    pi = 0
    for d in sorted(dates):
        if _is_festival(d):
            result[d] = FESTIVAL_PALETTE[fi % len(FESTIVAL_PALETTE)]
            fi += 1
        else:
            result[d] = PRE_PALETTE[pi % len(PRE_PALETTE)]
            pi += 1
    return result


def _date_label(date_str: str, weather_df: pd.DataFrame) -> str:
    """날짜 → '03/27(금) ☀️' 형식."""
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
    emoji = row.iloc[0].get("weather_emoji", "")
    return f"{base} {emoji}"


# ══════════════════════════════════════════════════════════════════════════════
# 탭 1 — 개요
# ══════════════════════════════════════════════════════════════════════════════

def chart_daily_trend(daily_df: pd.DataFrame, weather_df: pd.DataFrame) -> go.Figure:
    """일별 Device Count 트렌드 차트 (raw, 연인원 근사 / 요일 비교용)."""
    df = daily_df.sort_values("date").copy()
    df["label"]    = df["date"].apply(lambda d: _date_label(d, weather_df))
    df["festival"] = df["date"].apply(_is_festival)

    fig = go.Figure()

    for is_fest, color, name in [(False, "#4a90d9", "비축제"), (True, "#e5604a", "축제기간")]:
        sub = df[df["festival"] == is_fest]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x          = sub["label"],
            y          = sub["udc"],
            name       = name,
            marker_color = color,
            opacity    = 0.85,
            hovertemplate = "<b>%{x}</b><br>Device Count: %{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(**_base(
        title = "일별 Device Count 트렌드 (연인원 근사)",
        xaxis_title = "날짜",
        yaxis_title = "Device Count",
        legend      = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    ))
    return fig


def chart_dow_avg(daily_df: pd.DataFrame) -> go.Figure:
    """요일별 평균 Device Count 바 차트 (raw, 요일 비교용)."""
    df = daily_df.copy()
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["dow_label"] = df["dow"].map(lambda i: DOW_KO[i] + "요일")
    dow_avg = df.groupby(["dow", "dow_label"])["udc"].mean().reset_index()
    dow_avg = dow_avg.sort_values("dow")
    colors  = ["#e5604a" if d in WEEKEND_DAYS else "#4a90d9" for d in dow_avg["dow"]]

    fig = go.Figure(go.Bar(
        x             = dow_avg["dow_label"],
        y             = dow_avg["udc"],
        marker_color  = colors,
        hovertemplate = "%{x}<br>평균 Device Count: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_base(
        title       = "요일별 평균 Device Count",
        xaxis_title = "요일",
        yaxis_title = "Device Count",
        height      = 380,
    ))
    return fig


def chart_weather_udc_box(daily_df: pd.DataFrame, weather_df: pd.DataFrame) -> go.Figure:
    """날씨별 Device Count 분포 박스플롯."""
    if weather_df.empty:
        return go.Figure()
    merged = daily_df.merge(weather_df[["date", "weather", "weather_color"]], on="date", how="left")
    merged["weather"] = merged["weather"].fillna("Unknown")

    fig = go.Figure()
    for w_type in merged["weather"].unique():
        sub = merged[merged["weather"] == w_type]
        color = WEATHER_COLORS.get(w_type, "#888888")
        fig.add_trace(go.Box(
            y             = sub["udc"],
            name          = w_type,
            marker_color  = color,
            boxpoints     = "all",
            jitter        = 0.3,
            pointpos      = -1.8,
            hovertemplate = f"{w_type}<br>Device Count: %{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(**_base(
        title       = "날씨 유형별 Device Count 분포",
        yaxis_title = "Device Count",
        height      = 380,
    ))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 탭 2 — 시간대 분석
# ══════════════════════════════════════════════════════════════════════════════

def chart_hourly_overlay(
    fine_5min_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    selected_dates: list[str],
) -> go.Figure:
    """날짜별 색상 구분 5분 단위 보정 인원 오버레이 (최대 7일)."""
    dates     = selected_dates[:7]
    color_map = _date_color_map(dates)

    fig = go.Figure()
    for date in sorted(dates):
        sub = (
            fine_5min_df[fine_5min_df["date"] == date]
            .sort_values("bin_5min")
            .copy()
        )
        if sub.empty:
            continue
        sub["time_frac"] = sub["hour"] + sub["minute"] / 60.0
        label = _date_label(date, weather_df)
        color = color_map.get(date, "#888888")
        fig.add_trace(go.Scatter(
            x    = sub["time_frac"].tolist(),
            y    = sub["corrected_dc"].tolist(),
            name = label,
            mode = "lines",
            line = dict(color=color, width=2),
            hovertemplate = f"<b>{label}</b><br>%{{x:.2f}}시: %{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(**_base(
        title       = "시간대별 DC 오버레이 (5분 단위, 날짜별 비교)",
        xaxis_title = "시간",
        yaxis_title = "Device Count",
        xaxis       = dict(tickmode="linear", dtick=1, range=[-0.1, 24.1],
                           ticktext=[f"{h:02d}시" for h in range(25)],
                           tickvals=list(range(25))),
        legend      = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height      = 450,
    ))
    return fig


def chart_ios_android_30min_bar(
    fine_5min_df: pd.DataFrame,
    date_str: str,
    weather_df: pd.DataFrame,
) -> go.Figure:
    """단일 날짜 30분 단위 iOS/Android 누적 막대 차트.

    - 아래: Android (초록 #59a14f)
    - 위:  iOS (파랑 #4e9ff5)
    - 텍스트: 각 막대의 iOS 비율 (%)
    """
    sub = fine_5min_df[fine_5min_df["date"] == date_str].copy()
    if sub.empty:
        return go.Figure()

    # 30분 윈도우로 집계 (bin_5min // 6)
    sub["bin_30"] = sub["bin_5min"] // 6
    agg = sub.groupby("bin_30")[["ios_dc", "android_dc"]].sum().reset_index()

    # 시간 레이블
    agg["time_min"] = agg["bin_30"] * 30
    agg["time_label"] = agg["time_min"].apply(
        lambda m: f"{int(m) // 60:02d}:{int(m) % 60:02d}"
    )
    total = agg["ios_dc"] + agg["android_dc"]
    agg["ios_pct"] = np.where(total > 0, agg["ios_dc"] / total * 100, 0.0)

    label = _date_label(date_str, weather_df)

    fig = go.Figure()

    # Android (하단)
    fig.add_trace(go.Bar(
        x             = agg["time_label"].tolist(),
        y             = agg["android_dc"].tolist(),
        name          = "Android",
        marker_color  = "#59a14f",
        opacity       = 0.88,
        hovertemplate = "%{x}<br>Android DC: %{y:,.0f}<extra></extra>",
    ))

    # iOS (상단) + iOS% 텍스트 어노테이션
    fig.add_trace(go.Bar(
        x             = agg["time_label"].tolist(),
        y             = agg["ios_dc"].tolist(),
        name          = "iOS",
        marker_color  = "#4e9ff5",
        opacity       = 0.88,
        text          = agg["ios_pct"].apply(lambda v: f"{v:.0f}%" if v > 0 else ""),
        textposition  = "inside",
        insidetextanchor = "middle",
        textfont      = dict(size=9, color="white"),
        hovertemplate = "%{x}<br>iOS DC: %{y:,.0f}<br>iOS 비율: %{text}<extra></extra>",
    ))

    # X축 tick 간격 조정 (너무 많으면 4개마다)
    n_bins = len(agg)
    tick_every = max(1, n_bins // 12)
    tick_labels = [lbl if i % tick_every == 0 else "" for i, lbl in enumerate(agg["time_label"].tolist())]

    fig.update_layout(**_base(
        title       = f"{label} — 30분 단위 iOS/Android Device Count",
        xaxis_title = "시간",
        yaxis_title = "Device Count",
        barmode     = "stack",
        xaxis       = dict(tickmode="array",
                           tickvals=agg["time_label"].tolist(),
                           ticktext=tick_labels),
        legend      = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height      = 380,
    ))
    return fig


def chart_dow_hour_heatmap(fine_5min_df: pd.DataFrame) -> go.Figure:
    """요일 × 30분 단위 평균 Device Count 히트맵 (fine_5min_df 기반)."""
    df = fine_5min_df.copy()
    df["dow"]         = pd.to_datetime(df["date"]).dt.dayofweek
    df["dow_label"]   = df["dow"].map(lambda i: DOW_KO[i] + "요일")
    # bin_5min // 6 → 30분 윈도우 (0~47)
    df["bin_30"]      = df["bin_5min"] // 6
    df["total_min"]   = df["bin_30"] * 30
    df["time_label"]  = df["total_min"].apply(
        lambda m: f"{int(m) // 60:02d}:{int(m) % 60:02d}"
    )

    pivot = df.groupby(["dow_label", "time_label"])["corrected_dc"].mean().unstack(fill_value=0)
    # 요일 순서 정렬
    row_order = [DOW_KO[i] + "요일" for i in range(7) if DOW_KO[i] + "요일" in pivot.index]
    pivot = pivot.reindex(row_order)
    # X축 시간 순서 보장
    col_order = sorted(pivot.columns, key=lambda s: int(s[:2]) * 60 + int(s[3:]))
    pivot = pivot[col_order]

    fig = go.Figure(go.Heatmap(
        z             = pivot.values,
        x             = pivot.columns.tolist(),
        y             = pivot.index.tolist(),
        colorscale    = "YlOrRd",
        colorbar      = dict(title="평균 DC"),
        hovertemplate = "요일: %{y}<br>시간: %{x}<br>평균 DC: %{z:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_base(
        title  = "요일 × 시간(30분) 평균 Device Count 히트맵",
        height = 380,
    ))
    fig.update_yaxes(autorange="reversed")
    return fig


def chart_peak_hour_table(hourly_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """날짜별 피크 시간 요약 DataFrame (표 렌더링용)."""
    rows = []
    for date, grp in hourly_df.groupby("date"):
        peak_row = grp.loc[grp["dc"].idxmax()]
        label    = _date_label(date, weather_df)
        rows.append({
            "날짜"      : label,
            "피크 시간"  : f"{int(peak_row['hour']):02d}시",
            "피크 DC"   : f"{int(peak_row['dc']):,}",
            "일 총 DC"  : f"{int(grp['dc'].sum()):,}",
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 탭 3 — 유입/유출
# ══════════════════════════════════════════════════════════════════════════════

def chart_net_inflow_fine(
    fine_5min_df: pd.DataFrame,
    date_str: str,
    weather_df: pd.DataFrame,
    resolution_min: int = 30,
) -> go.Figure:
    """2-패널 콤보 차트.

    - 상단(60%): Device Count Area 차트 — 혼잡 레벨 한눈에 파악
    - 하단(40%): 순유입 바 (Δ) — 증가=초록, 감소=빨강
    - X축 공유 (Zoom 연동)
    """
    sub = fine_5min_df[fine_5min_df["date"] == date_str].sort_values("bin_5min").copy()
    if sub.empty:
        return go.Figure()

    label = _date_label(date_str, weather_df)
    bins_per_window = resolution_min // 5
    sub["window"] = sub["bin_5min"] // bins_per_window

    agg = (
        sub.groupby("window")["corrected_dc"]
        .mean()
        .reset_index()
        .rename(columns={"corrected_dc": "occupancy"})
    )
    agg["total_min"]  = agg["window"] * resolution_min
    agg["time_label"] = agg["total_min"].apply(
        lambda m: f"{int(m)//60:02d}:{int(m)%60:02d}"
    )
    agg["delta"]  = agg["occupancy"].diff().fillna(0).round(0)
    agg["colors"] = agg["delta"].apply(
        lambda v: "rgba(89,161,79,0.85)" if v >= 0 else "rgba(229,96,74,0.85)"
    )

    xs    = agg["time_label"].tolist()
    occ   = agg["occupancy"].tolist()
    delta = agg["delta"].tolist()
    cols  = agg["colors"].tolist()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.58, 0.42],
        vertical_spacing=0.06,
        subplot_titles=("Device Count", f"순유입/유출 ({resolution_min}분 단위)"),
    )

    # ── [상단] Device Count Area 차트 ───────────────────────────────────────
    # 반투명 채움 영역
    fig.add_trace(go.Scatter(
        x         = xs,
        y         = occ,
        name      = "Device Count",
        mode      = "lines",
        line      = dict(color="#4e9ff5", width=2.5, shape="spline", smoothing=0.4),
        fill      = "tozeroy",
        fillcolor = "rgba(78,159,245,0.18)",
        hovertemplate = "%{x}  Device Count: %{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    # 피크 포인트 강조 마커
    peak_idx = int(np.argmax(occ))
    fig.add_trace(go.Scatter(
        x    = [xs[peak_idx]],
        y    = [occ[peak_idx]],
        name = f"피크 {occ[peak_idx]:,.0f}",
        mode = "markers+text",
        marker = dict(color="#FFD700", size=10, symbol="diamond",
                      line=dict(color="white", width=1.5)),
        text = [f"  {occ[peak_idx]:,.0f}"],
        textposition = "middle right",
        textfont = dict(color="#FFD700", size=11),
        hovertemplate = "피크: %{y:,.0f}<extra></extra>",
        showlegend = False,
    ), row=1, col=1)

    # ── [하단] 순유입 바 차트 ────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x             = xs,
        y             = delta,
        name          = "순유입",
        marker_color  = cols,
        hovertemplate = "%{x}  순유입: %{y:+,.0f}<extra></extra>",
    ), row=2, col=1)

    # 하단 zero line 강조
    fig.add_hline(y=0, line=dict(color="#666", width=1, dash="dot"), row=2, col=1)

    # ── 레이아웃 ─────────────────────────────────────────────────────────────
    base = _base(
        title  = f"{label} — {resolution_min}분 단위 인원 변화",
        height = 540,
    )
    fig.update_layout(**base)
    fig.update_layout(
        showlegend  = True,
        legend      = dict(orientation="h", yanchor="bottom", y=1.02,
                           xanchor="right", x=1, font=dict(size=11)),
        barmode     = "relative",
    )
    # Y축 설정
    fig.update_yaxes(title_text="Device Count", row=1, col=1,
                     showgrid=True, gridcolor="#2a2d3e", rangemode="tozero")
    y_max = max(abs(agg["delta"].max()), abs(agg["delta"].min())) * 1.35 or 1
    fig.update_yaxes(title_text="순유입 (Δ인원)", row=2, col=1,
                     zeroline=True, zerolinecolor="#555",
                     range=[-y_max, y_max],
                     showgrid=True, gridcolor="#2a2d3e")
    # X축: 하단만 라벨 표시
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=2, col=1)
    # subplot 타이틀 색상 조정
    for ann in fig.layout.annotations:
        ann.font.color = "#aaaaaa"
        ann.font.size  = 12

    return fig


def chart_inflow_heatmap_fine(
    fine_5min_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    resolution_min: int = 30,
) -> go.Figure:
    """전 기간 Device Count 히트맵 (날짜 × 시간, fine_5min 기반).

    각 셀 = 해당 날짜·시간대의 평균 Device Count
    """
    df = fine_5min_df.copy()
    bins_per_window = resolution_min // 5
    df["window"] = df["bin_5min"] // bins_per_window
    df["total_min"] = df["window"] * resolution_min
    df["time_label"] = df["total_min"].apply(
        lambda m: f"{int(m)//60:02d}:{int(m)%60:02d}"
    )

    pivot = df.pivot_table(
        index="date", columns="time_label", values="corrected_dc",
        aggfunc="mean", fill_value=0,
    ).sort_index()

    # 시간 순서 보장
    col_order = sorted(pivot.columns, key=lambda s: int(s[:2]) * 60 + int(s[3:]))
    pivot = pivot[col_order]

    y_labels = [_date_label(d, weather_df) for d in pivot.index]

    fig = go.Figure(go.Heatmap(
        z           = pivot.values,
        x           = pivot.columns.tolist(),
        y           = y_labels,
        colorscale  = "OrRd",
        colorbar    = dict(title="Device Count"),
        hovertemplate = "날짜: %{y}<br>시간: %{x}<br>Device Count: %{z:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_base(
        title  = f"전 기간 Device Count 히트맵 (날짜 × {resolution_min}분)",
        height = max(420, len(pivot) * 32),
    ))
    fig.update_yaxes(autorange="reversed")
    return fig


def chart_inflow_outflow_bar(
    inflow_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    selected_date: str,
) -> go.Figure:
    """선택 날짜 유입(+)/유출(-) 이중 막대 + 순유입 라인."""
    sub = inflow_df[inflow_df["date"] == selected_date].sort_values("hour")
    if sub.empty:
        return go.Figure()

    label  = _date_label(selected_date, weather_df)
    net    = sub["inflow"] - sub["outflow"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x             = sub["hour"],
        y             = sub["inflow"],
        name          = "유입",
        marker_color  = "#50c878",
        hovertemplate = "%{x:02d}시 유입: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x             = sub["hour"],
        y             = -sub["outflow"],
        name          = "유출",
        marker_color  = "#e15759",
        hovertemplate = "%{x:02d}시 유출: %{customdata:,.0f}<extra></extra>",
        customdata    = sub["outflow"],
    ))
    fig.add_trace(go.Scatter(
        x             = sub["hour"],
        y             = net,
        mode          = "lines+markers",
        name          = "순유입",
        line          = dict(color="#f0c040", width=2.5),
        marker        = dict(size=5),
        hovertemplate = "%{x:02d}시 순유입: %{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(**_base(
        title       = f"{label} 유입/유출 현황",
        xaxis_title = "시간",
        yaxis_title = "기기 수",
        barmode     = "overlay",
        xaxis       = dict(tickmode="linear", dtick=1, range=[-0.5, 23.5]),
        legend      = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height      = 420,
    ))
    return fig


def chart_cumulative_occupancy(
    inflow_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    selected_date: str,
) -> go.Figure:
    """누적 Device Count 추정 (유입 누적 - 유출 누적)."""
    sub = inflow_df[inflow_df["date"] == selected_date].sort_values("hour")
    if sub.empty:
        return go.Figure()

    label    = _date_label(selected_date, weather_df)
    cum_in   = sub["inflow"].cumsum()
    cum_out  = sub["outflow"].cumsum()
    occupancy = (cum_in - cum_out).clip(lower=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x    = sub["hour"].tolist(),
        y    = occupancy.tolist(),
        mode = "lines",
        name = "Device Count",
        fill = "tozeroy",
        fillcolor = "rgba(229,96,74,0.2)",
        line = dict(color="#e5604a", width=2.5),
        hovertemplate = "%{x:02d}시 Device Count: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_base(
        title       = f"{label} 누적 Device Count 추정",
        xaxis_title = "시간",
        yaxis_title = "Device Count",
        xaxis       = dict(tickmode="linear", dtick=1, range=[-0.5, 23.5]),
        height      = 380,
    ))
    return fig


def chart_inflow_heatmap_all(inflow_df: pd.DataFrame, weather_df: pd.DataFrame) -> go.Figure:
    """전 기간 유입 히트맵 (날짜 × 시간)."""
    pivot = inflow_df.pivot_table(
        index="date", columns="hour", values="inflow", aggfunc="sum", fill_value=0
    ).sort_index()

    y_labels = [_date_label(d, weather_df) for d in pivot.index]

    fig = go.Figure(go.Heatmap(
        z           = pivot.values,
        x           = [f"{h:02d}시" for h in pivot.columns],
        y           = y_labels,
        colorscale  = "OrRd",
        colorbar    = dict(title="유입"),
        hovertemplate = "날짜: %{y}<br>시간: %{x}<br>유입: %{z:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_base(
        title  = "전 기간 유입 히트맵 (날짜 × 시간)",
        height = max(420, len(pivot) * 32),
    ))
    fig.update_yaxes(autorange="reversed")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 탭 4 — 구역별 분석
# ══════════════════════════════════════════════════════════════════════════════

def chart_zone_pie(zone_hourly_df: pd.DataFrame, selected_date: str) -> go.Figure:
    """선택 날짜 구역별 DC 파이 차트."""
    sub = zone_hourly_df[zone_hourly_df["date"] == selected_date]
    if sub.empty:
        return go.Figure()

    zone_total = sub.groupby("zone")["dc"].sum().reset_index()
    zone_total["label"] = zone_total["zone"].map(ZONE_LABELS).fillna(zone_total["zone"])
    zone_total["color"] = zone_total["zone"].map(ZONE_COLORS)

    fig = go.Figure(go.Pie(
        labels           = zone_total["label"],
        values           = zone_total["dc"],
        marker_colors    = zone_total["color"].tolist(),
        hole             = 0.38,
        textinfo         = "label+percent",
        hovertemplate    = "%{label}<br>DC: %{value:,.0f} (%{percent})<extra></extra>",
    ))
    fig.update_layout(**_base(
        title  = f"{selected_date} 구역별 DC 비율",
        height = 400,
    ))
    return fig


def chart_zone_bar_daily(zone_hourly_df: pd.DataFrame, weather_df: pd.DataFrame) -> go.Figure:
    """구역별 일별 DC 스택 바 차트."""
    df = zone_hourly_df.groupby(["date", "zone"])["dc"].sum().reset_index()
    df["label"] = df["date"].apply(lambda d: _date_label(d, weather_df))

    fig = go.Figure()
    for zone in ALL_ZONES + ["미분류"]:
        sub = df[df["zone"] == zone]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x             = sub["label"],
            y             = sub["dc"],
            name          = ZONE_LABELS.get(zone, zone),
            marker_color  = ZONE_COLORS.get(zone, "#888888"),
            hovertemplate = f"<b>{ZONE_LABELS.get(zone, zone)}</b><br>%{{x}}<br>DC: %{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(**_base(
        title       = "구역별 일별 DC 추이",
        xaxis_title = "날짜",
        yaxis_title = "Device Count",
        barmode     = "stack",
        legend      = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height      = 430,
    ))
    return fig


def chart_zone_hourly_stacked(
    zone_hourly_df: pd.DataFrame,
    selected_date: str,
) -> go.Figure:
    """선택 날짜 시간별 구역 누적 바."""
    sub = zone_hourly_df[zone_hourly_df["date"] == selected_date]
    all_hours = list(range(24))

    fig = go.Figure()
    for zone in ALL_ZONES + ["미분류"]:
        z_sub = sub[sub["zone"] == zone][["hour", "dc"]].set_index("hour")
        y_vals = [int(z_sub.loc[h, "dc"]) if h in z_sub.index else 0 for h in all_hours]
        if sum(y_vals) == 0:
            continue
        fig.add_trace(go.Bar(
            x             = all_hours,
            y             = y_vals,
            name          = ZONE_LABELS.get(zone, zone),
            marker_color  = ZONE_COLORS.get(zone, "#888888"),
            hovertemplate = f"<b>{ZONE_LABELS.get(zone, zone)}</b><br>%{{x:02d}}시: %{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(**_base(
        title       = f"{selected_date} 시간별 구역 누적 DC",
        xaxis_title = "시간",
        yaxis_title = "Device Count",
        barmode     = "stack",
        xaxis       = dict(tickmode="linear", dtick=1, range=[-0.5, 23.5]),
        legend      = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height      = 420,
    ))
    return fig


def chart_zone_map(swards_df: pd.DataFrame, zone_hourly_df: pd.DataFrame, selected_date: str) -> go.Figure:
    """S-Ward 좌표 + Ground.png 배경 지도 오버레이."""
    import base64
    from pathlib import Path

    BASE_DIR  = Path(__file__).parent
    img_path  = BASE_DIR / "Data" / "Ground.png"

    # 날짜별 구역 집계
    day_zone = zone_hourly_df[zone_hourly_df["date"] == selected_date].groupby("zone")["dc"].sum().to_dict()

    fig = go.Figure()

    # Ground.png 배경 (존재 시)
    if img_path.exists():
        import PIL.Image
        img = PIL.Image.open(img_path)
        w, h = img.size
        fig.add_layout_image(dict(
            source   = img,
            xref     = "x", yref = "y",
            x        = 0,   y    = 0,
            sizex    = w,   sizey = h,
            sizing   = "stretch",
            opacity  = 0.45,
            layer    = "below",
        ))
        fig.update_xaxes(range=[0, w], showgrid=False, zeroline=False)
        fig.update_yaxes(range=[h, 0], showgrid=False, zeroline=False, scaleanchor="x")
    else:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False, autorange="reversed")

    # sward 포인트
    if not swards_df.empty:
        # 유효 좌표만 (0,0 제외)
        valid = swards_df[(swards_df["x"] > 0) | (swards_df["y"] > 0)].copy()
        valid["zone"] = valid["name"].astype(str).map(
            lambda n: next((z for z, sw in __import__("config").ZONE_SWARDS.items() if n in sw), "미분류")
        )
        for zone, grp in valid.groupby("zone"):
            color = ZONE_COLORS.get(zone, "#888888")
            dc_val = day_zone.get(zone, 0)
            fig.add_trace(go.Scatter(
                x             = grp["x"],
                y             = grp["y"],
                mode          = "markers",
                name          = ZONE_LABELS.get(zone, zone),
                marker        = dict(
                    color   = color,
                    size    = 12,
                    opacity = 0.85,
                    line    = dict(color="white", width=1),
                ),
                text          = grp["name"],
                hovertemplate = (
                    f"<b>{ZONE_LABELS.get(zone, zone)}</b><br>"
                    "S-Ward: %{text}<br>"
                    f"구역 DC: {dc_val:,}<extra></extra>"
                ),
            ))

    fig.update_layout(**_base(
        title  = f"{selected_date} S-Ward 구역 지도",
        height = 600,
        paper_bgcolor = BG_COLOR,
        plot_bgcolor  = BG_COLOR,
    ))
    return fig


def chart_zone_map_with_slider(
    zone_hourly_df: pd.DataFrame,
    swards_df: pd.DataFrame,
    selected_date: str,
    selected_hour: int,
    base_dir: Path,
) -> go.Figure:
    """시간 슬라이더용 S-Ward 지도 — 선택 날짜·시간의 구역 DC를 원 크기/색상으로 표시.

    원 크기 = zone_dc / 전체 zone 평균 DC × BASE_MARKER_SIZE
    색상    = ZONE_COLORS
    hover   = S-Ward 이름, 구역, DC
    """
    IMG_W, IMG_H   = 3319, 6599
    BASE_MARKER    = 18   # 평균 DC 기준 원 크기(px)
    MIN_MARKER     = 6    # 최소 원 크기
    MAX_MARKER     = 40   # 최대 원 크기

    img_path = base_dir / "Data" / "Ground.png"

    # 선택 날짜·시간 구역 DC 집계
    mask     = (zone_hourly_df["date"] == selected_date) & (zone_hourly_df["hour"] == selected_hour)
    hour_sub = zone_hourly_df[mask]
    zone_dc: dict[str, float] = {}
    if not hour_sub.empty:
        zone_dc = hour_sub.groupby("zone")["dc"].sum().to_dict()

    # 평균 DC (0 방지)
    values = [v for v in zone_dc.values() if v > 0]
    avg_dc = float(np.mean(values)) if values else 1.0

    fig = go.Figure()

    # Ground.png 배경
    if img_path.exists():
        import PIL.Image
        img = PIL.Image.open(img_path)
        fig.add_layout_image(dict(
            source   = img,
            xref     = "x", yref = "y",
            x        = 0,   y    = 0,
            sizex    = IMG_W, sizey = IMG_H,
            sizing   = "stretch",
            opacity  = 0.45,
            layer    = "below",
        ))
        fig.update_xaxes(range=[0, IMG_W], showgrid=False, zeroline=False)
        fig.update_yaxes(range=[IMG_H, 0], showgrid=False, zeroline=False, scaleanchor="x")
    else:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False, autorange="reversed")

    if not swards_df.empty:
        valid = swards_df[(swards_df["x"] > 0) | (swards_df["y"] > 0)].copy()
        # sward → zone 매핑
        valid["zone"] = valid["name"].astype(str).map(
            lambda n: SWARD_TO_ZONE.get(n, "미분류")
        )

        for zone, grp in valid.groupby("zone"):
            color  = ZONE_COLORS.get(zone, "#888888")
            dc_val = zone_dc.get(zone, 0)
            # 원 크기 비례 계산
            ratio  = (dc_val / avg_dc) if avg_dc > 0 else 0.0
            size   = float(np.clip(BASE_MARKER * ratio, MIN_MARKER, MAX_MARKER))

            # y축 반전: Ground.png 원점이 좌상단 → Plotly y=0이 상단
            y_coords = IMG_H - grp["y"].values

            fig.add_trace(go.Scatter(
                x    = grp["x"].tolist(),
                y    = y_coords.tolist(),
                mode = "markers",
                name = ZONE_LABELS.get(zone, zone),
                marker = dict(
                    color   = color,
                    size    = size,
                    opacity = 0.85,
                    line    = dict(color="white", width=1),
                ),
                text = grp["name"].tolist(),
                customdata = [dc_val] * len(grp),
                hovertemplate = (
                    "<b>%{text}</b><br>"
                    f"구역: {ZONE_LABELS.get(zone, zone)}<br>"
                    "구역 DC: %{customdata:,}<extra></extra>"
                ),
            ))

    fig.update_layout(**_base(
        title         = f"{selected_date}  {selected_hour:02d}시 S-Ward 트래픽 지도",
        height        = 650,
        paper_bgcolor = BG_COLOR,
        plot_bgcolor  = BG_COLOR,
    ))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 탭 5 — 날씨 영향
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# AST / 5분 윈도우 차트
# ══════════════════════════════════════════════════════════════════════════════

def chart_daily_ast(daily_ast_df: pd.DataFrame, weather_df: pd.DataFrame) -> go.Figure:
    """일별 누적 체류시간 막대 차트.

    X축: 날짜(날씨 레이블), Y축: 누적 체류시간(시간 단위)
    축제기간 = 빨간, 비축제 = 파란
    """
    df = daily_ast_df.sort_values("date").copy()
    df["label"]    = df["date"].apply(lambda d: _date_label(d, weather_df))
    df["festival"] = df["date"].apply(_is_festival)
    df["ast_h"]    = df["ast_hours"]   # 이미 시간 단위

    fig = go.Figure()
    for is_fest, color, name in [(False, "#4a90d9", "비축제"), (True, "#e5604a", "축제기간")]:
        sub = df[df["festival"] == is_fest]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x             = sub["label"],
            y             = sub["ast_h"],
            name          = name,
            marker_color  = color,
            opacity       = 0.85,
            hovertemplate = "<b>%{x}</b><br>누적 체류시간: %{y:,.1f}시간<extra></extra>",
        ))

    fig.update_layout(**_base(
        title       = "일별 누적 체류시간 (시간 단위)",
        xaxis_title = "날짜",
        yaxis_title = "누적 체류시간 (시간)",
        barmode     = "overlay",
        legend      = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    ))
    return fig


def chart_fine_5min(
    fine_5min_df: pd.DataFrame,
    date_str: str,
    weather_df: pd.DataFrame,
) -> go.Figure:
    """5분 윈도우 Device Count 라인 차트.

    X축: 연속 시간(0~24), Y축: corrected_dc
    """
    sub = fine_5min_df[fine_5min_df["date"] == date_str].sort_values("bin_5min").copy()
    if sub.empty:
        return go.Figure()

    label = _date_label(date_str, weather_df)
    sub["time_frac"] = sub["hour"] + sub["minute"] / 60.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x    = sub["time_frac"].tolist(),
        y    = sub["corrected_dc"].tolist(),
        mode = "lines",
        name = "Device Count",
        line = dict(color="#e5604a", width=2),
        fill = "tozeroy",
        fillcolor = "rgba(229,96,74,0.12)",
        hovertemplate = "%{x:.2f}시: %{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(**_base(
        title       = f"{label} — 5분 단위 Device Count",
        xaxis_title = "시간",
        yaxis_title = "Device Count",
        xaxis       = dict(tickmode="linear", dtick=1, range=[-0.1, 24.1],
                           ticktext=[f"{h:02d}시" for h in range(25)],
                           tickvals=list(range(25))),
        height      = 380,
    ))
    return fig


def chart_hourly_ast(hourly_ast_df: pd.DataFrame, date_str: str) -> go.Figure:
    """시간별 누적 체류시간 막대 차트.

    X축: 0~23시, Y축: 누적 체류시간(분 단위)
    """
    sub = hourly_ast_df[hourly_ast_df["date"] == date_str].sort_values("hour").copy()
    if sub.empty:
        return go.Figure()

    all_hours = list(range(24))
    sub_idx   = sub.set_index("hour")
    y_vals    = [float(sub_idx.loc[h, "ast_minutes"]) if h in sub_idx.index else 0.0
                 for h in all_hours]

    fig = go.Figure(go.Bar(
        x             = all_hours,
        y             = y_vals,
        marker_color  = "#f28e2b",
        hovertemplate = "%{x:02d}시 누적 체류시간: %{y:,.0f}분<extra></extra>",
    ))
    fig.update_layout(**_base(
        title       = f"{date_str} — 시간별 누적 체류시간",
        xaxis_title = "시간",
        yaxis_title = "누적 체류시간 (분)",
        xaxis       = dict(tickmode="linear", dtick=1, range=[-0.5, 23.5]),
        height      = 360,
    ))
    return fig


def chart_cumulative_ast_overlay(
    hourly_ast_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    dates: list[str],
) -> go.Figure:
    """날짜별 0→24시 누적 체류시간 비교 오버레이.

    - X축: 0~23시
    - Y축: 해당 시간까지의 누적합 (분 단위)
    - 날짜별 한 줄씩 — 선택된 날짜를 겹쳐서 비교
    - 축제기간은 밝은 색, 비축제기간은 회색 계열
    - 피크(최종값) 날짜를 굵게 강조
    """
    all_hours = list(range(24))
    fig = go.Figure()

    if hourly_ast_df.empty or not dates:
        return fig

    # 날짜별 색상 팔레트 (DC 오버레이와 동일한 방식)
    PALETTE = [
        "#e15759", "#f28e2b", "#59a14f", "#4e9ff5",
        "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    ]

    max_final = 0.0  # 전체 최대값 (Y축 범위용)

    for i, date_str in enumerate(dates):
        sub = hourly_ast_df[hourly_ast_df["date"] == date_str].sort_values("hour")
        if sub.empty:
            continue

        sub_idx = sub.set_index("hour")
        hourly  = [float(sub_idx.loc[h, "ast_minutes"]) if h in sub_idx.index else 0.0
                   for h in all_hours]
        cumsum  = list(np.cumsum(hourly))  # 0시부터 해당 시간까지 누적합

        final_val = cumsum[-1]
        max_final = max(max_final, final_val)

        is_festival = FESTIVAL_START <= date_str <= FESTIVAL_END
        color = PALETTE[i % len(PALETTE)]
        label = _date_label(date_str, weather_df)

        fig.add_trace(go.Scatter(
            x    = all_hours,
            y    = cumsum,
            name = label,
            mode = "lines",
            line = dict(
                color  = color,
                width  = 2.5 if is_festival else 1.5,
                dash   = "solid" if is_festival else "dot",
            ),
            opacity = 0.95 if is_festival else 0.65,
            hovertemplate = (
                f"<b>{label}</b><br>"
                "%{x:02d}시까지 누적: %{y:,.0f}분<extra></extra>"
            ),
        ))

        # 최종값 레이블 (오른쪽 끝)
        fig.add_trace(go.Scatter(
            x    = [23],
            y    = [cumsum[-1]],
            mode = "markers+text",
            showlegend = False,
            marker = dict(color=color, size=7),
            text = [f" {cumsum[-1]:,.0f}분"],
            textposition = "middle right",
            textfont = dict(color=color, size=10),
            hoverinfo = "skip",
        ))

    # 축제 기간 배경 음영 (0~24 중 축제 날짜 라인들의 배경 강조는 X축 영역이라 의미 없으므로 생략)
    fig.update_layout(**_base(
        title       = "날짜별 누적 체류시간 — 0시→24시 누적 비교",
        xaxis_title = "시간",
        yaxis_title = "누적 체류시간 (분)",
        xaxis       = dict(
            tickmode = "linear",
            dtick    = 1,
            range    = [-0.3, 24.5],
            tickvals = list(range(0, 24, 2)),
            ticktext = [f"{h:02d}시" for h in range(0, 24, 2)],
        ),
        yaxis  = dict(rangemode="tozero"),
        legend = dict(orientation="h", yanchor="bottom", y=1.02,
                      xanchor="right", x=1),
        height = 360,
    ))
    return fig


def chart_weather_scatter_temp(daily_df: pd.DataFrame, weather_df: pd.DataFrame) -> go.Figure:
    """기온(최고) vs Device Count 산점도."""
    if weather_df.empty:
        return go.Figure()
    merged = daily_df.merge(
        weather_df[["date", "temp_max", "temp_min", "weather", "weather_color", "weather_emoji"]],
        on="date", how="left",
    )
    merged["label"] = merged["date"].apply(lambda d: _date_label(d, weather_df))

    fig = go.Figure()
    for w_type in merged["weather"].dropna().unique():
        sub = merged[merged["weather"] == w_type].copy()
        color = WEATHER_COLORS.get(w_type, "#888888")
        fig.add_trace(go.Scatter(
            x             = sub["temp_max"],
            y             = sub["udc"],
            mode          = "markers+text",
            name          = w_type,
            marker        = dict(color=color, size=11, opacity=0.85),
            text          = sub["label"],
            textposition  = "top center",
            textfont      = dict(size=9),
            hovertemplate = "<b>%{text}</b><br>최고기온: %{x:.1f}°C<br>Device Count: %{y:,.0f}<extra></extra>",
        ))
    fig.update_layout(**_base(
        title       = "최고기온 vs Device Count 산점도",
        xaxis_title = "최고기온 (°C)",
        yaxis_title = "Device Count",
        height      = 400,
    ))
    return fig


def chart_weather_scatter_precip(daily_df: pd.DataFrame, weather_df: pd.DataFrame) -> go.Figure:
    """강수량 vs Device Count 산점도."""
    if weather_df.empty:
        return go.Figure()
    merged = daily_df.merge(
        weather_df[["date", "precipitation", "weather", "weather_color"]],
        on="date", how="left",
    )
    merged["label"] = merged["date"].apply(lambda d: _date_label(d, weather_df))

    fig = go.Figure()
    for w_type in merged["weather"].dropna().unique():
        sub = merged[merged["weather"] == w_type].copy()
        color = WEATHER_COLORS.get(w_type, "#888888")
        fig.add_trace(go.Scatter(
            x             = sub["precipitation"],
            y             = sub["udc"],
            mode          = "markers+text",
            name          = w_type,
            marker        = dict(color=color, size=11, opacity=0.85),
            text          = sub["label"],
            textposition  = "top center",
            textfont      = dict(size=9),
            hovertemplate = "<b>%{text}</b><br>강수량: %{x:.1f}mm<br>Device Count: %{y:,.0f}<extra></extra>",
        ))
    fig.update_layout(**_base(
        title       = "강수량 vs Device Count 산점도",
        xaxis_title = "강수량 (mm)",
        yaxis_title = "Device Count",
        height      = 400,
    ))
    return fig


def chart_weather_hourly_pattern(
    hourly_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> go.Figure:
    """날씨 유형별 평균 시간대 DC 패턴."""
    if weather_df.empty:
        return go.Figure()
    merged = hourly_df.merge(weather_df[["date", "weather"]], on="date", how="left")
    merged["weather"] = merged["weather"].fillna("Unknown")

    fig = go.Figure()
    for w_type in merged["weather"].unique():
        sub = merged[merged["weather"] == w_type]
        avg = sub.groupby("hour")["dc"].mean().reset_index()
        color = WEATHER_COLORS.get(w_type, "#888888")
        fig.add_trace(go.Scatter(
            x             = avg["hour"],
            y             = avg["dc"],
            mode          = "lines+markers",
            name          = w_type,
            line          = dict(color=color, width=2.5),
            marker        = dict(size=6),
            hovertemplate = f"<b>{w_type}</b><br>%{{x:02d}}시: %{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(**_base(
        title       = "날씨 유형별 평균 시간대 DC 패턴",
        xaxis_title = "시간",
        yaxis_title = "평균 DC",
        xaxis       = dict(tickmode="linear", dtick=1, range=[-0.5, 23.5]),
        legend      = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height      = 420,
    ))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# S-Ward 시간별 히트맵 지도
# ══════════════════════════════════════════════════════════════════════════════

def chart_sward_heatmap_slider(
    sward_hourly_df: pd.DataFrame,
    swards_df: pd.DataFrame,
    date_str: str,
    hour: int,
    base_dir: Path,
    cumulative: bool = False,
) -> go.Figure:
    """S-Ward 시간별 트래픽 Heatmap (Gaussian blur 오버레이).

    cumulative=False : 선택 시간대 단독 DC (혼잡도)
    cumulative=True  : 0시 ~ 선택 시간까지 DC 합산 (누적 방문 강도)
    """
    from scipy.ndimage import gaussian_filter
    import PIL.Image

    IMG_W, IMG_H = 3319, 6599
    SCALE = 0.25

    img_path = base_dir / "Data" / "Ground.png"

    # ── DC 집계 (시간별 or 누적) ──────────────────────────────────────────────
    date_mask = sward_hourly_df["date"] == date_str
    if cumulative:
        sub = sward_hourly_df[date_mask & (sward_hourly_df["hour"] <= hour)]
        sward_dc: dict[str, float] = (
            sub.groupby("sward")["dc"].sum().to_dict() if not sub.empty else {}
        )
    else:
        sub = sward_hourly_df[date_mask & (sward_hourly_df["hour"] == hour)]
        sward_dc = (
            sub.set_index("sward")["dc"].to_dict() if not sub.empty else {}
        )

    fig = go.Figure()

    # ── 배경 이미지 (어둡게 — 히트맵 대비 강조) ──────────────────────────────
    if img_path.exists():
        bg_img = PIL.Image.open(img_path)
        fig.add_layout_image(dict(
            source=bg_img, xref="x", yref="y",
            x=0, y=0, sizex=IMG_W, sizey=IMG_H,
            sizing="stretch", opacity=0.18, layer="below",
        ))

    # scaleanchor="x" + 충분한 height → aspect ratio 유지
    # IMG_W:IMG_H = 3319:6599 ≈ 1:2  →  컨테이너 폭 ~950px 기준 height ≈ 1900
    fig.update_xaxes(range=[0, IMG_W], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[IMG_H, 0], showgrid=False, zeroline=False,
                     visible=False, scaleanchor="x", scaleratio=1)

    # ── Gaussian blur 히트맵 — 2-레이어 ──────────────────────────────────────
    if not swards_df.empty:
        valid = swards_df[(swards_df["x"] > 0) | (swards_df["y"] > 0)].copy()
        valid["dc"] = valid["name"].map(
            lambda n: float(sward_dc.get(str(int(float(n))), 0))
        )
        active = valid[valid["dc"] > 0]

        if not active.empty:
            w = max(1, int(IMG_W * SCALE))
            h = max(1, int(IMG_H * SCALE))
            intensity = np.zeros((h, w), dtype=np.float32)

            # ── raw DC값으로 intensity 구성 ──────────────────────────────────
            for _, row in active.iterrows():
                px = int(np.clip(row["x"] * SCALE, 0, w - 1))
                py = int(np.clip(row["y"] * SCALE, 0, h - 1))
                intensity[py, px] += float(row["dc"])

            # ── 모드별 global_max 계산 (날짜가 달라도 동일 기준) ─────────────────
            if cumulative:
                # 전체 데이터에서 S-Ward 하루 누적 DC 최대값
                global_max = float(
                    sward_hourly_df.groupby(["date", "sward"])["dc"].sum().max()
                ) or 1.0
                # 현재 선택 날짜·시간의 누적 최대값
                cur_max = float(
                    sward_hourly_df[
                        (sward_hourly_df["date"] == date_str) &
                        (sward_hourly_df["hour"] <= hour)
                    ].groupby("sward")["dc"].sum().max()
                ) or 1.0
            else:
                # 전체 데이터에서 단일 시간 S-Ward DC 최대값
                global_max = float(sward_hourly_df["dc"].max()) or 1.0
                # 현재 선택 시간의 최대값
                cur_max = float(active["dc"].max()) or 1.0

            # 상대 스케일 (0~1): 현재 데이터가 global_max 대비 얼마나 강한가
            # MIN_SCALE 없음 — 조용한 시간대(새벽 등)는 자연스럽게 투명해야 함
            scale = float(np.clip(cur_max / global_max, 0.0, 1.0))

            # ── 표준 히트맵 컬러맵: 투명 → 초록 → 노랑 → 주황 → 빨강 ──────────
            def _heat_rgba(norm: np.ndarray, alpha_boost: float = 1.0) -> np.ndarray:
                """norm ∈ [0,1] → RGBA uint8.

                - 색상: norm 값에 따라 초록→빨강 전환
                - 알파: norm^0.55 — 작은 scale이면 자연스럽게 투명
                  * 4am  (scale≈0.02, peak norm=0.02) → alpha≈11% → 거의 투명
                  * 20h  (scale≈0.90, peak norm=0.90) → alpha≈88% → 선명
                """
                sv = [0.00, 0.25, 0.50, 0.75, 1.00]
                sr = [  20,  210,  255,  255,  210]
                sg = [200,  230,  150,   50,    0]
                sb = [  0,    0,    0,    0,    0]
                r = np.interp(norm, sv, sr).astype(np.uint8)
                g = np.interp(norm, sv, sg).astype(np.uint8)
                b = np.interp(norm, sv, sb).astype(np.uint8)
                # norm^0.55: 중간 감마 — 활발한 시간대는 임팩트 있게,
                #             조용한 시간대는 자연스럽게 투명
                a = np.where(
                    norm < 0.003,
                    np.uint8(0),
                    np.clip(
                        np.power(norm.astype(np.float32), 0.55) * 235 * alpha_boost,
                        0, 235,
                    ).astype(np.uint8),
                )
                return np.stack([r, g, b, a], axis=-1)

            def _blur_and_scale(arr: np.ndarray, sigma_px: float) -> np.ndarray:
                """blur → [0,1] 정규화(가시성) → scale 적용(날짜간 비교)."""
                blurred = gaussian_filter(arr, sigma=sigma_px)
                b_max   = blurred.max()
                if b_max <= 0:
                    return blurred
                return np.clip((blurred / b_max) * scale, 0.0, 1.0)

            # Layer 1 — 외곽 Glow (넓은 블러, 색감 넓게 퍼뜨림)
            glow = _blur_and_scale(intensity, sigma_px=130 * SCALE)
            rgba_glow = _heat_rgba(glow, alpha_boost=0.72)
            pil_glow  = PIL.Image.fromarray(rgba_glow, "RGBA")
            pil_glow  = pil_glow.resize((IMG_W, IMG_H), PIL.Image.LANCZOS)
            fig.add_layout_image(dict(
                source=pil_glow, xref="x", yref="y",
                x=0, y=0, sizex=IMG_W, sizey=IMG_H,
                sizing="stretch", opacity=0.85, layer="above",
            ))

            # Layer 2 — 핫 코어 (좁은 블러, 고밀도 영역 선명하게)
            core = _blur_and_scale(intensity, sigma_px=48 * SCALE)
            rgba_core = _heat_rgba(core, alpha_boost=1.0)
            pil_core  = PIL.Image.fromarray(rgba_core, "RGBA")
            pil_core  = pil_core.resize((IMG_W, IMG_H), PIL.Image.LANCZOS)
            fig.add_layout_image(dict(
                source=pil_core, xref="x", yref="y",
                x=0, y=0, sizex=IMG_W, sizey=IMG_H,
                sizing="stretch", opacity=0.97, layer="above",
            ))

            # ── 투명 Hover 마커 ──────────────────────────────────────────────
            fig.add_trace(go.Scatter(
                x    = active["x"].tolist(),
                y    = active["y"].tolist(),
                mode = "markers",
                name = "S-Ward",
                marker = dict(size=16, opacity=0.0, color="rgba(0,0,0,0)"),
                text       = active["name"].tolist(),
                customdata = active["dc"].tolist(),
                hovertemplate = (
                    "<b>%{text}</b><br>"
                    f"시간: {hour:02d}시<br>"
                    "Device Count: %{customdata:,.0f}<extra></extra>"
                ),
            ))

    # height = 컨테이너 폭(약 950px) × (IMG_H/IMG_W) ≈ 1900
    # scaleanchor가 width를 자동 조정하므로, height가 충분히 크면 전체 지도가 표시됨
    fig.update_layout(**_base(
        title         = (
            f"{date_str}  00~{hour:02d}시 누적 Heatmap"
            if cumulative else
            f"{date_str}  {hour:02d}시 — 트래픽 Heatmap"
        ),
        height        = 1900,
        paper_bgcolor = "#0a0a0f",
        plot_bgcolor  = "#0a0a0f",
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.45) -> str:
    """'#rrggbb' → 'rgba(r,g,b,alpha)' 변환."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _zone_centroids(swards_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """swards_df에서 구역별 중심 좌표 계산."""
    valid = swards_df[(swards_df["x"] > 0) | (swards_df["y"] > 0)].copy()
    valid["name_str"] = valid["name"].apply(lambda n: str(int(float(n))))
    valid["zone"]     = valid["name_str"].map(SWARD_TO_ZONE).fillna("미분류")
    result: dict[str, tuple[float, float]] = {}
    for zone, grp in valid[valid["zone"] != "미분류"].groupby("zone"):
        result[zone] = (float(grp["x"].mean()), float(grp["y"].mean()))
    return result


def _zone_flow_for_hour(
    flow_df: pd.DataFrame,
    date_str: str,
    hour: int,
) -> pd.DataFrame:
    """flow_df → 해당 날짜·시간의 구역 간 이동량 집계 (self-loop·미분류 제거)."""
    bin_lo = hour * 2
    bin_hi = hour * 2 + 1
    mask = (
        (flow_df["date"] == date_str) &
        (flow_df["bin_30"].between(bin_lo, bin_hi))
    )
    sub = flow_df[mask].copy()
    if sub.empty:
        return pd.DataFrame(columns=["from_zone", "to_zone", "count"])

    sub["from_zone"] = sub["from_sward"].map(SWARD_TO_ZONE).fillna("미분류")
    sub["to_zone"]   = sub["to_sward"].map(SWARD_TO_ZONE).fillna("미분류")
    sub = sub[
        (sub["from_zone"] != "미분류") &
        (sub["to_zone"]   != "미분류") &
        (sub["from_zone"] != sub["to_zone"])
    ]
    if sub.empty:
        return pd.DataFrame(columns=["from_zone", "to_zone", "count"])

    return (
        sub.groupby(["from_zone", "to_zone"])["count"]
        .sum()
        .reset_index()
        .sort_values("count", ascending=False)
    )


def chart_flow_sankey(
    flow_df: pd.DataFrame,
    date_str: str,
    hour: int,
) -> go.Figure:
    """구역 간 이동 흐름 Sankey 다이어그램.

    - 노드: A~E구역 (5개)
    - 링크: 구역 간 이동 횟수, 두께 = count
    - 색상: ZONE_COLORS
    """
    zone_flow = _zone_flow_for_hour(flow_df, date_str, hour)
    ZONES = ALL_ZONES  # ['A구역','B구역','C구역','D구역','E구역']

    def _empty_fig(msg: str) -> go.Figure:
        f = go.Figure()
        f.update_layout(
            title=msg, template="plotly_dark",
            paper_bgcolor=BG_COLOR, font=dict(color="#ccd6f6"),
            margin=dict(l=30, r=30, t=50, b=30), height=380,
        )
        return f

    if zone_flow.empty:
        return _empty_fig(f"{date_str} {hour:02d}시 — 구역 간 이동 (데이터 없음)")

    zone_idx   = {z: i for i, z in enumerate(ZONES)}
    sources    = [zone_idx[r["from_zone"]] for _, r in zone_flow.iterrows()
                  if r["from_zone"] in zone_idx and r["to_zone"] in zone_idx]
    targets    = [zone_idx[r["to_zone"]]   for _, r in zone_flow.iterrows()
                  if r["from_zone"] in zone_idx and r["to_zone"] in zone_idx]
    values_cnt = [int(r["count"])           for _, r in zone_flow.iterrows()
                  if r["from_zone"] in zone_idx and r["to_zone"] in zone_idx]

    if not sources:
        return _empty_fig(f"{date_str} {hour:02d}시 — 구역 간 이동 (데이터 없음)")

    node_colors = [ZONE_COLORS.get(z, "#888888") for z in ZONES]
    node_labels = [ZONE_LABELS.get(z, z)         for z in ZONES]
    link_colors = [_hex_to_rgba(ZONE_COLORS.get(ZONES[s], "#888888"), 0.42) for s in sources]

    fig = go.Figure(go.Sankey(
        arrangement = "snap",
        textfont    = dict(color="#ccd6f6", size=12),
        node = dict(
            pad       = 22,
            thickness = 22,
            line      = dict(color="rgba(255,255,255,0.15)", width=0.5),
            label     = node_labels,
            color     = node_colors,
        ),
        link = dict(
            source = sources,
            target = targets,
            value  = values_cnt,
            color  = link_colors,
        ),
    ))
    fig.update_layout(
        title         = f"{date_str}  {hour:02d}시 — 구역 간 이동 흐름",
        template      = "plotly_dark",
        paper_bgcolor = BG_COLOR,
        font          = dict(color="#ccd6f6"),
        margin        = dict(l=30, r=30, t=55, b=30),
        height        = 400,
    )
    return fig


def chart_flow_zone_map(
    flow_df: pd.DataFrame,
    swards_df: pd.DataFrame,
    date_str: str,
    hour: int,
    base_dir: Path,
) -> go.Figure:
    """구역 중심점 기반 이동 동선 지도.

    - 배경: Ground.png (opacity 0.4)
    - 화살표: 구역 중심점 간 (5개 구역 → 최대 20쌍)
    - 굵기·색상: 이동 횟수 비례 (YlOrRd)
    - 좌표: raw y, range=[IMG_H,0]
    """
    import PIL.Image

    IMG_W, IMG_H = 3319, 6599
    img_path = base_dir / "Data" / "Ground.png"
    ARROW_COLORS = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]

    fig = go.Figure()
    if img_path.exists():
        bg = PIL.Image.open(img_path)
        fig.add_layout_image(dict(
            source=bg, xref="x", yref="y",
            x=0, y=0, sizex=IMG_W, sizey=IMG_H,
            sizing="stretch", opacity=0.4, layer="below",
        ))
    fig.update_xaxes(range=[0, IMG_W], showgrid=False, zeroline=False)
    fig.update_yaxes(range=[IMG_H, 0], showgrid=False, zeroline=False, scaleanchor="x")

    centroids  = _zone_centroids(swards_df)
    zone_flow  = _zone_flow_for_hour(flow_df, date_str, hour)
    zone_flow  = zone_flow[
        zone_flow["from_zone"].isin(centroids) &
        zone_flow["to_zone"].isin(centroids)
    ]

    # ── 구역 중심점 마커 ─────────────────────────────────────────────────────
    for zone, (cx, cy) in centroids.items():
        color = ZONE_COLORS.get(zone, "#888888")
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode="markers+text",
            showlegend=False,
            hoverinfo="skip",
            marker=dict(
                symbol="circle",
                size=30,
                color=color,
                opacity=0.65,
                line=dict(color="white", width=2),
            ),
            text=[ZONE_LABELS.get(zone, zone)],
            textposition="top center",
            textfont=dict(color="white", size=11, family="Arial Black"),
        ))

    if zone_flow.empty:
        fig.update_layout(**_base(
            title=f"{date_str}  {hour:02d}시 — 구역 간 이동 지도 (이동 없음)",
            height=1520, paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        ))
        return fig

    max_cnt = float(zone_flow["count"].max())

    for _, row in zone_flow.iterrows():
        x1, y1 = centroids[row["from_zone"]]
        x2, y2 = centroids[row["to_zone"]]
        norm   = float(row["count"]) / max_cnt
        lw     = float(np.clip(norm * 14 + 3, 3, 17))
        a_sz   = float(np.clip(norm * 35 + 14, 14, 50))
        c_idx  = int(np.clip(norm * (len(ARROW_COLORS) - 1), 0, len(ARROW_COLORS) - 1))
        color  = ARROW_COLORS[c_idx]
        angle  = float(np.degrees(np.arctan2(x2 - x1, -(y2 - y1))))
        hover  = (
            f"<b>{ZONE_LABELS.get(row['from_zone'], row['from_zone'])} "
            f"→ {ZONE_LABELS.get(row['to_zone'], row['to_zone'])}</b><br>"
            f"이동 횟수: {int(row['count']):,}"
        )
        # 선
        fig.add_trace(go.Scatter(
            x=[x1, x2], y=[y1, y2],
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
            line=dict(color=color, width=lw),
            opacity=0.70,
        ))
        # 화살표 머리
        fig.add_trace(go.Scatter(
            x=[x2], y=[y2],
            mode="markers",
            showlegend=False,
            marker=dict(
                symbol="arrow-wide",
                size=a_sz,
                angle=angle,
                color=color,
                opacity=0.92,
                line=dict(color="rgba(255,255,255,0.6)", width=0.8),
            ),
            text=[hover],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(**_base(
        title  = f"{date_str}  {hour:02d}시 — 구역 간 이동 동선",
        height = 1520,
        paper_bgcolor = BG_COLOR,
        plot_bgcolor  = BG_COLOR,
    ))
    return fig


# ── (레거시 호환) ──────────────────────────────────────────────────────────────
def chart_flow_arrows(
    flow_df: pd.DataFrame,
    swards_df: pd.DataFrame,
    date_str: str,
    hour: int,
    base_dir: Path,
    top_n: int = 25,
) -> go.Figure:
    """30분 단위 주요 동선 화살표 지도.

    - 배경: Ground.png
    - 화살표: Scatter lines + arrow-wide marker (symbol="arrow-wide")
    - 굵기·색상: count 비례 (YlOrRd)
    - 좌표: raw y (range=[IMG_H,0]로 반전 처리)

    각도 계산:
        Plotly marker.angle: 0=북(위), 시계방향 양수.
        range=[IMG_H,0]에서 y_data 증가 = 화면 아래
        → angle = degrees(arctan2(dx, -dy))
    """
    import PIL.Image

    IMG_W, IMG_H = 3319, 6599
    img_path = base_dir / "Data" / "Ground.png"

    # ── 배경 설정 ──────────────────────────────────────────────────────────────
    fig = go.Figure()
    if img_path.exists():
        bg = PIL.Image.open(img_path)
        fig.add_layout_image(dict(
            source=bg, xref="x", yref="y",
            x=0, y=0, sizex=IMG_W, sizey=IMG_H,
            sizing="stretch", opacity=0.5, layer="below",
        ))
    fig.update_xaxes(range=[0, IMG_W], showgrid=False, zeroline=False)
    fig.update_yaxes(range=[IMG_H, 0], showgrid=False, zeroline=False, scaleanchor="x")

    # ── 빈 상태 처리 ──────────────────────────────────────────────────────────
    def _empty_title():
        fig.update_layout(**_base(
            title=f"{date_str}  {hour:02d}시 — 주요 동선 (데이터 없음)",
            height=1300, paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        ))
        return fig

    if swards_df.empty or flow_df.empty:
        return _empty_title()

    # ── S-Ward 좌표 dict (string key, float→int 변환으로 ".0" 제거) ────────────
    coord: dict[str, tuple[float, float]] = {}
    for _, r in swards_df.iterrows():
        if float(r["x"]) > 0 or float(r["y"]) > 0:
            # swards.csv name이 float64로 읽힐 때 "27041299.0" → "27041299"
            key = str(int(float(r["name"])))
            coord[key] = (float(r["x"]), float(r["y"]))

    # ── 해당 날짜·시간 필터 ───────────────────────────────────────────────────
    # bin_30 = time_index // 180  →  hour H = bin_30 2H, 2H+1
    bin_lo = hour * 2
    bin_hi = hour * 2 + 1
    mask = (
        (flow_df["date"] == date_str) &
        (flow_df["bin_30"].between(bin_lo, bin_hi))
    )
    sub = (
        flow_df[mask]
        .groupby(["from_sward", "to_sward"])["count"]
        .sum()
        .reset_index()
    )
    if sub.empty:
        return _empty_title()

    sub = sub[sub["from_sward"].isin(coord) & sub["to_sward"].isin(coord)]
    sub = sub.sort_values("count", ascending=False).head(top_n).reset_index(drop=True)
    if sub.empty:
        return _empty_title()

    # ── 시각화 준비 ───────────────────────────────────────────────────────────
    max_cnt = float(sub["count"].max()) if sub["count"].max() > 0 else 1.0
    # YlOrRd 5단계 색상 (하위→상위)
    COLORS = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]

    # 라인 데이터 (None 구분자로 멀티-세그먼트)
    line_x: list = []
    line_y: list = []
    # 화살표 머리 데이터
    head_x: list = []
    head_y: list = []
    head_angle: list = []
    head_size: list = []
    head_color: list = []
    hover_texts: list = []

    for _, row in sub.iterrows():
        x1, y1 = coord[row["from_sward"]]
        x2, y2 = coord[row["to_sward"]]
        if x1 == x2 and y1 == y2:
            continue  # 같은 위치 skip

        norm  = float(row["count"]) / max_cnt
        c_idx = int(np.clip(norm * (len(COLORS) - 1), 0, len(COLORS) - 1))
        color = COLORS[c_idx]
        lw    = float(np.clip(norm * 7 + 1.5, 1.5, 9.0))
        a_sz  = float(np.clip(norm * 18 + 8, 8, 28))

        # 화살표 방향 각도 (range=[IMG_H,0] 기준)
        dx = x2 - x1
        dy = y2 - y1  # positive = 화면 아래
        angle = float(np.degrees(np.arctan2(dx, -dy)))  # Plotly: 0=북, 시계방향

        line_x += [x1, x2, None]
        line_y += [y1, y2, None]
        head_x.append(x2)
        head_y.append(y2)
        head_angle.append(angle)
        head_size.append(a_sz)
        head_color.append(color)
        hover_texts.append(
            f"<b>{row['from_sward']} → {row['to_sward']}</b><br>"
            f"전환 횟수: {int(row['count']):,}"
        )

    if not head_x:
        return _empty_title()

    # ── 라인 트레이스 (굵기 평균 단색 — 빠른 렌더링) ─────────────────────────
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y,
        mode="lines",
        showlegend=False,
        hoverinfo="skip",
        line=dict(color="#fd8d3c", width=3),
        opacity=0.55,
    ))

    # ── 화살표 머리 — 개별 색상·크기·각도 ────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=head_x, y=head_y,
        mode="markers",
        showlegend=False,
        marker=dict(
            symbol="arrow-wide",
            size=head_size,
            angle=head_angle,
            color=head_color,
            opacity=0.85,
            line=dict(color="rgba(255,255,255,0.4)", width=0.5),
        ),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
    ))

    # ── 출발지 점 (반투명) ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=head_x, y=head_y,  # 도착지 위치 (시인성용)
        mode="markers",
        showlegend=False,
        hoverinfo="skip",
        marker=dict(symbol="circle", size=5, color="white", opacity=0.4),
    ))

    # ── 컬러바 더미 (범례) ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        name=f"상위 {len(head_x)}개 동선 (굵기=빈도)",
        marker=dict(
            color=[0, max_cnt],
            colorscale=[[0, COLORS[0]], [0.25, COLORS[1]], [0.5, COLORS[2]],
                        [0.75, COLORS[3]], [1.0, COLORS[4]]],
            showscale=True,
            colorbar=dict(title="전환 횟수", thickness=12, len=0.5),
            size=10,
        ),
    ))

    fig.update_layout(**_base(
        title  = f"{date_str}  {hour:02d}시 — 주요 이동 동선 (상위 {len(head_x)}개)",
        height = 1300,
        paper_bgcolor = BG_COLOR,
        plot_bgcolor  = BG_COLOR,
    ))
    return fig


# ── S-Ward 레벨 동선 흐름 지도 ─────────────────────────────────────────────────
def chart_flow_sward_map(
    flow_df: pd.DataFrame,
    swards_df: pd.DataFrame,
    date_str: str,
    hour: int,
    base_dir: Path,
    top_n: int = 50,
) -> go.Figure:
    """S-Ward 위치 기반 미세 동선 흐름 지도.

    - 배경: Ground.png (opacity 0.5)
    - S-Ward 점: 구역 색상으로 소형 원 (60개)
    - 화살표: S-Ward 간 이동 경로, 굵기·색상=빈도 비례 (작은 화살표)
    - top_n 개 전환만 표시 (기본 50개)
    """
    import PIL.Image
    from collections import defaultdict

    IMG_W, IMG_H = 3319, 6599
    img_path = base_dir / "Data" / "Ground.png"
    # YlOrRd 5단계 (저→고 빈도)
    COLORS = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]
    N_BUCKET = len(COLORS)

    fig = go.Figure()

    # ── 배경 이미지 ──────────────────────────────────────────────────────────
    if img_path.exists():
        bg = PIL.Image.open(img_path)
        fig.add_layout_image(dict(
            source=bg, xref="x", yref="y",
            x=0, y=0, sizex=IMG_W, sizey=IMG_H,
            sizing="stretch", opacity=0.5, layer="below",
        ))
    fig.update_xaxes(range=[0, IMG_W], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[IMG_H, 0], showgrid=False, zeroline=False,
                     scaleanchor="x", visible=False)

    def _empty():
        fig.update_layout(**_base(
            title=f"{date_str}  {hour:02d}시 — S-Ward 동선 (데이터 없음)",
            height=1520, paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        ))
        return fig

    if swards_df.empty or flow_df.empty:
        return _empty()

    # ── S-Ward 좌표 dict  (name float64 → "27041299" 정규화) ─────────────────
    coord: dict[str, tuple[float, float]] = {}
    sward_zone: dict[str, str] = {}
    for _, r in swards_df.iterrows():
        key = str(int(float(r["name"])))
        if float(r["x"]) > 0 or float(r["y"]) > 0:
            coord[key] = (float(r["x"]), float(r["y"]))
        sward_zone[key] = SWARD_TO_ZONE.get(key, "미분류")

    # ── 전체 S-Ward 위치 점 (구역 색상) ─────────────────────────────────────
    # 구역별로 묶어서 trace — 범례 표시
    from config import ALL_ZONES
    for zone in ALL_ZONES + ["미분류"]:
        zx, zy, zt = [], [], []
        for sw, (sx, sy) in coord.items():
            if sward_zone.get(sw) == zone:
                zx.append(sx)
                zy.append(sy)
                zt.append(sw)
        if not zx:
            continue
        fig.add_trace(go.Scatter(
            x=zx, y=zy,
            mode="markers",
            name=ZONE_LABELS.get(zone, zone),
            showlegend=True,
            marker=dict(
                symbol="circle",
                size=9,
                color=ZONE_COLORS.get(zone, "#888888"),
                opacity=0.70,
                line=dict(color="rgba(255,255,255,0.5)", width=1),
            ),
            text=zt,
            hovertemplate="S-Ward: %{text}<extra></extra>",
        ))

    # ── 해당 날짜·시간 전환 데이터 ───────────────────────────────────────────
    bin_lo, bin_hi = hour * 2, hour * 2 + 1
    mask = (
        (flow_df["date"] == date_str) &
        (flow_df["bin_30"].between(bin_lo, bin_hi))
    )
    sub = (
        flow_df[mask]
        .groupby(["from_sward", "to_sward"])["count"]
        .sum()
        .reset_index()
    )
    if sub.empty:
        return _empty()

    sub = (
        sub[sub["from_sward"].isin(coord) & sub["to_sward"].isin(coord)]
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    if sub.empty:
        return _empty()

    max_cnt = float(sub["count"].max()) or 1.0

    # ── 선 — 5개 색상 버킷으로 묶어서 렌더링 효율화 ──────────────────────────
    bucket_lines: dict[int, dict] = {i: {"x": [], "y": [], "w": []} for i in range(N_BUCKET)}
    # 화살표 머리 — 개별 속성 (각도, 크기, 색상)
    head_x, head_y, head_angle, head_size, head_color, head_hover = [], [], [], [], [], []

    for _, row in sub.iterrows():
        x1, y1 = coord[row["from_sward"]]
        x2, y2 = coord[row["to_sward"]]
        if x1 == x2 and y1 == y2:
            continue

        norm   = float(row["count"]) / max_cnt
        b_idx  = int(np.clip(norm * (N_BUCKET - 1), 0, N_BUCKET - 1))
        color  = COLORS[b_idx]
        lw     = float(np.clip(norm * 4.0 + 1.0, 1.0, 5.0))   # 1~5px (얇게)
        a_sz   = float(np.clip(norm * 10.0 + 6.0, 6.0, 16.0)) # 6~16px (작게)
        dx, dy = x2 - x1, y2 - y1
        angle  = float(np.degrees(np.arctan2(dx, -dy)))

        bk = bucket_lines[b_idx]
        bk["x"] += [x1, x2, None]
        bk["y"] += [y1, y2, None]
        bk["w"].append(lw)

        head_x.append(x2)
        head_y.append(y2)
        head_angle.append(angle)
        head_size.append(a_sz)
        head_color.append(color)
        head_hover.append(
            f"<b>{row['from_sward']} → {row['to_sward']}</b><br>"
            f"전환 횟수: {int(row['count']):,}"
        )

    if not head_x:
        return _empty()

    # ── 선 트레이스 (버킷별) ─────────────────────────────────────────────────
    for b_idx, bk in bucket_lines.items():
        if not bk["x"]:
            continue
        avg_w = float(np.mean(bk["w"])) if bk["w"] else 1.5
        fig.add_trace(go.Scatter(
            x=bk["x"], y=bk["y"],
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
            line=dict(color=COLORS[b_idx], width=avg_w),
            opacity=0.65,
        ))

    # ── 화살표 머리 트레이스 ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=head_x, y=head_y,
        mode="markers",
        showlegend=False,
        marker=dict(
            symbol="arrow-wide",
            size=head_size,
            angle=head_angle,
            color=head_color,
            opacity=0.92,
            line=dict(color="rgba(255,255,255,0.5)", width=0.8),
        ),
        text=head_hover,
        hovertemplate="%{text}<extra></extra>",
    ))

    # ── 컬러바 (범례용 더미 trace) ───────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        showlegend=False,
        marker=dict(
            color=[0, max_cnt],
            colorscale=[[i / (N_BUCKET - 1), c] for i, c in enumerate(COLORS)],
            showscale=True,
            cmin=0, cmax=int(max_cnt),
            colorbar=dict(
                title=dict(text="전환 횟수", side="right"),
                thickness=12, len=0.45,
                x=1.01,
            ),
            size=6,
        ),
    ))

    fig.update_layout(**_base(
        title  = f"{date_str}  {hour:02d}시 — S-Ward 이동 흐름 (상위 {len(head_x)}개)",
        height = 1520,
        paper_bgcolor = BG_COLOR,
        plot_bgcolor  = BG_COLOR,
    ))
    fig.update_layout(
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(30,30,30,0.7)",
            font=dict(size=10),
        )
    )
    return fig
