"""Plot helpers shared across pages (Plotly, warm palette, LTR-safe)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import MAX_PLOT_POINTS
from core.state import next_key
from core.theme import PALETTE, SEQUENCE


def plot(fig: go.Figure, height: int | None = None) -> None:
    """Render a Plotly figure with a stable unique key."""
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, key=next_key("fig"), config={"displaylogo": False})


def sample_for_plot(df: pd.DataFrame, n: int = MAX_PLOT_POINTS, seed: int = 0) -> tuple[pd.DataFrame, bool]:
    if len(df) <= n:
        return df, False
    return df.sample(n, random_state=seed), True


def hist(x: pd.Series, title: str = "", nbins: int = 40, color: str | None = None, name: str | None = None) -> go.Figure:
    fig = go.Figure(go.Histogram(x=x.dropna(), nbinsx=nbins, marker_color=color or PALETTE["coral"],
                                 name=name or str(x.name), opacity=0.85))
    fig.update_layout(title=title, bargap=0.04, showlegend=False)
    return fig


def overlay_hist(series: dict[str, pd.Series], title: str = "", nbins: int = 40, norm: str = "probability density") -> go.Figure:
    fig = go.Figure()
    for i, (name, s) in enumerate(series.items()):
        fig.add_trace(go.Histogram(x=s.dropna(), nbinsx=nbins, name=name, opacity=0.55, histnorm=norm,
                                   marker_color=SEQUENCE[i % len(SEQUENCE)]))
    fig.update_layout(barmode="overlay", title=title, legend=dict(orientation="h", y=1.1))
    return fig


def ecdf(series: dict[str, pd.Series], title: str = "ECDF") -> go.Figure:
    fig = go.Figure()
    for i, (name, s) in enumerate(series.items()):
        x = np.sort(s.dropna().to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(shape="hv", width=2.2,
                                                                              color=SEQUENCE[i % len(SEQUENCE)])))
    fig.update_layout(title=title, yaxis_title="F(x) = P(X ≤ x)", legend=dict(orientation="h", y=1.1))
    return fig


def box(df: pd.DataFrame, y: str, x: str | None = None, title: str = "", points: str | bool = "outliers") -> go.Figure:
    import plotly.express as px
    fig = px.box(df, x=x, y=y, points=points, color=x, color_discrete_sequence=SEQUENCE, title=title)
    fig.update_layout(showlegend=False)
    return fig


def heatmap(mat: pd.DataFrame, title: str = "", zmin: float = -1, zmax: float = 1, text: bool = True,
            colorscale=None) -> go.Figure:
    from core.theme import DIVERGING_SCALE
    fig = go.Figure(go.Heatmap(z=mat.to_numpy(), x=[str(c) for c in mat.columns], y=[str(i) for i in mat.index],
                               zmin=zmin, zmax=zmax, colorscale=colorscale or DIVERGING_SCALE,
                               text=np.round(mat.to_numpy(), 2) if text else None,
                               texttemplate="%{text}" if text else None, hoverongaps=False))
    fig.update_layout(title=title, yaxis=dict(autorange="reversed"))
    return fig


def missing_matrix(df: pd.DataFrame, title: str = "مصفوفة الفقد Missingness matrix", max_rows: int = 400) -> go.Figure:
    view = df if len(df) <= max_rows else df.iloc[np.linspace(0, len(df) - 1, max_rows).astype(int)]
    m = view.isna().astype(int)
    fig = go.Figure(go.Heatmap(z=m.to_numpy(), x=[str(c) for c in m.columns], y=list(range(len(m))),
                               colorscale=[[0, "#E7F5FF"], [1, PALETTE["coral_deep"]]], showscale=False,
                               hovertemplate="row %{y}<br>%{x}: %{z}<extra></extra>"))
    fig.update_layout(title=title, yaxis=dict(autorange="reversed", title="rows"), xaxis=dict(tickangle=-35))
    return fig


def add_animation_controls(fig: go.Figure, frame_names: Sequence[str], duration: int = 600,
                           prefix: str = "") -> go.Figure:
    """Play / Pause buttons and a step slider for a Plotly figure with frames."""
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=-0.12, xanchor="left", yanchor="top", showactive=False,
            pad=dict(r=8, t=8),
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True,
                                      transition=dict(duration=max(0, duration // 2)))]),
                dict(label="❚❚ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                        transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.18, y=-0.08, len=0.8, xanchor="left", currentvalue=dict(prefix=prefix, font=dict(size=13)),
            steps=[dict(label=str(n), method="animate",
                        args=[[n], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                        transition=dict(duration=0))]) for n in frame_names],
        )],
        margin=dict(b=110),
    )
    return fig
