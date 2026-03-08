from pathlib import Path
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

STYLE_PATH = Path(__file__).resolve().parent / "styles.css"
DEFAULT_API_BASE = os.getenv("SALES_API_BASE", "http://127.0.0.1:8000")


def _api_get(api_base: str, endpoint: str, params: dict | None = None) -> dict:
    response = requests.get(f"{api_base.rstrip('/')}{endpoint}", params=params, timeout=20)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def get_filters(api_base: str) -> dict:
    return _api_get(api_base, "/filters")


@st.cache_data(show_spinner=False)
def get_health(api_base: str) -> dict:
    return _api_get(api_base, "/health")


@st.cache_data(show_spinner=False)
def get_overview(api_base: str, years: tuple, regions: tuple, channels: tuple) -> dict:
    params = {
        "years": list(years),
        "regions": list(regions),
        "channels": list(channels),
    }
    return _api_get(api_base, "/overview", params=params)


@st.cache_data(show_spinner=False)
def get_distributed(api_base: str, batch_size: int, stream_cursor: int, lookback_batches: int) -> dict:
    params = {
        "batch_size": batch_size,
        "stream_cursor": stream_cursor,
        "lookback_batches": lookback_batches,
    }
    return _api_get(api_base, "/distributed", params=params)


def inject_styles() -> None:
    css = STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def metric_card(title: str, value: str) -> None:
    st.markdown(
        f"""
        <div class=\"metric-card\">
            <p class=\"metric-title\">{title}</p>
            <p class=\"metric-value\">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_dashboard(api_base: str) -> None:
    st.sidebar.title("Filters")
    filter_data = get_filters(api_base)
    year_values = filter_data.get("years", [])
    selected_years = st.sidebar.multiselect("Select Year", year_values, default=year_values)

    region_values = filter_data.get("regions", [])
    selected_regions = st.sidebar.multiselect("Select Region", region_values, default=region_values)

    channels = filter_data.get("channels", [])
    selected_channels = st.sidebar.multiselect("Sales Channel", channels, default=channels)

    payload = get_overview(
        api_base,
        tuple(selected_years),
        tuple(selected_regions),
        tuple(selected_channels),
    )

    if payload.get("empty", False):
        st.warning("No data matches the current filters. Adjust filters to continue.")
        return

    metrics = payload["metrics"]
    monthly = pd.DataFrame(payload["monthly"])
    top_items = pd.DataFrame(payload["top_items"])
    by_region = pd.DataFrame(payload["by_region"])
    preview = pd.DataFrame(payload["preview"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Revenue", f"${metrics['total_revenue']:,.0f}")
    with c2:
        metric_card("Total Profit", f"${metrics['total_profit']:,.0f}")
    with c3:
        metric_card("Average Margin", f"{metrics['avg_margin']:.2f}%")
    with c4:
        metric_card("Unique Orders", f"{metrics['order_count']:,}")

    st.markdown("<h3 class='section-title'>Revenue Trend by Month</h3>", unsafe_allow_html=True)
    fig_monthly = px.line(
        monthly,
        x="Month",
        y="Total Revenue",
        markers=True,
        color_discrete_sequence=["#2c7a4b"],
    )
    fig_monthly.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_monthly, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.markdown("<h3 class='section-title'>Top 10 Item Types by Profit</h3>", unsafe_allow_html=True)
        fig_items = px.bar(
            top_items,
            x="Total Profit",
            y="Item Type",
            orientation="h",
            color="Total Profit",
            color_continuous_scale=["#d5eadb", "#2c7a4b"],
        )
        fig_items.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_items, use_container_width=True)

    with right:
        st.markdown("<h3 class='section-title'>Profit by Region</h3>", unsafe_allow_html=True)
        fig_region = px.bar(
            by_region,
            x="Region",
            y="Total Profit",
            color="Total Profit",
            color_continuous_scale=["#e6f4ea", "#2f855a"],
        )
        fig_region.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
        fig_region.update_xaxes(tickangle=35)
        st.plotly_chart(fig_region, use_container_width=True)

    st.markdown("<h3 class='section-title'>Filtered Data Preview</h3>", unsafe_allow_html=True)
    st.dataframe(preview, use_container_width=True, hide_index=True)


def render_distributed_story(api_base: str, total_rows: int) -> None:
    st.sidebar.title("Streaming Controls")

    batch_size = st.sidebar.slider("Micro-batch size", min_value=5, max_value=120, value=30, step=5)
    max_cursor = max(batch_size, total_rows)
    default_cursor = min(max(batch_size * 4, batch_size), max_cursor)
    stream_cursor = st.sidebar.slider(
        "Stream position",
        min_value=batch_size,
        max_value=max_cursor,
        value=default_cursor,
        step=batch_size,
    )
    lookback_batches = st.sidebar.slider("Lookback batches", min_value=1, max_value=10, value=4, step=1)

    payload = get_distributed(api_base, batch_size, stream_cursor, lookback_batches)
    if payload.get("empty", False):
        st.warning("No stream data available for current controls.")
        return

    metrics = payload["metrics"]
    minute_revenue = pd.DataFrame(payload["minute_revenue"])
    by_country = pd.DataFrame(payload["by_country"])
    serving_table = pd.DataFrame(payload["serving_table"])

    st.markdown("<h3 class='section-title'>Distributed Architecture</h3>", unsafe_allow_html=True)
    st.graphviz_chart(
        """
        digraph G {
            rankdir=LR;
            node [shape=box style=filled fontname="Segoe UI" fontsize=11 color="#d4e8dc"];

            producer [label="Order Producers\n(API / CSV / Events)"];
            kafka [label="Kafka Topic\norder-events"];
            spark [label="Spark Structured Streaming\nValidation + Enrichment"];
            bronze [label="Delta Bronze\nRaw Append"];
            silver [label="Delta Silver\nCleaned + DQ Checks"];
            cassandra [label="Cassandra\nServing Tables"];
            dashboard [label="Streamlit Frontend\nRealtime KPIs"];

            producer -> kafka -> spark;
            spark -> bronze;
            bronze -> silver;
            silver -> cassandra;
            cassandra -> dashboard;
            silver -> dashboard;
        }
        """
    )

    st.caption(
        "Data moves through Kafka into Spark micro-batches, then lands in Delta layers before serving low-latency "
        "queries from Cassandra."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Events in Window", f"{metrics['events_ingested']:,}")
    with c2:
        metric_card("Current Batch Revenue", f"${metrics['batch_revenue']:,.0f}")
    with c3:
        metric_card("Current Batch Profit", f"${metrics['batch_profit']:,.0f}")
    with c4:
        metric_card("Estimated Throughput", f"{metrics['throughput']:.1f} rows/s")

    left, right = st.columns(2)
    with left:
        st.markdown("<h3 class='section-title'>Simulated Revenue Stream</h3>", unsafe_allow_html=True)
        fig_stream = px.area(
            minute_revenue,
            x="Event Minute",
            y="Total Revenue",
            color_discrete_sequence=["#2f855a"],
        )
        fig_stream.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_stream, use_container_width=True)

    with right:
        st.markdown("<h3 class='section-title'>Top Countries in Current Window</h3>", unsafe_allow_html=True)
        fig_country = px.bar(
            by_country,
            x="Country",
            y="Total Profit",
            color="Total Profit",
            color_continuous_scale=["#e6f4ea", "#2f855a"],
        )
        fig_country.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
        fig_country.update_xaxes(tickangle=35)
        st.plotly_chart(fig_country, use_container_width=True)

    st.markdown("<h3 class='section-title'>Cassandra Serving Table Preview</h3>", unsafe_allow_html=True)
    st.dataframe(serving_table, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Amazon Sales Frontend", page_icon=":bar_chart:", layout="wide")
    inject_styles()

    st.sidebar.title("API")
    api_base = st.sidebar.text_input("FastAPI base URL", value=DEFAULT_API_BASE).strip()

    try:
        health = get_health(api_base)
    except requests.RequestException:
        st.error(
            "Cannot reach FastAPI backend. Start API first: `uvicorn backend.main:app --reload` from project root."
        )
        st.stop()

    st.markdown(
        """
        <div class=\"hero\">
            <h1>Amazon Sales Analytics Dashboard</h1>
            <p>Interactive frontend for PySpark analysis and distributed-database storytelling.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Page",
        options=["Sales Overview", "Distributed DB Story"],
        index=0,
    )

    if page == "Sales Overview":
        render_overview_dashboard(api_base)
    else:
        render_distributed_story(api_base, int(health.get("rows", 0)))


if __name__ == "__main__":
    main()
