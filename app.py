from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent / "AmazonSalesData.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Margin %"] = ((df["Unit Price"] - df["Unit Cost"]) / df["Unit Price"] * 100).round(2)
    return df


def inject_styles():
    st.markdown("""
    <style>
    .metric-card { background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0; }
    .metric-title { color: #666; font-size: 14px; margin: 0; }
    .metric-value { color: #2c7a4b; font-size: 24px; font-weight: bold; margin: 5px 0 0 0; }
    .section-title { color: #2c7a4b; font-size: 18px; margin: 20px 0 10px 0; }
    .hero { background: linear-gradient(135deg, #2c7a4b 0%, #1a5c35 100%); padding: 40px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px; }
    .hero h1 { margin: 0; font-size: 32px; }
    .hero p { margin: 10px 0 0 0; opacity: 0.9; }
    </style>
    """, unsafe_allow_html=True)


def metric_card(title, value):
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-title">{title}</p>
        <p class="metric-value">{value}</p>
    </div>
    """, unsafe_allow_html=True)


def render_overview():
    df = load_data()

    st.sidebar.title("Filters")
    years = sorted(df["Year"].unique())
    selected_years = st.sidebar.multiselect("Select Year", years, default=years)

    regions = sorted(df["Region"].unique())
    selected_regions = st.sidebar.multiselect("Select Region", regions, default=regions)

    channels = sorted(df["Sales Channel"].unique())
    selected_channels = st.sidebar.multiselect("Sales Channel", channels, default=channels)

    filtered = df[
        (df["Year"].isin(selected_years)) &
        (df["Region"].isin(selected_regions)) &
        (df["Sales Channel"].isin(selected_channels))
    ]

    if filtered.empty:
        st.warning("No data matches current filters.")
        return

    total_revenue = filtered["Total Revenue"].sum()
    total_profit = filtered["Total Profit"].sum()
    avg_margin = filtered["Margin %"].mean()
    order_count = filtered["Order ID"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Revenue", f"${total_revenue:,.0f}")
    with c2:
        metric_card("Total Profit", f"${total_profit:,.0f}")
    with c3:
        metric_card("Average Margin", f"{avg_margin:.2f}%")
    with c4:
        metric_card("Unique Orders", f"{order_count:,}")

    monthly = filtered.groupby(filtered["Order Date"].dt.to_period("M")).agg({
        "Total Revenue": "sum", "Total Profit": "sum"
    }).reset_index()
    monthly["Month"] = monthly["Order Date"].astype(str)

    st.markdown("<h3 class='section-title'>Revenue Trend by Month</h3>", unsafe_allow_html=True)
    fig_monthly = px.line(monthly, x="Month", y="Total Revenue", markers=True, color_discrete_sequence=["#2c7a4b"])
    fig_monthly.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_monthly, use_container_width=True)

    left, right = st.columns(2)

    with left:
        top_items = filtered.groupby("Item Type").agg({"Total Profit": "sum"}).reset_index().sort_values("Total Profit", ascending=False).head(10)
        st.markdown("<h3 class='section-title'>Top 10 Item Types by Profit</h3>", unsafe_allow_html=True)
        fig_items = px.bar(top_items, x="Total Profit", y="Item Type", orientation="h", color="Total Profit", color_continuous_scale=["#d5eadb", "#2c7a4b"])
        fig_items.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_items, use_container_width=True)

    with right:
        by_region = filtered.groupby("Region").agg({"Total Profit": "sum"}).reset_index().sort_values("Total Profit", ascending=False)
        st.markdown("<h3 class='section-title'>Profit by Region</h3>", unsafe_allow_html=True)
        fig_region = px.bar(by_region, x="Region", y="Total Profit", color="Total Profit", color_continuous_scale=["#e6f4ea", "#2f855a"])
        fig_region.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
        fig_region.update_xaxes(tickangle=35)
        st.plotly_chart(fig_region, use_container_width=True)

    st.markdown("<h3 class='section-title'>Data Preview</h3>", unsafe_allow_html=True)
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)


def render_analytics():
    df = load_data()

    st.markdown("<h3 class='section-title'>Key Business Questions</h3>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["1. Highest Profit Region", "2. Orders in 72h", "3. Weekend Sales", "4. YoY Growth"])

    with tab1:
        region_profit = df.groupby("Region")["Total Profit"].sum().idxmax()
        region_profit_val = df.groupby("Region")["Total Profit"].sum().max()
        country_profit = df[df["Region"] == region_profit].groupBy("Country")["Total Profit"].sum().idxmax()
        country_profit_val = df[df["Region"] == region_profit].groupBy("Country")["Total Profit"].sum().max()
        st.metric("Highest Profit Region", region_profit, f"${region_profit_val:,.0f}")
        st.metric("Country with Highest Profit", country_profit, f"${country_profit_val:,.0f}")

    with tab2:
        df_sorted = df.sort_values("Order Date")
        df_sorted["Order Window"] = df_sorted["Order Date"].dt.floor("72h")
        orders_72h = df_sorted.groupby("Order Window").size().max()
        st.metric("Highest Orders in 72 Hours", orders_72h)

    with tab3:
        df["DayOfWeek"] = df["Order Date"].dt.dayofweek
        weekend_sales = df[(df["Region"] == "Sub-Saharan Africa") & (df["DayOfWeek"].isin([5, 6]))]
        weekend_items = weekend_sales.groupby("Item Type")["Units Sold"].sum().sort_values(ascending=False)
        st.metric("Most Sold on Weekends (Sub-Saharan Africa)", weekend_items.index[0])
        st.bar_chart(weekend_items.head(5))

    with tab4:
        yearly = df.groupby("Year")["Total Revenue"].sum().reset_index()
        yearly["YoY Growth %"] = yearly["Total Revenue"].pct_change() * 100
        st.dataframe(yearly, hide_index=True)
        fig_yoy = px.bar(yearly, x="Year", y="YoY Growth %", color_discrete_sequence=["#2c7a4b"])
        st.plotly_chart(fig_yoy, use_container_width=True)


def render_distributed_story():
    st.markdown("<h3 class='section-title'>Distributed Database Architecture</h3>", unsafe_allow_html=True)
    st.graphviz_chart("""
    digraph G {
        rankdir=LR;
        node [shape=box style=filled fontname="Segoe UI" fontsize=11 color="#d4e8dc"];
        producer [label="Order Producers\\n(API / CSV / Events)"];
        kafka [label="Kafka Topic\\norder-events"];
        spark [label="Spark Structured Streaming\\nValidation + Enrichment"];
        bronze [label="Delta Bronze\\nRaw Append"];
        silver [label="Delta Silver\\nCleaned + DQ Checks"];
        cassandra [label="Cassandra\\nServing Tables"];
        dashboard [label="Streamlit Frontend\\nRealtime KPIs"];
        producer -> kafka -> spark;
        spark -> bronze;
        bronze -> silver;
        silver -> cassandra;
        cassandra -> dashboard;
        silver -> dashboard;
    }
    """)
    st.caption("This project demonstrates a distributed data pipeline: Kafka → Spark → Delta Lake → Cassandra. For Streamlit Cloud deployment, we use pandas as a demonstration with the same analytics.")

    df = load_data()
    st.markdown("<h3 class='section-title'>Sample Analytics (Pandas Demo)</h3>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Records", f"{len(df):,}")
    with c2:
        metric_card("Total Revenue", f"${df['Total Revenue'].sum():,.0f}")
    with c3:
        metric_card("Total Profit", f"${df['Total Profit'].sum():,.0f}")
    with c4:
        metric_card("Avg Margin", f"{df['Margin %'].mean():.1f}%")

    by_country = df.groupby("Country")["Total Profit"].sum().sort_values(ascending=False).head(10)
    fig_country = px.bar(x=by_country.values, y=by_country.index, orientation="h", labels={"x": "Total Profit", "y": "Country"}, title="Top 10 Countries by Profit")
    st.plotly_chart(fig_country, use_container_width=True)


def main():
    st.set_page_config(page_title="Amazon Sales Analytics", page_icon=":bar_chart:", layout="wide")
    inject_styles()

    st.markdown("""
    <div class="hero">
        <h1>Amazon Sales Analytics Dashboard</h1>
        <p>Data visualization and analytics using pandas (Standalone Streamlit Cloud Version)</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("Page", ["Sales Overview", "Business Analytics", "Distributed DB Story"], index=0)

    if page == "Sales Overview":
        render_overview()
    elif page == "Business Analytics":
        render_analytics()
    else:
        render_distributed_story()


if __name__ == "__main__":
    main()