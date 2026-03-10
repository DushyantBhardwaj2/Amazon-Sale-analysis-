from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

DATA_PATH = Path(__file__).resolve().parent.parent / "AmazonSalesData.csv"

BASE_COLUMNS = [
    "Region",
    "Country",
    "Item Type",
    "Sales Channel",
    "Order Priority",
    "Order Date",
    "Order ID",
    "Ship Date",
    "Units Sold",
    "Unit Price",
    "Unit Cost",
    "Total Revenue",
    "Total Cost",
    "Total Profit",
]

DERIVED_COLUMNS = ["Event Time", "Year", "Month", "Margin %"]


def _empty_sales_data() -> pd.DataFrame:
    # Keep schema stable so API endpoints can return empty payloads safely.
    return pd.DataFrame(columns=BASE_COLUMNS + DERIVED_COLUMNS)


def _demo_sales_data() -> pd.DataFrame:
    """Provide a local fallback dataset so dashboards still render without CSV."""
    regions = [
        ("Asia", "India"),
        ("Europe", "Germany"),
        ("Sub-Saharan Africa", "Nigeria"),
        ("Middle East and North Africa", "Egypt"),
        ("North America", "United States"),
        ("Central America and the Caribbean", "Mexico"),
    ]
    item_types = ["Office Supplies", "Cosmetics", "Household", "Baby Food", "Clothes"]
    channels = ["Online", "Offline"]
    priorities = ["L", "M", "H", "C"]

    rows: list[dict[str, Any]] = []
    start = pd.Timestamp("2022-01-01")

    for i in range(360):
        region, country = regions[i % len(regions)]
        item_type = item_types[i % len(item_types)]
        channel = channels[i % len(channels)]
        priority = priorities[i % len(priorities)]

        units = 60 + (i % 140)
        unit_price = 20.0 + float((i * 3) % 80)
        unit_cost = round(unit_price * 0.62, 2)
        total_revenue = round(units * unit_price, 2)
        total_cost = round(units * unit_cost, 2)
        total_profit = round(total_revenue - total_cost, 2)

        order_date = start + pd.Timedelta(days=i)
        ship_date = order_date + pd.Timedelta(days=(i % 7) + 1)

        rows.append(
            {
                "Region": region,
                "Country": country,
                "Item Type": item_type,
                "Sales Channel": channel,
                "Order Priority": priority,
                "Order Date": order_date,
                "Order ID": 100000 + i,
                "Ship Date": ship_date,
                "Units Sold": units,
                "Unit Price": unit_price,
                "Unit Cost": unit_cost,
                "Total Revenue": total_revenue,
                "Total Cost": total_cost,
                "Total Profit": total_profit,
            }
        )

    return pd.DataFrame(rows)

app = FastAPI(title="Amazon Sales Analytics API", version="1.0.0")

raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _coerce_numeric(pdf: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "Units Sold",
        "Unit Price",
        "Unit Cost",
        "Total Revenue",
        "Total Cost",
        "Total Profit",
        "Order ID",
    ]
    for col_name in numeric_cols:
        pdf[col_name] = pd.to_numeric(pdf[col_name], errors="coerce")
    return pdf


@lru_cache(maxsize=1)
def load_sales_data() -> pd.DataFrame:
    """Load CSV once and derive fields used across API endpoints."""
    if not DATA_PATH.exists():
        pdf = _demo_sales_data()
    else:
        use_spark_backend = os.getenv("USE_SPARK_BACKEND", "0") == "1"

        if use_spark_backend:
            try:
                from pyspark.sql import SparkSession
                from pyspark.sql.functions import col, to_date

                spark = SparkSession.builder.appName("AmazonSalesBackend").master("local[*]").getOrCreate()
                sdf = spark.read.csv(str(DATA_PATH), header=True, inferSchema=True)
                sdf = sdf.withColumn("Order Date", to_date(col("Order Date"), "M/d/yyyy"))
                pdf = sdf.toPandas()
                spark.stop()
            except Exception:
                pdf = pd.read_csv(DATA_PATH)
                pdf["Order Date"] = pd.to_datetime(pdf["Order Date"], format="%m/%d/%Y", errors="coerce")
        else:
            # Default to pandas for instant startup in demos and classroom systems.
            pdf = pd.read_csv(DATA_PATH)
            pdf["Order Date"] = pd.to_datetime(pdf["Order Date"], format="%m/%d/%Y", errors="coerce")

    pdf = _coerce_numeric(pdf)
    pdf = pdf.dropna(subset=["Order Date"]).copy()

    pdf["Event Time"] = (
        pdf["Order Date"]
        + pd.to_timedelta(pdf["Order ID"].fillna(0).astype("int64") % 24, unit="h")
        + pd.to_timedelta(pdf["Order ID"].fillna(0).astype("int64") % 60, unit="m")
    )
    pdf["Year"] = pdf["Order Date"].dt.year
    pdf["Month"] = pdf["Order Date"].dt.to_period("M").astype(str)
    pdf["Margin %"] = ((pdf["Total Profit"] / pdf["Total Revenue"]) * 100).round(2)
    return pdf


def _apply_filters(
    df: pd.DataFrame,
    years: list[int] | None,
    regions: list[str] | None,
    channels: list[str] | None,
) -> pd.DataFrame:
    filtered = df.copy()
    if years:
        filtered = filtered[filtered["Year"].isin(years)]
    if regions:
        filtered = filtered[filtered["Region"].isin(regions)]
    if channels:
        filtered = filtered[filtered["Sales Channel"].isin(channels)]
    return filtered


@app.get("/health")
def health() -> dict[str, Any]:
    df = load_sales_data()
    return {"status": "ok", "rows": int(len(df))}


@app.get("/filters")
def filters() -> dict[str, Any]:
    df = load_sales_data()
    return {
        "years": sorted(df["Year"].dropna().astype(int).unique().tolist()),
        "regions": sorted(df["Region"].dropna().astype(str).unique().tolist()),
        "channels": sorted(df["Sales Channel"].dropna().astype(str).unique().tolist()),
    }


@app.get("/overview")
def overview(
    years: list[int] | None = Query(default=None),
    regions: list[str] | None = Query(default=None),
    channels: list[str] | None = Query(default=None),
) -> dict[str, Any]:
    df = load_sales_data()
    filtered = _apply_filters(df, years, regions, channels)

    if filtered.empty:
        return {"empty": True}

    monthly = (
        filtered.groupby("Month", as_index=False)["Total Revenue"].sum().sort_values("Month").to_dict("records")
    )
    top_items = (
        filtered.groupby("Item Type", as_index=False)["Total Profit"]
        .sum()
        .sort_values("Total Profit", ascending=False)
        .head(10)
        .to_dict("records")
    )
    by_region = (
        filtered.groupby("Region", as_index=False)["Total Profit"]
        .sum()
        .sort_values("Total Profit", ascending=False)
        .to_dict("records")
    )

    preview = filtered[
        [
            "Order Date",
            "Region",
            "Country",
            "Item Type",
            "Sales Channel",
            "Units Sold",
            "Total Revenue",
            "Total Profit",
            "Margin %",
        ]
    ].copy()
    preview["Order Date"] = preview["Order Date"].dt.strftime("%Y-%m-%d")
    preview_records = preview.sort_values("Order Date", ascending=False).head(200).to_dict("records")

    return {
        "empty": False,
        "metrics": {
            "total_revenue": float(filtered["Total Revenue"].sum()),
            "total_profit": float(filtered["Total Profit"].sum()),
            "avg_margin": float(filtered["Margin %"].mean()),
            "order_count": int(filtered["Order ID"].nunique()),
        },
        "monthly": monthly,
        "top_items": top_items,
        "by_region": by_region,
        "preview": preview_records,
    }


@app.get("/distributed")
def distributed(
    batch_size: int = 30,
    stream_cursor: int | None = None,
    lookback_batches: int = 4,
) -> dict[str, Any]:
    df = load_sales_data().sort_values("Event Time").reset_index(drop=True)
    total_rows = len(df)

    if total_rows == 0:
        return {"empty": True}

    safe_batch = max(5, min(batch_size, 120))
    default_cursor = min(max(safe_batch * 4, safe_batch), total_rows)
    safe_cursor = stream_cursor if stream_cursor is not None else default_cursor
    safe_cursor = max(safe_batch, min(safe_cursor, total_rows))
    safe_lookback = max(1, min(lookback_batches, 10))

    window_size = safe_batch * safe_lookback
    stream_window = df.iloc[max(0, safe_cursor - window_size) : safe_cursor].copy()
    current_batch = df.iloc[max(0, safe_cursor - safe_batch) : safe_cursor].copy()

    if stream_window.empty:
        return {"empty": True, "total_rows": total_rows}

    stream_window["Event Minute"] = stream_window["Event Time"].dt.floor("h")

    minute_revenue = (
        stream_window.groupby("Event Minute", as_index=False)["Total Revenue"]
        .sum()
        .sort_values("Event Minute")
    )
    minute_revenue["Event Minute"] = minute_revenue["Event Minute"].dt.strftime("%Y-%m-%d %H:%M")

    by_country = (
        stream_window.groupby("Country", as_index=False)["Total Profit"]
        .sum()
        .sort_values("Total Profit", ascending=False)
        .head(8)
        .to_dict("records")
    )

    serving_table = (
        stream_window.groupby(["Region", "Item Type"], as_index=False)
        .agg({"Total Revenue": "sum", "Total Profit": "sum", "Units Sold": "sum"})
        .sort_values("Total Profit", ascending=False)
        .head(20)
        .to_dict("records")
    )

    return {
        "empty": False,
        "total_rows": total_rows,
        "batch_size": safe_batch,
        "stream_cursor": safe_cursor,
        "lookback_batches": safe_lookback,
        "metrics": {
            "events_ingested": int(len(stream_window)),
            "batch_revenue": float(current_batch["Total Revenue"].sum()),
            "batch_profit": float(current_batch["Total Profit"].sum()),
            "throughput": float(safe_batch / 5.0),
        },
        "minute_revenue": minute_revenue.to_dict("records"),
        "by_country": by_country,
        "serving_table": serving_table,
    }
