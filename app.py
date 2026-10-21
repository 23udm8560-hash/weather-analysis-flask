import os
import base64
from io import BytesIO
import joblib
import numpy as np
from datetime import timedelta

import requests
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file


app = Flask(__name__)

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_FILE = os.path.join(DATA_DIR, "weather_data.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# API endpoints
GEOCODING_API = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API = "https://api.open-meteo.com/v1/forecast"


def get_city_coordinates(city):
    """Find the latitude and longitude of a city."""

    response = requests.get(
        GEOCODING_API,
        params={
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json",
        },
        timeout=15,
    )

    response.raise_for_status()
    results = response.json().get("results", [])

    if not results:
        raise ValueError(
            f"City '{city}' was not found. Check the spelling."
        )

    location = results[0]

    return {
        "name": location["name"],
        "country": location.get("country", ""),
        "latitude": location["latitude"],
        "longitude": location["longitude"],
    }


def collect_weather_data(location):
    """Collect hourly weather data from the weather API."""

    response = requests.get(
        WEATHER_API,
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "hourly": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "precipitation,"
                "wind_speed_10m"
            ),
            "past_days": 7,
            "forecast_days": 1,
            "timezone": "auto",
        },
        timeout=30,
    )

    response.raise_for_status()
    weather = response.json()

    hourly = weather.get("hourly")

    if not hourly or not hourly.get("time"):
        raise ValueError("The weather API returned no hourly data.")

    df = pd.DataFrame({
        "time": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "rainfall": hourly["precipitation"],
        "wind_speed": hourly["wind_speed_10m"],
    })

    # Add location details
    df["city"] = location["name"]
    df["country"] = location["country"]

    # Clean and prepare the data
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    numeric_columns = [
        "temperature",
        "humidity",
        "rainfall",
        "wind_speed",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["time"])
    df = df.drop_duplicates(subset=["city", "time"])
    df = df.sort_values("time")

    return df


def save_weather_data(df):
    """Append new records to the CSV without duplicating timestamps."""

    if os.path.exists(CSV_FILE):
        existing = pd.read_csv(CSV_FILE, parse_dates=["time"])

        combined = pd.concat(
            [existing, df],
            ignore_index=True,
        )
    else:
        combined = df.copy()

    combined["time"] = pd.to_datetime(
        combined["time"], errors="coerce"
    )

    combined = combined.dropna(subset=["time"])

    combined = combined.drop_duplicates(
        subset=["city", "time"],
        keep="last",
    )

    combined = combined.sort_values(
        ["city", "time"]
    )

    combined.to_csv(CSV_FILE, index=False)


def load_city_data(city):
    """Load previously collected records for the selected city."""

    if not os.path.exists(CSV_FILE):
        return pd.DataFrame()

    df = pd.read_csv(CSV_FILE, parse_dates=["time"])

    if "city" not in df.columns:
        return pd.DataFrame()

    return df[df["city"].str.casefold() == city.casefold()].copy()


def create_chart(df, chart_type):
    """Create a chart and return it as a Base64 image."""

    if df.empty:
        return None

    df = df.sort_values("time")

    fig, ax = plt.subplots(figsize=(10, 4))

    if chart_type == "temperature":
        ax.plot(
            df["time"],
            df["temperature"],
            marker=".",
            linewidth=1.5,
        )
        ax.set_title("Hourly Temperature")
        ax.set_ylabel("Temperature (°C)")

    elif chart_type == "humidity":
        ax.plot(
            df["time"],
            df["humidity"],
            marker=".",
            linewidth=1.5,
        )
        ax.set_title("Hourly Relative Humidity")
        ax.set_ylabel("Humidity (%)")
        ax.set_ylim(0, 100)

    elif chart_type == "rainfall":
        ax.bar(
            df["time"],
            df["rainfall"],
            width=0.03,
        )
        ax.set_title("Hourly Precipitation")
        ax.set_ylabel("Precipitation (mm)")

    ax.set_xlabel("Date and time")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    image_buffer = BytesIO()

    fig.savefig(
        image_buffer,
        format="png",
        dpi=110,
        bbox_inches="tight",
    )

    plt.close(fig)

    image_buffer.seek(0)

    return base64.b64encode(
        image_buffer.getvalue()
    ).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
    """Display the dashboard and process city searches."""

    city = ""
    error = None
    stats = None
    records = []
    charts = {}

    if request.method == "POST":
        city = request.form.get("city", "").strip()

        if not city or len(city) > 100:
            error = "Please enter a valid city name."

        else:
            try:
                location = get_city_coordinates(city)

                new_data = collect_weather_data(location)

                save_weather_data(new_data)

                # Read all saved records for this city.
                df = load_city_data(location["name"])

                if df.empty:
                    raise ValueError(
                        "No saved weather data was found."
                    )

                # Numeric cleaning for analysis.
                numeric_columns = [
                    "temperature",
                    "humidity",
                    "rainfall",
                    "wind_speed",
                ]

                for column in numeric_columns:
                    df[column] = pd.to_numeric(
                        df[column], errors="coerce"
                    )

                # Statistics for the dashboard.
                stats = {
                    "records": len(df),
                    "avg_temp": round(
                        df["temperature"].mean(), 1
                    ),
                    "max_temp": round(
                        df["temperature"].max(), 1
                    ),
                    "min_temp": round(
                        df["temperature"].min(), 1
                    ),
                    "avg_humidity": round(
                        df["humidity"].mean(), 1
                    ),
                    "total_rain": round(
                        df["rainfall"].sum(), 1
                    ),
                    "avg_wind": round(
                        df["wind_speed"].mean(), 1
                    ),
                }

                charts = {
                    "temperature": create_chart(
                        df, "temperature"
                    ),
                    "humidity": create_chart(
                        df, "humidity"
                    ),
                    "rainfall": create_chart(
                        df, "rainfall"
                    ),
                }

                # Show the latest 24 records.
                records = (
                    df.sort_values("time", ascending=False)
                    .head(24)
                    .fillna("")
                    .to_dict(orient="records")
                )

                city = location["name"]

            except requests.RequestException:
                error = (
                    "Unable to connect to the weather service. "
                    "Check your internet connection and try again."
                )

            except (ValueError, KeyError) as exc:
                error = str(exc)

            except Exception:
                app.logger.exception("Dashboard processing failed")
                error = (
                    "Something went wrong while processing "
                    "the weather data."
                )

    return render_template(
        "index.html",
        city=city,
        error=error,
        stats=stats,
        records=records,
        charts=charts,
    )


@app.route("/download")
def download_csv():
    """Download all collected weather data."""

    if not os.path.exists(CSV_FILE):
        return "No weather data has been collected yet.", 404

    return send_file(
        CSV_FILE,
        as_attachment=True,
        download_name="weather_data.csv",
        mimetype="text/csv",
    )


if __name__ == "__main__":
    app.run(debug=True)
