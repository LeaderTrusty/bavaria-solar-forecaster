import datetime
import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Bavaria Solar PV Forecaster — Physics-Informed ML & SHAP",
    page_icon="☀️",
    layout="wide",
)

# --- Constants & Readable Feature Labels ---
LAT_DEG, LON_DEG, TILT_DEG = 48.9, 11.4, 30.0

FEATURE_LABELS = {
    "poa_effective": "Tilted POA Irradiance (Temp-Adjusted)",
    "poa_irradiance": "Tilted POA Irradiance (Raw)",
    "poa_roll_3h": "3h Rolling POA Irradiance",
    "shortwave_radiation": "Horizontal Shortwave Radiation",
    "direct_radiation": "Direct Beam Radiation",
    "diffuse_radiation": "Diffuse Sky Radiation",
    "rad_roll_3h": "3h Rolling Shortwave Radiation",
    "cloud_cover": "Cloud Cover (%)",
    "cloud_roll_3h": "3h Rolling Cloud Cover (%)",
    "temperature_2m": "Ambient Temperature (°C)",
    "pv_cell_temp_c": "Estimated PV Cell Temp (°C)",
    "temp_derating_factor": "Thermal Derating Factor",
    "solar_elevation_deg": "Solar Elevation Angle (°)",
    "direct_rad_ratio": "Direct-to-Total Radiation Ratio",
    "wind_speed_10m": "Wind Speed (10m)",
    "precipitation": "Precipitation (mm)",
    "is_daylight": "Daylight Indicator",
    "hour_sin": "Hour of Day (sin)",
    "hour_cos": "Hour of Day (cos)",
    "doy_sin": "Day of Year (sin)",
    "doy_cos": "Day of Year (cos)",
    "cf_lag_24h": "Capacity Factor Lag (24h)",
    "cf_lag_48h": "Capacity Factor Lag (48h)",
    "cf_lag_168h": "Capacity Factor Lag (168h / 1wk)",
    "cf_roll_mean_24h_lag24": "Prior 24h Mean Capacity Factor",
    "cf_roll_max_168h_lag24": "Prior 7-Day Peak Capacity Factor",
}

PRESETS = {
    "☀️ Summer Peak Noon (July)": {
        "date": datetime.date(2026, 7, 15),
        "local_hour": 13,
        "cloud_cover": 10,
        "temp_c": 29.0,
        "wind_kmh": 10.0,
        "precip_mm": 0.0,
        "prior_sky": "Clear / Sunny",
        "cap_scale": 100,
    },
    "🔥 Heatwave Derating Stress (38°C)": {
        "date": datetime.date(2026, 7, 24),
        "local_hour": 14,
        "cloud_cover": 5,
        "temp_c": 38.0,
        "wind_kmh": 6.0,
        "precip_mm": 0.0,
        "prior_sky": "Clear / Sunny",
        "cap_scale": 100,
    },
    "⛈️ Summer Storm Front (Overcast)": {
        "date": datetime.date(2026, 7, 15),
        "local_hour": 13,
        "cloud_cover": 95,
        "temp_c": 18.0,
        "wind_kmh": 28.0,
        "precip_mm": 8.5,
        "prior_sky": "Partly Cloudy",
        "cap_scale": 100,
    },
    "❄️ Crisp Winter Midday (January)": {
        "date": datetime.date(2026, 1, 18),
        "local_hour": 12,
        "cloud_cover": 20,
        "temp_c": -2.0,
        "wind_kmh": 12.0,
        "precip_mm": 0.0,
        "prior_sky": "Partly Cloudy",
        "cap_scale": 100,
    },
    "🚀 Future Bavaria Grid (+25% PV Fleet)": {
        "date": datetime.date(2026, 6, 21),
        "local_hour": 13,
        "cloud_cover": 15,
        "temp_c": 26.0,
        "wind_kmh": 12.0,
        "precip_mm": 0.0,
        "prior_sky": "Clear / Sunny",
        "cap_scale": 125,
    },
}


# --- Cached Loaders ---
@st.cache_resource
def load_model_and_explainer():
    artifact = joblib.load("lgbm_bavaria_solar_cf.joblib")
    model = artifact["model"]
    features = artifact["features"]
    latest_cap_mw = float(artifact["latest_capacity_mw"])
    explainer = shap.TreeExplainer(model)
    return model, features, latest_cap_mw, explainer


@st.cache_data
def load_holdout_predictions():
    path = "bavaria_solar_test_comparison_2026.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["timestamp_utc"])


# --- Physics-Consistent Day Generator ---
def simulate_day_features(
    sim_date: datetime.date,
    cloud_cover: float,
    temp_c: float,
    wind_kmh: float,
    precip_mm: float,
    prior_sky: str,
) -> pd.DataFrame:
    """Generate a physically consistent 24-hour UTC feature DataFrame for Bavaria."""
    doy = sim_date.timetuple().tm_yday
    hours_utc = np.arange(24, dtype=float)

    # 1. Astronomical Solar Geometry for Bavaria (48.9°N, 11.4°E)
    b_rad = 2.0 * np.pi * (doy - 81) / 365.0
    declination_rad = np.radians(23.44 * np.sin(b_rad))
    eot_min = 9.87 * np.sin(2.0 * b_rad) - 7.53 * np.cos(b_rad) - 1.5 * np.sin(b_rad)
    solar_time = hours_utc + (LON_DEG / 15.0) + (eot_min / 60.0)
    hour_angle_rad = np.radians(15.0 * (solar_time - 12.0))

    lat_rad = np.radians(LAT_DEG)
    tilt_rad = np.radians(TILT_DEG)
    eff_lat_rad = np.radians(LAT_DEG - TILT_DEG)

    sin_elev = (
        np.sin(lat_rad) * np.sin(declination_rad)
        + np.cos(lat_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad)
    )
    cos_inc = (
        np.sin(eff_lat_rad) * np.sin(declination_rad)
        + np.cos(eff_lat_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad)
    )

    solar_elevation_deg = np.degrees(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))
    is_daylight = (solar_elevation_deg > 0).astype(int)

    # 2. Clear-Sky GHI & Kasten-Czeplak Cloud Attenuation
    ghi_clear = np.where(
        sin_elev > 0,
        1098.0 * sin_elev * np.exp(-0.057 / np.maximum(sin_elev, 0.02)),
        0.0,
    )
    cloud_frac = cloud_cover / 100.0
    shortwave_radiation = ghi_clear * (1.0 - 0.75 * (cloud_frac**3.4))

    direct_frac = np.where(
        shortwave_radiation > 0,
        np.clip(0.82 * (1.0 - 0.88 * (cloud_frac**1.8)), 0.05, 0.85),
        0.0,
    )
    direct_radiation = shortwave_radiation * direct_frac
    diffuse_radiation = shortwave_radiation - direct_radiation

    # 3. Tilted 30° Plane-of-Array (POA) Irradiance & Thermal Derating
    rb = np.where(
        sin_elev > 0.15,
        np.clip(cos_inc / np.maximum(sin_elev, 0.15), 0.0, 2.5),
        0.0,
    )
    poa_irradiance = (
        direct_radiation * rb
        + diffuse_radiation * (1.0 + np.cos(tilt_rad)) / 2.0
    )

    # Simple diurnal temperature swing around selected midday ambient temp
    diurnal_temp = temp_c - 3.5 * np.cos(2.0 * np.pi * (hours_utc - 13.0) / 24.0)
    # Lock exact slider temperature at the user's selected midday/target hour via post-override
    pv_cell_temp_c = diurnal_temp + 0.03125 * shortwave_radiation
    temp_derating_factor = 1.0 - 0.004 * np.maximum(0.0, pv_cell_temp_c - 25.0)
    poa_effective = poa_irradiance * temp_derating_factor
    direct_rad_ratio = np.clip(
        direct_radiation / (shortwave_radiation + 1e-5), 0.0, 1.0
    )

    # 4. Prior-Day Weather Regime -> Physically Plausible Day-Ahead Lags
    prior_cloud_map = {
        "Clear / Sunny": 0.15,
        "Partly Cloudy": 0.50,
        "Overcast / Rainy": 0.90,
    }
    p_cloud = prior_cloud_map.get(prior_sky, 0.50)
    p_sw = ghi_clear * (1.0 - 0.75 * (p_cloud**3.4))
    p_dir_f = np.where(
        p_sw > 0, np.clip(0.82 * (1.0 - 0.88 * (p_cloud**1.8)), 0.05, 0.85), 0.0
    )
    p_poa = (p_sw * p_dir_f) * rb + (p_sw * (1.0 - p_dir_f)) * (
        1.0 + np.cos(tilt_rad)
    ) / 2.0
    p_cf = np.clip((p_poa / 1000.0) * 0.88, 0.0, 1.15)
    p_cf[solar_elevation_deg <= 0] = 0.0

    # 7-day peak reflects a typical bright day in this season
    clear_poa = (ghi_clear * 0.82) * rb + (ghi_clear * 0.18) * (
        1.0 + np.cos(tilt_rad)
    ) / 2.0
    seasonal_peak_cf = float(np.clip((clear_poa.max() / 1000.0) * 0.88, 0.05, 1.15))

    df_day = pd.DataFrame(
        {
            "utc_hour": hours_utc.astype(int),
            "shortwave_radiation": shortwave_radiation,
            "direct_radiation": direct_radiation,
            "diffuse_radiation": diffuse_radiation,
            "poa_irradiance": poa_irradiance,
            "poa_effective": poa_effective,
            "cloud_cover": np.full(24, float(cloud_cover)),
            "temperature_2m": diurnal_temp,
            "wind_speed_10m": np.full(24, float(wind_kmh)),
            "precipitation": np.full(24, float(precip_mm)),
            "solar_elevation_deg": solar_elevation_deg,
            "is_daylight": is_daylight,
            "pv_cell_temp_c": pv_cell_temp_c,
            "temp_derating_factor": temp_derating_factor,
            "direct_rad_ratio": direct_rad_ratio,
            "hour_sin": np.sin(2.0 * np.pi * hours_utc / 24.0),
            "hour_cos": np.cos(2.0 * np.pi * hours_utc / 24.0),
            "doy_sin": np.full(24, np.sin(2.0 * np.pi * doy / 365.25)),
            "doy_cos": np.full(24, np.cos(2.0 * np.pi * doy / 365.25)),
            "cf_lag_24h": p_cf,
            "cf_lag_48h": p_cf * 0.96,
            "cf_lag_168h": p_cf * 0.98,
            "cf_roll_mean_24h_lag24": np.full(24, float(p_cf.mean())),
            "cf_roll_max_168h_lag24": np.full(24, seasonal_peak_cf),
        }
    )

    # Centered 3-hour rolling averages across the 24-hour curve
    df_day["rad_roll_3h"] = (
        df_day["shortwave_radiation"].rolling(3, center=True, min_periods=1).mean()
    )
    df_day["cloud_roll_3h"] = (
        df_day["cloud_cover"].rolling(3, center=True, min_periods=1).mean()
    )
    df_day["poa_roll_3h"] = (
        df_day["poa_effective"].rolling(3, center=True, min_periods=1).mean()
    )
    return df_day


def predict_mwh_curve(model, features, df_day, effective_cap_mw):
    cf_preds = np.clip(model.predict(df_day[features]), 0.0, None)
    mwh_preds = cf_preds * effective_cap_mw
    mwh_preds[df_day["solar_elevation_deg"].values <= -1.0] = 0.0
    return cf_preds, mwh_preds


# --- Load Artifacts ---
if not os.path.exists("lgbm_bavaria_solar_cf.joblib"):
    st.error(
        "Model file `lgbm_bavaria_solar_cf.joblib` not found in the repository root. "
        "Please upload it from Google Colab."
    )
    st.stop()

model, CF_FEATURES, BASE_CAP_MW, explainer = load_model_and_explainer()
holdout_df = load_holdout_predictions()

# --- Initialize Session State for Sliders ---
default_p = PRESETS["☀️ Summer Peak Noon (July)"]
for k, v in default_p.items():
    if k not in st.session_state:
        st.session_state[k] = v


def apply_preset():
    chosen = st.session_state["preset_selector"]
    if chosen in PRESETS:
        for k, v in PRESETS[chosen].items():
            st.session_state[k] = v


# --- Sidebar Controls ---
st.sidebar.header("🎛️ Bavaria What-If Controls")
st.sidebar.selectbox(
    "Quick Scenario Presets",
    options=list(PRESETS.keys()),
    key="preset_selector",
    on_change=apply_preset,
)

st.sidebar.divider()
sim_date = st.sidebar.date_input(
    "Date (Seasonal Sun Angle)",
    min_value=datetime.date(2026, 1, 1),
    max_value=datetime.date(2026, 12, 31),
    key="date",
)
local_hour = st.sidebar.slider("Local Bavarian Time (Hour)", 0, 23, key="local_hour")
cloud_cover = st.sidebar.slider("Cloud Cover (%)", 0, 100, key="cloud_cover")
temp_c = st.sidebar.slider("Ambient Temperature (°C)", -15.0, 42.0, step=0.5, key="temp_c")
wind_kmh = st.sidebar.slider("Wind Speed (km/h)", 0.0, 50.0, step=1.0, key="wind_kmh")
precip_mm = st.sidebar.slider("Precipitation (mm/h)", 0.0, 20.0, step=0.5, key="precip_mm")
prior_sky = st.sidebar.selectbox(
    "Yesterday's Sky Regime (Lag Context)",
    ["Clear / Sunny", "Partly Cloudy", "Overcast / Rainy"],
    key="prior_sky",
)
cap_scale = st.sidebar.slider(
    "Bavaria PV Fleet Capacity (% of 2026)",
    min_value=80,
    max_value=150,
    step=5,
    key="cap_scale",
    help="Simulate grid output as Bavaria adds new solar installations.",
)

# Convert local Bavarian time (CET=+1, CEST=+2) to UTC hour
is_summer_time = 4 <= sim_date.month <= 10
utc_offset = 2 if is_summer_time else 1
tz_label = "CEST (UTC+2)" if is_summer_time else "CET (UTC+1)"
target_utc_hour = (local_hour - utc_offset) % 24
effective_cap_mw = BASE_CAP_MW * (cap_scale / 100.0)

# --- Page Header ---
st.title("☀️ Bavaria Solar Power Forecaster & SHAP Explainer")
st.markdown(
    "Physics-informed **LightGBM** forecasting for Germany's **TenneT / Bavaria** control zone, "
    "combining tilted Plane-of-Array (POA) solar geometry, silicon thermal derating, and causal fleet capacity scaling."
)

tab_sim, tab_bench, tab_method = st.tabs(
    [
        "🎛️ Live What-If Simulator & SHAP",
        "📈 2026 Holdout Benchmark",
        "🧠 Physics & Architecture",
    ]
)

# ==============================================================================
# TAB 1: LIVE WHAT-IF SIMULATOR + REAL-TIME TREE SHAP WATERFALL
# ==============================================================================
with tab_sim:
    # Simulate selected weather & clear-sky (0% clouds) reference day
    df_day = simulate_day_features(
        sim_date, cloud_cover, temp_c, wind_kmh, precip_mm, prior_sky
    )
    # Apply exact user ambient temperature at the selected target hour
    df_day.loc[target_utc_hour, "temperature_2m"] = float(temp_c)
    sw_h = df_day.loc[target_utc_hour, "shortwave_radiation"]
    poa_h = df_day.loc[target_utc_hour, "poa_irradiance"]
    cell_t_h = float(temp_c) + 0.03125 * sw_h
    derate_h = 1.0 - 0.004 * max(0.0, cell_t_h - 25.0)
    df_day.loc[target_utc_hour, "pv_cell_temp_c"] = cell_t_h
    df_day.loc[target_utc_hour, "temp_derating_factor"] = derate_h
    df_day.loc[target_utc_hour, "poa_effective"] = poa_h * derate_h
    df_day["poa_roll_3h"] = (
        df_day["poa_effective"].rolling(3, center=True, min_periods=1).mean()
    )

    df_clear = simulate_day_features(
        sim_date, 0.0, temp_c, wind_kmh, 0.0, "Clear / Sunny"
    )

    cf_curve, mwh_curve = predict_mwh_curve(model, CF_FEATURES, df_day, effective_cap_mw)
    _, clear_mwh_curve = predict_mwh_curve(model, CF_FEATURES, df_clear, effective_cap_mw)

    # Extract metrics for the selected hour
    row_x = df_day.iloc[[target_utc_hour]][CF_FEATURES]
    pred_mwh = float(mwh_curve[target_utc_hour])
    pred_cf_pct = float(cf_curve[target_utc_hour] * 100.0)
    clear_mwh = float(clear_mwh_curve[target_utc_hour])
    elev_deg = float(df_day.loc[target_utc_hour, "solar_elevation_deg"])
    derating_loss_pct = (1.0 - derate_h) * 100.0

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        f"Predicted Output ({local_hour:02d}:00 {tz_label.split()[0]})",
        f"{pred_mwh:,.0f} MWh",
        delta=f"{pred_mwh - clear_mwh:+,.0f} MWh vs. clear sky" if elev_deg > 0 else "Nighttime (0 MWh)",
    )
    k2.metric(
        "Fleet Capacity Factor",
        f"{pred_cf_pct:.1f}%",
        delta=f"Effective Fleet: {effective_cap_mw:,.0f} MW",
        delta_color="off",
    )
    k3.metric(
        "Tilted POA Irradiance (30°)",
        f"{df_day.loc[target_utc_hour, 'poa_effective']:,.0f} W/m²",
        delta=f"Sun Elevation: {elev_deg:.1f}°",
        delta_color="off",
    )
    k4.metric(
        "Estimated PV Cell Temp",
        f"{cell_t_h:.1f} °C",
        delta=f"-{derating_loss_pct:.1f}% thermal loss" if derating_loss_pct > 0.05 else "No thermal derating",
        delta_color="inverse" if derating_loss_pct > 0.05 else "off",
    )

    st.divider()

    col_curve, col_shap = st.columns([1.1, 1.15], gap="large")

    # --- Left Column: 24-Hour Bavarian Solar Curve ---
    with col_curve:
        st.subheader("📈 Simulated 24-Hour Bavaria Solar Curve")
        local_hours_axis = (df_day["utc_hour"].values + utc_offset) % 24
        order_idx = np.argsort(local_hours_axis)
        h_sorted = local_hours_axis[order_idx]
        mwh_sorted = mwh_curve[order_idx]
        clear_sorted = clear_mwh_curve[order_idx]

        fig_day = go.Figure()
        fig_day.add_trace(
            go.Scatter(
                x=h_sorted,
                y=clear_sorted,
                mode="lines",
                name="Clear-Sky Potential (0% Clouds)",
                line=dict(color="#94a3b8", width=2, dash="dot"),
            )
        )
        fig_day.add_trace(
            go.Scatter(
                x=h_sorted,
                y=mwh_sorted,
                mode="lines+markers",
                name="Simulated Scenario",
                line=dict(color="#f28e2b", width=3),
                fill="tozeroy",
                fillcolor="rgba(242, 142, 43, 0.18)",
            )
        )
        fig_day.add_trace(
            go.Scatter(
                x=[local_hour],
                y=[pred_mwh],
                mode="markers",
                name="Selected Hour",
                marker=dict(color="#1f4e79", size=13, line=dict(color="white", width=2)),
            )
        )
        fig_day.update_layout(
            xaxis_title=f"Local Bavarian Time ({tz_label})",
            yaxis_title="Predicted Solar Generation (MWh)",
            xaxis=dict(tickmode="linear", dtick=2),
            height=430,
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_day, width="stretch")

    # --- Right Column: Real-Time TreeSHAP Waterfall (MWh) ---
    with col_shap:
        st.subheader("🔍 Real-Time SHAP Waterfall Explanation (MWh)")

        # Exact TreeSHAP values scaled by effective fleet capacity -> exact MWh
        raw_shap_vals = np.asarray(explainer.shap_values(row_x)).ravel() * effective_cap_mw
        base_mwh = float(np.asarray(explainer.expected_value).ravel()[0]) * effective_cap_mw
        raw_pred_mwh = base_mwh + float(raw_shap_vals.sum())

        # Sort by absolute impact: keep Top 8 drivers + group remaining features
        top_n = 8
        idx_sorted = np.argsort(np.abs(raw_shap_vals))[::-1]
        top_idx = idx_sorted[:top_n]
        rest_idx = idx_sorted[top_n:]

        wf_labels = ["Base Value (Avg)"]
        wf_values = [base_mwh]
        wf_measures = ["absolute"]
        wf_text = [f"{base_mwh:,.0f}"]

        for i in top_idx:
            fname = CF_FEATURES[i]
            pretty = FEATURE_LABELS.get(fname, fname)
            fval = float(row_x.iloc[0, i])
            val_mwh = float(raw_shap_vals[i])
            wf_labels.append(f"{pretty} = {fval:.1f}")
            wf_values.append(val_mwh)
            wf_measures.append("relative")
            wf_text.append(f"{val_mwh:+,.0f}")

        if len(rest_idx) > 0:
            rest_sum = float(raw_shap_vals[rest_idx].sum())
            wf_labels.append(f"{len(rest_idx)} Other Features")
            wf_values.append(rest_sum)
            wf_measures.append("relative")
            wf_text.append(f"{rest_sum:+,.0f}")

        wf_labels.append("Model Prediction")
        wf_values.append(raw_pred_mwh)
        wf_measures.append("total")
        wf_text.append(f"{raw_pred_mwh:,.0f}")

        fig_wf = go.Figure(
            go.Waterfall(
                orientation="h",
                measure=wf_measures[::-1],
                y=wf_labels[::-1],
                x=wf_values[::-1],
                text=wf_text[::-1],
                textposition="outside",
                increasing=dict(marker=dict(color="#2a9d8f")),
                decreasing=dict(marker=dict(color="#e15759")),
                totals=dict(marker=dict(color="#1f4e79")),
                connector=dict(line=dict(color="#cbd5e1", width=1)),
            )
        )
        fig_wf.update_layout(
            xaxis_title="Contribution to Hourly Generation (MWh)",
            height=430,
            margin=dict(l=10, r=40, t=20, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_wf, width="stretch")

        if elev_deg <= -1.0:
            st.info(
                "🌙 **Nighttime Physics Clip Applied:** Sun elevation is below -1.0°, "
                "so the final production forecast is locked to **0 MWh** regardless of small tree residuals."
            )

# ==============================================================================
# TAB 2: 2026 UNSEEN HOLDOUT BENCHMARK & DIAGNOSTICS
# ==============================================================================
with tab_bench:
    st.subheader("Unseen 2026 Holdout Performance (Train: 2023–2024 | Val: 2025 | Test: 2026)")

    if holdout_df is None:
        st.warning(
            "Upload `bavaria_solar_test_comparison_2026.csv` to your GitHub repo root "
            "to enable the interactive 2026 holdout benchmark explorer."
        )
    else:
        day_mask = holdout_df["solar_elevation_deg"] > 0
        peak_thresh = holdout_df.loc[day_mask, "solar_mwh"].quantile(0.90)
        peak_mask = holdout_df["solar_mwh"] >= peak_thresh

        def compute_metrics(name, col):
            y_t = holdout_df["solar_mwh"]
            y_p = holdout_df[col]
            return {
                "Model": name,
                "Overall MAE (MWh)": round(mean_absolute_error(y_t, y_p), 1),
                "Daytime MAE (MWh)": round(mean_absolute_error(y_t[day_mask], y_p[day_mask]), 1),
                "Top-10% Peak MAE (MWh)": round(mean_absolute_error(y_t[peak_mask], y_p[peak_mask]), 1),
                "RMSE (MWh)": round(float(np.sqrt(mean_squared_error(y_t, y_p))), 1),
                "R²": round(r2_score(y_t, y_p), 4),
            }

        bench_table = pd.DataFrame(
            [
                compute_metrics("1. Seasonal Naive (24h Lag)", "pred_naive"),
                compute_metrics("2. Ridge Regression (Linear)", "pred_ridge"),
                compute_metrics("3. LightGBM (Raw MWh Target)", "pred_lgbm"),
                compute_metrics("4. LightGBM (Capacity-Normalized CF)", "pred_lgbm_cf"),
            ]
        )
        st.dataframe(bench_table, hide_index=True, width="stretch")

        st.divider()

        # Interactive 7-day slice inspector
        min_d = holdout_df["timestamp_utc"].min().date()
        max_d = holdout_df["timestamp_utc"].max().date()
        default_start = datetime.date(2026, 7, 1)
        if not (min_d <= default_start <= max_d):
            default_start = min_d

        pick_start = st.date_input(
            "Inspect a 7-Day Window in 2026:",
            value=default_start,
            min_value=min_d,
            max_value=max_d,
        )
        start_ts = pd.Timestamp(pick_start, tz="UTC")
        end_ts = start_ts + pd.Timedelta(days=7)
        slice_df = holdout_df[
            (holdout_df["timestamp_utc"] >= start_ts)
            & (holdout_df["timestamp_utc"] < end_ts)
        ]

        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Scatter(
                x=slice_df["timestamp_utc"],
                y=slice_df["solar_mwh"],
                name="Actual (SMARD TenneT)",
                line=dict(color="#1f4e79", width=2.5),
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=slice_df["timestamp_utc"],
                y=slice_df["pred_lgbm_cf"],
                name="LightGBM (Capacity-Normalized CF)",
                line=dict(color="#2a9d8f", width=2.2, dash="dash"),
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=slice_df["timestamp_utc"],
                y=slice_df["pred_lgbm"],
                name="LightGBM (Raw MWh — Clipped)",
                line=dict(color="#e15759", width=1.6, dash="dot"),
            )
        )
        fig_ts.update_layout(
            title="Hourly Forecast Comparison on Unseen 2026 Holdout Data",
            xaxis_title="Timestamp (UTC)",
            yaxis_title="Solar PV Generation (MWh)",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_ts, width="stretch")

# ==============================================================================
# TAB 3: PHYSICS & METHODOLOGY
# ==============================================================================
with tab_method:
    st.subheader("Engineering Choices Behind This Forecaster")
    st.markdown(
        """
        1. **Strict UTC Multi-API Synchronization:** Hourly TenneT control-zone generation data from Germany's **SMARD.de** API is merged with averaged Bavarian weather across **Munich, Nuremberg, and Würzburg** (Open-Meteo) strictly on UTC timestamps to eliminate CET/CEST Daylight Saving Time gaps and duplicates.
        2. **Plane-of-Array (POA) Geometry & Silicon Thermal Derating:** Horizontal shortwave radiation is projected onto a standard 30° south-facing Bavarian panel tilt using astronomical declination and hour-angle equations. Cell temperature is estimated via NOCT heating ($T_{\\text{cell}} = T_{\\text{ambient}} + 0.03125 \\cdot G$) with a $-0.4\\% / ^\\circ\\text{C}$ efficiency derating above $25^\\circ\\text{C}$.
        3. **Eliminating Tree Extrapolation Clipping via Causal Capacity Scaling:** Because Bavaria rapidly expanded installed solar capacity between 2023 and 2026, gradient-boosted trees trained on raw 2023–2024 MWh hit an artificial ceiling on 2026 summer peaks. By estimating effective fleet capacity causally (shifted 24h in the past) and predicting a stationary **Capacity Factor**, the model generalizes cleanly to unseen fleet growth without future data leakage.
        """
    )
