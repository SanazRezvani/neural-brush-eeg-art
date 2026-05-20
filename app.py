
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy.signal import welch


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="Neural Brush EEG Art",
    layout="wide",
)

st.title("Neural Brush: EEG-Controlled Generative Art")
st.caption("EEG band-power features control a particle-based visual artwork with fading trails, flow-field motion, and EEG-reactive colour dynamics.")


# ============================================
# Data loading
# ============================================

from pathlib import Path

@st.cache_data(show_spinner=True)
def load_eeg_csv(uploaded_file_or_path):
    df = pd.read_csv(uploaded_file_or_path, low_memory=False)
    return df


# Default demo EEG path
default_path = Path("sample_data/demo_eeg.csv")


# ============================================
# Sidebar EEG controls
# ============================================

st.sidebar.header("1. EEG Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload EEG CSV",
    type=["csv"]
)

use_demo = st.sidebar.checkbox(
    "Use built-in demo EEG",
    value=True
)


# ============================================
# Load EEG
# ============================================

if uploaded_file is not None:

    df = load_eeg_csv(uploaded_file)

    st.sidebar.success("Custom EEG uploaded")

elif use_demo and default_path.exists():

    df = load_eeg_csv(default_path)

    st.sidebar.info("Using built-in demo EEG")

else:

    st.warning(
        "Please upload an EEG CSV or enable the built-in demo EEG."
    )

    st.stop()


# ============================================
# Validate required columns
# ============================================

required_columns = ["Time", "C3", "CZ", "C4"]

missing = [c for c in required_columns if c not in df.columns]

if missing:

    st.error(f"Missing required EEG columns: {missing}")

    st.stop()

# Estimate sampling rate from Time column
time_values = pd.to_numeric(df["Time"], errors="coerce").dropna().values
dt = np.median(np.diff(time_values[: min(len(time_values), 10000)]))
fs = int(round(1 / dt)) if dt > 0 else 1000

st.sidebar.write(f"Detected sampling rate: **{fs} Hz**")
st.sidebar.write(f"Rows: **{len(df):,}**")
st.sidebar.write(f"Duration: **{time_values[-1]:.1f} seconds**")


# -----------------------------
# Channel selection
# -----------------------------
exclude_cols = {
    "Time",
    "Trig",
    "Cues",
    "PhanFrame",
    "PhanTime",
    "RelTime",
    "RecordingTimestamp",
    "LocalTimeStamp",
    "Blinks",
}

numeric_channels = []
for col in df.columns:
    if col not in exclude_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.95:
            numeric_channels.append(col)

default_channels = [ch for ch in ["C3", "CZ", "C4"] if ch in numeric_channels]
if not default_channels:
    default_channels = numeric_channels[:3]

st.sidebar.header("2. Feature Settings")
selected_channels = st.sidebar.multiselect(
    "EEG channels for control",
    options=numeric_channels,
    default=default_channels,
)

window_sec = st.sidebar.slider("Feature window size (seconds)", 1.0, 5.0, 2.0, 0.5)
step_sec = st.sidebar.slider("Step size (seconds)", 0.25, 2.0, 0.5, 0.25)

st.sidebar.header("3. Visual Style")
trail_steps = st.sidebar.slider("Trail length", 4, 40, 24, 2)
trail_strength = st.sidebar.slider("Trail brightness", 0.05, 0.60, 0.28, 0.05)
colour_intensity = st.sidebar.slider("EEG colour intensity", 0.20, 1.00, 0.78, 0.05)

if not selected_channels:
    st.warning("Please select at least one EEG channel.")
    st.stop()


# -----------------------------
# Feature extraction
# -----------------------------
def bandpower(x, fs, band):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < fs:
        return np.nan

    freqs, psd = welch(x, fs=fs, nperseg=min(len(x), fs * 2))
    idx = (freqs >= band[0]) & (freqs <= band[1])

    if not np.any(idx):
        return np.nan

    return np.trapezoid(psd[idx], freqs[idx])


@st.cache_data(show_spinner=True)
def compute_features(df, channels, fs, window_sec, step_sec):
    window = int(window_sec * fs)
    step = int(step_sec * fs)

    times = pd.to_numeric(df["Time"], errors="coerce").values

    feature_rows = []
    for start in range(0, len(df) - window, step):
        end = start + window
        segment = df.iloc[start:end]

        alpha_values = []
        beta_values = []
        theta_values = []

        for ch in channels:
            sig = pd.to_numeric(segment[ch], errors="coerce").values
            sig = sig - np.nanmean(sig)

            theta_values.append(bandpower(sig, fs, (4, 7)))
            alpha_values.append(bandpower(sig, fs, (8, 12)))
            beta_values.append(bandpower(sig, fs, (13, 30)))

        blink_value = 0
        if "Blinks" in segment.columns:
            blink_value = int(pd.to_numeric(segment["Blinks"], errors="coerce").fillna(0).max() > 0)

        feature_rows.append(
            {
                "time": float(np.nanmean(times[start:end])),
                "theta": float(np.nanmean(theta_values)),
                "alpha": float(np.nanmean(alpha_values)),
                "beta": float(np.nanmean(beta_values)),
                "blink": blink_value,
            }
        )

    features = pd.DataFrame(feature_rows)

    # Robust normalisation for visual mapping
    for col in ["theta", "alpha", "beta"]:
        low = features[col].quantile(0.05)
        high = features[col].quantile(0.95)
        features[col + "_norm"] = ((features[col] - low) / (high - low + 1e-9)).clip(0, 1)

    # Useful art-control values
    features["relaxation"] = features["alpha_norm"]
    features["focus"] = features["beta_norm"]
    features["dreaminess"] = features["theta_norm"]

    return features


features = compute_features(
    df,
    selected_channels,
    fs,
    float(window_sec),
    float(step_sec),
)


# -----------------------------
# Particle art generator
# -----------------------------
def smoothstep(u):
    """Smooth interpolation curve used for value-noise / Perlin-like fields."""
    return u * u * (3 - 2 * u)


def value_noise_2d(x, y, seed=0):
    """
    Lightweight dependency-free 2D value noise.

    This gives a Perlin-like flowing field without requiring the external `noise`
    package. It is not mathematically identical to classic Perlin noise, but it
    produces the same type of organic vector-field motion needed for this MVP.
    """
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    sx = smoothstep(x - x0)
    sy = smoothstep(y - y0)

    def hash_grid(ix, iy):
        n = ix * 374761393 + iy * 668265263 + seed * 1442695041
        n = (n ^ (n >> 13)) * 1274126177
        n = n ^ (n >> 16)
        return (n & 0xFFFFFFFF) / 0xFFFFFFFF

    n00 = hash_grid(x0, y0)
    n10 = hash_grid(x1, y0)
    n01 = hash_grid(x0, y1)
    n11 = hash_grid(x1, y1)

    ix0 = n00 * (1 - sx) + n10 * sx
    ix1 = n01 * (1 - sx) + n11 * sx
    return ix0 * (1 - sy) + ix1 * sy


def flow_field_positions(
    base_x,
    base_y,
    t,
    relaxation,
    focus,
    dreaminess,
    blink,
    steps=18,
):
    """
    Move particles through a Perlin-like noise vector field.

    Instead of simple circular orbiting, each particle samples an invisible
    organic field. The field angle changes across x/y space and over time, so
    particles move like smoke, ink, or neural dust.
    """
    x = base_x.copy()
    y = base_y.copy()

    # EEG controls the field personality.
    field_scale = 1.15 + 1.8 * dreaminess       # larger theta = wider, softer flow structures
    temporal_flow = 0.20 + 0.55 * relaxation    # alpha = smoother evolving field
    turbulence = 0.55 + 2.65 * focus            # beta = more energetic turns
    step_size = 0.018 + 0.030 * focus + 0.010 * dreaminess
    burst = 0.10 * int(blink)

    for i in range(steps):
        tt = t * temporal_flow + i * 0.025

        # Two independent noise layers create the x/y vector direction.
        n1 = value_noise_2d(x * field_scale + tt, y * field_scale - tt, seed=11)
        n2 = value_noise_2d(x * field_scale - tt * 0.7, y * field_scale + tt * 0.9, seed=29)

        # Convert noise to direction angle.
        angle = 2 * np.pi * (n1 + 0.55 * n2) * turbulence

        # Small inward/outward breathing force keeps the artwork centred but alive.
        r = np.sqrt(x * x + y * y) + 1e-6
        radial_breath = (0.010 * dreaminess - 0.006 * relaxation) * np.sin(t + r * 3)

        x += step_size * np.cos(angle) + radial_breath * x / r + burst * x / r
        y += step_size * np.sin(angle) + radial_breath * y / r + burst * y / r

    return x, y


def make_particle_art(relaxation, focus, dreaminess, blink, frame_index, n_particles=900, trail_steps=24, trail_strength=0.28):
    """
    Create an artistic EEG-reactive particle artwork using a Perlin-like flow field.

    Visual mapping:
    - Alpha / relaxation: smoother, slower, more coherent field evolution
    - Beta / focus: stronger turbulence and faster particle motion
    - Theta / dreaminess: larger soft structures and expanded atmospheric field
    - Blink: momentary outward pulse / ripple
    - Trails: older particle positions are redrawn with fading opacity, creating a long-exposure brushstroke effect
    - Colour: EEG controls warm/cool hue, saturation, and luminosity
    """
    rng = np.random.default_rng(42)

    # Stable particle identities. A fixed random seed keeps particles continuous
    # across frames, while the flow-field equations move them organically.
    base_angle = rng.uniform(0, 2 * np.pi, n_particles)
    base_radius = rng.power(1.65, n_particles) * (0.95 + 0.45 * dreaminess) + 0.04
    phase = rng.uniform(0, 2 * np.pi, n_particles)

    # Initial cloud: slightly asymmetric so it feels less like a perfect diagram.
    base_x = base_radius * np.cos(base_angle) + 0.05 * np.sin(phase)
    base_y = base_radius * np.sin(base_angle) + 0.05 * np.cos(phase * 1.3)

    t = frame_index * 0.090

    # Main current particle positions from the flow field.
    x, y = flow_field_positions(
        base_x,
        base_y,
        t,
        relaxation,
        focus,
        dreaminess,
        blink,
        steps=18,
    )

    # Fading trails. We compute previous positions from earlier time points and
    # draw them before the current particles.
    trail_x = []
    trail_y = []
    trail_alpha = []
    trail_size_scale = []

    for k in range(trail_steps, 0, -1):
        age = k / max(trail_steps, 1)
        tk = t - k * 0.055
        tx, ty = flow_field_positions(
            base_x,
            base_y,
            tk,
            relaxation,
            focus,
            dreaminess,
            blink,
            steps=18,
        )
        trail_x.append(tx)
        trail_y.append(ty)
        trail_alpha.append(trail_strength * (1 - age) ** 1.9 + 0.004)
        trail_size_scale.append(0.18 + 0.72 * (1 - age))

    # ---------------------------------------------------------
    # EEG-reactive colour system
    # ---------------------------------------------------------
    # This section maps EEG into three colour dimensions:
    # 1) warm <-> cool
    #    - beta/focus warms the image toward orange, amber, and magenta
    #    - alpha/relaxation cools it toward blue, violet, and cyan
    #
    # 2) saturated <-> faded
    #    - beta/focus and blink increase saturation
    #    - alpha/relaxation softens/fades the palette
    #
    # 3) dark <-> luminous
    #    - theta/dreaminess and blink increase brightness/glow
    #    - calmer/low-energy moments become darker and more subtle

    # Positive values are warmer; negative values are cooler.
    warmth = np.clip(0.62 * focus + 0.18 * blink - 0.55 * relaxation, -1, 1)

    # Saturation controls how vivid or washed-out the particles look.
    saturation_level = np.clip(0.28 + 0.58 * focus + 0.32 * blink - 0.20 * relaxation, 0.16, 1.0)

    # Luminance controls how bright/glowing the whole field becomes.
    luminance_level = np.clip(0.28 + 0.52 * dreaminess + 0.18 * focus + 0.28 * blink, 0.18, 1.0)

    # A small noise texture prevents every particle from having the same colour.
    colour_noise = value_noise_2d(base_x * 2.2 + t, base_y * 2.2 - t, seed=71)

    # HSV hue design:
    # cool centre: blue/cyan/violet region
    # warm centre: orange/magenta region
    cool_hue = 0.62 + 0.12 * colour_noise       # blue -> violet
    warm_hue = 0.04 + 0.09 * colour_noise       # red/orange -> amber
    mix = (warmth + 1) / 2
    hue = (1 - mix) * cool_hue + mix * warm_hue

    # Add gentle individual variation around the chosen warm/cool family.
    hue = (hue + 0.045 * np.sin(base_angle * 3 + t)) % 1.0
    sat = np.clip(saturation_level * (0.70 + 0.30 * colour_noise), 0, 1)
    val = np.clip(luminance_level * (0.62 + 0.38 * colour_noise), 0, 1)
    colours = hsv_to_rgb(np.column_stack([hue, sat, val]))

    # Dark <-> luminous also affects the background itself.
    bg_level = int(3 + 18 * luminance_level)
    bg_colour = f"#{bg_level:02x}{bg_level:02x}{min(bg_level + 8, 35):02x}"

    # Size and transparency mapping.
    size = 8 + 55 * focus + 40 * dreaminess + 110 * blink * rng.random(n_particles)
    alpha_main = np.clip(0.22 + 0.38 * relaxation + 0.18 * luminance_level, 0.16, 0.88)

    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor=bg_colour)
    ax.set_facecolor(bg_colour)

    # Soft atmospheric halo.
    halo_radius = 0.55 + 1.15 * dreaminess + 0.35 * blink
    for i in range(9):
        halo = plt.Circle(
            (0, 0),
            halo_radius + i * 0.16,
            fill=False,
            linewidth=2.2,
            alpha=max(0.012, 0.11 - i * 0.011),
        )
        ax.add_patch(halo)

    # Draw trail memory first.
    for tx, ty, ta, ts in zip(trail_x, trail_y, trail_alpha, trail_size_scale):
        ax.scatter(tx, ty, s=size * ts, c=colours, alpha=ta * colour_intensity, edgecolors="none")

    # Glow layer + current particles.
    ax.scatter(x, y, s=size * 4.4, c=colours, alpha=0.035 + 0.055 * luminance_level, edgecolors="none")
    ax.scatter(x, y, s=size, c=colours, alpha=alpha_main * colour_intensity, edgecolors="none")

    # A few faint field streamlines, also generated by the same field logic.
    line_rng = np.random.default_rng(7)
    for _ in range(28):
        start_angle = line_rng.uniform(0, 2 * np.pi)
        start_r = line_rng.uniform(0.15, 1.55 + 0.25 * dreaminess)
        lx = np.array([start_r * np.cos(start_angle)])
        ly = np.array([start_r * np.sin(start_angle)])
        path_x = [lx[0]]
        path_y = [ly[0]]
        for j in range(12):
            px, py = flow_field_positions(
                lx,
                ly,
                t + j * 0.03,
                relaxation,
                focus,
                dreaminess,
                0,
                steps=2,
            )
            lx, ly = px, py
            path_x.append(px[0])
            path_y.append(py[0])
        line_colour = hsv_to_rgb([[(0.62 * (1 - mix) + 0.04 * mix) % 1.0, 0.35 + 0.45 * saturation_level, 0.35 + 0.45 * luminance_level]])[0]
        ax.plot(path_x, path_y, linewidth=0.55, color=line_colour, alpha=0.035 + 0.06 * relaxation)

    # Central breathing core.
    core_size = 520 + 2400 * dreaminess + 1500 * blink
    core_hue = (0.62 * (1 - mix) + 0.04 * mix) % 1.0
    core_colour = hsv_to_rgb([[core_hue, saturation_level, luminance_level]])[0]
    ax.scatter([0], [0], s=core_size, c=[core_colour], alpha=0.12 + 0.12 * luminance_level, edgecolors="none")
    ax.scatter([0], [0], s=90 + 760 * focus, c=[core_colour], alpha=0.42 + 0.28 * luminance_level, edgecolors="none")

    # Blink ripple: a clear embodied event.
    if blink:
        ripple_circle = plt.Circle((0, 0), 1.95, fill=False, linewidth=4, alpha=0.45)
        ax.add_patch(ripple_circle)

    ax.set_xlim(-2.35, 2.35)
    ax.set_ylim(-2.35, 2.35)
    ax.set_aspect("equal")
    ax.axis("off")

    title = (
        f"NEURAL BRUSH FLOW FIELD   |   α calm {relaxation:.2f}   β focus {focus:.2f}   "
        f"θ dream {dreaminess:.2f}   blink {blink}"
    )
    ax.set_title(title, fontsize=10, color="#e8e2ff", pad=16)

    return fig

# -----------------------------
# Interface
# -----------------------------
left, right = st.columns([1, 1.25])

with left:
    st.subheader("EEG-derived control signals")
    st.line_chart(
        features.set_index("time")[["relaxation", "focus", "dreaminess"]],
        height=260,
    )

    st.dataframe(
        features[["time", "relaxation", "focus", "dreaminess", "blink"]].head(20),
        use_container_width=True,
    )

with right:
    st.subheader("Generated neural artwork")

    mode = st.radio("Display mode", ["Single frame", "Animate"], horizontal=True)

    if mode == "Single frame":

        # Safety check FIRST
        if features.empty:
            st.error(
                "No EEG features were created. Try using a longer EEG sample."
            )
            st.stop()

        # Frame slider
        frame_index = st.slider(
            "Choose time window",
            0,
            len(features) - 1,
            0
        )

        # Safe looping index
        frame_index = frame_index % len(features)

        # Get EEG feature row
        row = features.iloc[frame_index]

        # Generate artwork
        fig = make_particle_art(
            row["relaxation"],
            row["focus"],
            row["dreaminess"],
            int(row["blink"]),
            frame_index,
            trail_steps=trail_steps,
            trail_strength=trail_strength,
        )

        st.pyplot(fig)

        plt.close(fig)

    else:
        max_frames = st.slider("Number of animation frames", 20, min(300, len(features)), 80)
        speed = st.slider("Animation delay", 0.02, 0.30, 0.08, 0.02)

        placeholder = st.empty()

        start_button = st.button("Start animation")

        if start_button:

            # Safety check BEFORE animation loop
            if features.empty:
                st.error(
                    "No EEG features were created. Try using a longer EEG sample."
                )
                st.stop()

            for frame_index in range(max_frames):

                # Safe looping index
                frame_index = frame_index % len(features)

                # Current EEG feature row
                row = features.iloc[frame_index]

                # Generate artwork
                fig = make_particle_art(
                    row["relaxation"],
                    row["focus"],
                    row["dreaminess"],
                    int(row["blink"]),
                    frame_index,
                    trail_steps=trail_steps,
                    trail_strength=trail_strength,
                )

                # Display frame
                placeholder.pyplot(fig)

                plt.close(fig)

                time.sleep(speed)


st.markdown("---")
st.subheader("Interpretation")
st.markdown(
    """
This app maps EEG features to visual behaviour:

- **Alpha power (8-12 Hz)** → smoother, calmer particle movement.
- **Beta power (13-30 Hz)** → faster, more energetic particle behaviour.
- **Theta power (4-7 Hz)** → central pulse / dreamy expansion.
- **Blink events** → short visual burst.
- **Perlin-like flow field** → particles move like smoke, ink, or neural dust instead of simple circular orbits.
- **Fading trails** → previous particle positions remain visible as neural brushstrokes.
- **Warm-cool colour mapping** → beta/focus warms the palette; alpha/relaxation cools it.
- **Saturated-faded mapping** → beta and blink events make particles more vivid; calm states soften the palette.
- **Dark-luminous mapping** → theta/dreaminess and blink events increase glow and brightness.

This is not yet a clinical or validated cognitive-state classifier. It is an artistic BCI prototype that turns EEG-derived features into a real-time visual experience.
"""
)
