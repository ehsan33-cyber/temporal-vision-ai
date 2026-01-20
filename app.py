
import os
import tempfile
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib
import mne

# ---- MNE ----
try:
    import mne
    MNE_AVAILABLE = True
except Exception:
    MNE_AVAILABLE = False
    mne = None

# ---- OpenAI ----
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
    OpenAI = None

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Temporal Vision AI – EEG", layout="wide")
st.title("Temporal Vision AI")
st.caption("Stable EDF viewer: windowed loading + per-window markers + scrolling + GPT-5 summary")


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_CHANNELS = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2"
]


# -----------------------------
# Session state init
# -----------------------------
if "raw" not in st.session_state:
    st.session_state.raw = None
if "tmp_path" not in st.session_state:
    st.session_state.tmp_path = None
if "ch_names" not in st.session_state:
    st.session_state.ch_names = DEFAULT_CHANNELS
if "sfreq" not in st.session_state:
    st.session_state.sfreq = 256.0
if "picks" not in st.session_state:
    st.session_state.picks = None
if "total_dur" not in st.session_state:
    st.session_state.total_dur = 90.0
if "file_key" not in st.session_state:
    st.session_state.file_key = None


# -----------------------------
# Utilities
# -----------------------------
def cleanup_previous_tmp():
    """Delete old temp EDF file if present."""
    path = st.session_state.tmp_path
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass
    st.session_state.tmp_path = None
    st.session_state.raw = None
    st.session_state.picks = None

def make_blank_eeg(ch_names, sfreq=256.0, duration_s=90.0):
    n_ch = len(ch_names)
    n_samp = int(duration_s * sfreq)
    data_uV = np.zeros((n_ch, n_samp), dtype=np.float32)
    times = np.arange(n_samp) / sfreq
    return data_uV, times, sfreq, ch_names

def read_edf_safely(uploaded_file, resample_hz=None):
    """
    Save upload to a temp file and read with preload=False to avoid RAM spikes.
    Optionally resample for visualization (reduces CPU + plot load).
    """
    if not MNE_AVAILABLE:
        st.error("Missing dependency: mne. Install: pip install mne")
        st.stop()

    # Save to temp file (prevents keeping big bytes in RAM)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    raw = mne.io.read_raw_edf(tmp_path, preload=False, verbose="ERROR")

    # Optional visualization resample (still not preloaded; MNE will handle reads)
    if resample_hz and resample_hz > 0:
        try:
            raw = raw.copy().resample(resample_hz, npad="auto")
        except Exception as e:
            st.warning(f"Resample failed; continuing without resampling. ({e})")

    return raw, tmp_path

def load_window_uV(raw, picks, t0, window_s):
    """Load only the visible time window from EDF and return microvolts."""
    sfreq = float(raw.info["sfreq"])
    n_samp_total = raw.n_times

    start = int(max(0, np.floor(t0 * sfreq)))
    stop = int(min(n_samp_total, np.ceil((t0 + window_s) * sfreq)))
    if stop <= start + 2:
        stop = min(n_samp_total, start + 3)

    data = raw.get_data(picks=picks, start=start, stop=stop)  # small window only
    data_uV = (data * 1e6).astype(np.float32)  # EDF typically in Volts
    times = np.arange(start, stop) / sfreq
    return data_uV, times, sfreq

def simple_seizure_proxy_score(window_data_uV):
    """
    Demo heuristic score (NOT clinical):
    - line length + power
    """
    ll = float(np.mean(np.sum(np.abs(np.diff(window_data_uV, axis=1)), axis=1)))
    pwr = float(np.mean(window_data_uV ** 2))
    return float(np.log1p(ll) + 0.5 * np.log1p(pwr))

def detect_events_in_window(window_data_uV, sfreq, win_s=2.0, hop_s=0.5, z_thresh=2.7):
    """
    Detect seizure-like intervals within the CURRENT WINDOW only.
    Returns intervals in seconds relative to the window start (0..window_s).
    """
    n_ch, n_samp = window_data_uV.shape
    win = int(win_s * sfreq)
    hop = int(hop_s * sfreq)
    if win < 8 or hop < 1 or n_samp < win:
        return []

    scores = []
    centers = []
    for start in range(0, n_samp - win + 1, hop):
        seg = window_data_uV[:, start:start + win]
        scores.append(simple_seizure_proxy_score(seg))
        centers.append((start + win // 2) / sfreq)

    scores = np.asarray(scores, dtype=np.float32)
    if scores.size < 3 or np.std(scores) < 1e-8:
        return []

    z = (scores - np.mean(scores)) / np.std(scores)
    hits = z > z_thresh
    if not np.any(hits):
        return []

    idx = np.where(hits)[0]
    groups = []
    g = [idx[0]]
    for i in idx[1:]:
        if i == g[-1] + 1:
            g.append(i)
        else:
            groups.append(g)
            g = [i]
    groups.append(g)

    events = []
    for g in groups:
        start_c = centers[g[0]]
        end_c = centers[g[-1]]
        start_t = max(0.0, start_c - win_s / 2)
        end_t = min(n_samp / sfreq, end_c + win_s / 2)
        if (end_t - start_t) >= max(1.0, win_s):
            events.append((float(start_t), float(end_t)))
    return events

def bandpower_simple(uV, sfreq, bands=None):
    """
    Lightweight bandpower estimate using rFFT power.
    uV: (n_ch, n_samp)
    Returns dict of band -> mean power across channels.
    """
    if bands is None:
        bands = {
            "delta_1_4": (1, 4),
            "theta_4_8": (4, 8),
            "alpha_8_13": (8, 13),
            "beta_13_30": (13, 30),
        }

    n_ch, n_samp = uV.shape
    if n_samp < int(sfreq * 1.5):
        return {k: None for k in bands}  # too short for stable estimate

    x = uV - np.mean(uV, axis=1, keepdims=True)
    # rFFT
    freqs = np.fft.rfftfreq(n_samp, d=1.0 / sfreq)
    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2  # power

    out = {}
    for name, (f0, f1) in bands.items():
        mask = (freqs >= f0) & (freqs < f1)
        if not np.any(mask):
            out[name] = None
        else:
            out[name] = float(np.mean(spec[:, mask]))
    return out

def build_eeg_figure(window_data_uV, times, ch_names, t0, scale_uV, show_labels=True, event_intervals_global=None):
    """
    Stacked EEG with baselines + optional shaded marker regions.
    """
    n_ch, n_samp = window_data_uV.shape

    spacing_uV = scale_uV * 3.0
    offsets_uV = (np.arange(n_ch)[::-1] * spacing_uV).astype(np.float32)

    fig = go.Figure()

    # plot each channel + baseline
    for ci in range(n_ch):
        y = (window_data_uV[ci] / max(1e-6, scale_uV)) + (offsets_uV[ci] / max(1e-6, scale_uV))
        fig.add_trace(
            go.Scatter(
                x=times,
                y=y,
                mode="lines",
                line=dict(width=1.2),
                name=ch_names[ci],
                hovertemplate=f"{ch_names[ci]}<br>t=%{{x:.2f}} s<extra></extra>",
            )
        )
        fig.add_hline(
            y=(offsets_uV[ci] / max(1e-6, scale_uV)),
            line_width=1,
            line_dash="dot",
            opacity=0.25,
        )

    # overlay markers (global time coords)
    if event_intervals_global:
        for (a, b) in event_intervals_global:
            fig.add_vrect(x0=a, x1=b, opacity=0.15, line_width=0)

    # y tick labels at baseline lines
    if show_labels:
        tickvals = (offsets_uV / max(1e-6, scale_uV)).tolist()
        ticktext = ch_names[::-1]
        fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
    else:
        fig.update_yaxes(showticklabels=False)

    fig.update_layout(
        height=720,
        margin=dict(l=70, r=30, t=30, b=50),
        showlegend=False,
        plot_bgcolor="#060b14",
        paper_bgcolor="#0b1220",
        font=dict(color="rgba(255,255,255,0.85)"),
    )

    fig.update_xaxes(showgrid=True, minor=dict(showgrid=True), zeroline=False, title="Time (s)")
    fig.update_yaxes(showgrid=True, minor=dict(showgrid=True), zeroline=False, title="")

    return fig

def get_openai_client():
    """
    Creates an OpenAI client from Streamlit secrets.
    Requires: pip install openai
    In Streamlit secrets:
      OPENAI_API_KEY="..."
    """
    if not OPENAI_AVAILABLE:
        st.error("Missing dependency: openai. Install: pip install openai")
        return None

    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None

    if not key:
        st.error("OPENAI_API_KEY not found. Add it to Streamlit secrets.")
        return None

    return OpenAI(api_key=key)

def gpt5_summarize(payload: dict, model_name: str):
    client = get_openai_client()
    if client is None:
        return None

    prompt = f"""
You are assisting with an EEG review summary for research/prototyping.
Write a concise, clinically-styled narrative based ONLY on the provided features.
Do NOT diagnose.

FEATURES:
{payload}
"".strip()

    try:
        # ✅ Correct GPT-5 Responses API format
        resp = client.responses.create(
            model=model_name,
            input=prompt,
            reasoning={
                "effort": "none"   # low compute, safer
            },
            text={
                "verbosity": "low"
            }
        )
    except TypeError:
        # 🔁 Fallback (older SDK safety)
        resp = client.responses.create(
            model=model_name,
            input=prompt,
        )

    return resp.output_text

You are assisting with an EEG review summary for research/prototyping.
Write a concise, clinically-styled narrative based ONLY on the provided features.
Do NOT diagnose. If information is insufficient, say so.
Include:
- background rhythm impressions if possible
- artifact or limitations if noted
- mention detected intervals (if any) as "algorithm-flagged"
- recommend clinician review

FEATURES:
{payload}
""".strip()

    resp = client.responses.create(
        model=model_name,             # e.g. "gpt-5.2"
        input=prompt,
        reasoning_effort="minimal",
        verbosity="low",
    )
    return resp.output_text


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Input")
    use_edf = st.toggle("Upload EDF", value=True)

    uploaded = None
    if use_edf:
        uploaded = st.file_uploader("Upload .edf", type=["edf"])

    st.divider()
    st.header("Performance / Stability")
    resample_hz = st.selectbox("Resample for display (recommended)", [None, 128, 256, 512], index=2)
    max_channels = st.slider("Max channels to display", 4, 32, 19, 1)

    st.divider()
    st.header("View")
    window_s = st.slider("Window length (s)", 5, 60, 15, 1)
    scale_uV = st.slider("Scale (µV per division)", 10, 300, 75, 5)
    show_labels = st.toggle("Show channel labels", value=True)

    st.divider()
    st.header("AI markers (demo)")
    enable_markers = st.toggle("Overlay seizure markers", value=True)
    z_thresh = st.slider("Sensitivity (z-threshold)", 1.5, 5.0, 2.7, 0.1)
    det_win_s = st.slider("Detection window (s)", 1.0, 6.0, 2.0, 0.5)
    det_hop_s = st.slider("Detection hop (s)", 0.25, 2.0, 0.5, 0.25)

    st.divider()
    st.header("GPT-5 Summary")
    enable_gpt = st.toggle("Enable GPT summary", value=False)
    model_name = st.text_input("Model name", value="gpt-5.2")
    summarize_btn = st.button("Summarize this window")


# -----------------------------
# Load data (stable)
# -----------------------------
# Detect a "new file" and clean up old temp file
new_file_key = None
if uploaded is not None:
    new_file_key = f"{uploaded.name}:{uploaded.size}"
if new_file_key != st.session_state.file_key:
    # New upload or cleared upload → cleanup
    cleanup_previous_tmp()
    st.session_state.file_key = new_file_key

# If EDF uploaded, read it (preload=False)
if use_edf and uploaded is not None:
    if st.session_state.raw is None:
        with st.spinner("Reading EDF (safe mode: preload=False)…"):
            raw, tmp_path = read_edf_safely(uploaded, resample_hz=resample_hz)
            st.session_state.raw = raw
            st.session_state.tmp_path = tmp_path

            sfreq = float(raw.info["sfreq"])
            st.session_state.sfreq = sfreq
            st.session_state.total_dur = float(raw.n_times / sfreq)

            # pick EEG channels if possible, else all
            picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            if len(picks) == 0:
                picks = np.arange(len(raw.ch_names))

            # cap channels for performance
            picks = picks[:max_channels]
            st.session_state.picks = picks
            st.session_state.ch_names = [raw.ch_names[i] for i in picks]

    raw = st.session_state.raw
    sfreq = st.session_state.sfreq
    total_dur = st.session_state.total_dur
    picks = st.session_state.picks
    ch_names = st.session_state.ch_names

else:
    # Blank demo
    raw = None
    picks = None
    data_uV, times, sfreq, ch_names = make_blank_eeg(DEFAULT_CHANNELS, sfreq=256.0, duration_s=90.0)
    total_dur = float(times[-1]) if len(times) else 90.0


# -----------------------------
# Metrics + Scroll
# -----------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Sampling rate", f"{sfreq:.1f} Hz")
with col2:
    st.metric("Channels", f"{len(ch_names)}")
with col3:
    st.metric("Duration", f"{total_dur:.1f} s")

max_t0 = max(0.0, total_dur - window_s)
t0 = st.slider("Scroll (start time, seconds)", 0.0, float(max_t0), 0.0, 0.1)

# -----------------------------
# Load only the visible window
# -----------------------------
if raw is not None:
    window_data_uV, times, sfreq2 = load_window_uV(raw, picks, t0, window_s)
    sfreq = sfreq2
else:
    # blank
    i0 = int(t0 * sfreq)
    i1 = int((t0 + window_s) * sfreq)
    i1 = min(data_uV.shape[1], max(i1, i0 + 3))
    window_data_uV = data_uV[:, i0:i1]
    times = np.arange(i0, i1) / sfreq


# -----------------------------
# AI markers (per-window only)
# -----------------------------
event_intervals_global = []
if enable_markers and raw is not None:
    local_events = detect_events_in_window(
        window_data_uV=window_data_uV,
        sfreq=sfreq,
        win_s=det_win_s,
        hop_s=det_hop_s,
        z_thresh=z_thresh,
    )
    # Convert to global time coords for plotting
    event_intervals_global = [(t0 + a, t0 + b) for (a, b) in local_events]
    st.caption("Shaded regions are **algorithm-flagged** (demo heuristic). Not a diagnosis.")


# -----------------------------
# Plot
# -----------------------------
fig = build_eeg_figure(
    window_data_uV=window_data_uV,
    times=times,
    ch_names=ch_names,
    t0=t0,
    scale_uV=scale_uV,
    show_labels=show_labels,
    event_intervals_global=event_intervals_global,
)
st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# GPT-5 Summary (features only)
# -----------------------------
if enable_gpt and summarize_btn:
    # Compute compact features for GPT (don’t send raw arrays)
    bp = bandpower_simple(window_data_uV, sfreq)
    rms_uV = float(np.sqrt(np.mean(window_data_uV ** 2)))
    ll = float(np.mean(np.sum(np.abs(np.diff(window_data_uV, axis=1)), axis=1)))
    peak_uV = float(np.max(np.abs(window_data_uV)))

    payload = {
        "window_start_s": float(t0),
        "window_length_s": float(window_s),
        "sampling_rate_hz": float(sfreq),
        "channels_displayed": ch_names,
        "signal_features": {
            "rms_uV": rms_uV,
            "mean_line_length": ll,
            "peak_abs_uV": peak_uV,
            "bandpower_simple": bp,
        },
        "algorithm_flagged_intervals_s": [
            {"start": float(a), "end": float(b), "duration": float(b - a)}
            for (a, b) in event_intervals_global
        ],
        "limitations": [
            "Prototype summary. Not for clinical diagnosis.",
            "Bandpower estimates are simplistic and for demo only.",
            "Markers are based on a heuristic unless replaced with a trained model."
        ],
    }

    with st.spinner("Generating summary with GPT-5…"):
        text = gpt5_summarize(payload, model_name=model_name)
        if text:
            st.subheader("GPT Summary (Prototype)")
            st.write(text)
            st.download_button(
                "Download summary (.txt)",
                data=text.encode("utf-8"),
                file_name="eeg_summary.txt",
                mime="text/plain",
            )

# -----------------------------
# Optional: show intervals list
# -----------------------------
if enable_markers:
    with st.expander("Algorithm-flagged intervals (this window)", expanded=False):
        if not event_intervals_global:
            st.write("None.")
        else:
            for i, (a, b) in enumerate(event_intervals_global, 1):
                st.write(f"{i}. {a:.2f}s → {b:.2f}s  (duration {b-a:.2f}s)")


# -----------------------------
# Cleanup hint (for long sessions)
# -----------------------------
with st.expander("Troubleshooting", expanded=False):
    st.write(
        "- If uploads still crash on Streamlit Cloud: reduce **Max channels**, shorten **Window length**, and set **Resample** to 128 or 256.\n"
        "- For huge EDFs (hours long), this app stays stable because it only loads the visible window.\n"
        "- GPT summary requires the OpenAI Python package and an API key in Streamlit secrets."
    )
