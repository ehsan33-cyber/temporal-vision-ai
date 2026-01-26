import os
import tempfile
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# Optional dependencies
# -----------------------------
try:
    import mne
    MNE_AVAILABLE = True
except Exception:
    MNE_AVAILABLE = False
    mne = None

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
    OpenAI = None


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Temporal Vision AI – EEG", layout="wide")
st.title("Temporal Vision AI")
st.caption("EEG viewer: stable EDF upload + labels + baselines + scrolling + demo markers + GPT summary")


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
# Session State Init
# -----------------------------
def ss_init():
    defaults = {
        "raw": None,
        "tmp_path": None,
        "picks": None,
        "ch_names": DEFAULT_CHANNELS,
        "sfreq": 256.0,
        "total_dur": 90.0,
        "file_key": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

ss_init()


# -----------------------------
# Helpers
# -----------------------------
def cleanup_tmp_file():
    path = st.session_state.get("tmp_path")
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass
    st.session_state.raw = None
    st.session_state.tmp_path = None
    st.session_state.picks = None


def make_blank_eeg(ch_names, sfreq=256.0, duration_s=90.0):
    n_ch = len(ch_names)
    n_samp = int(duration_s * sfreq)
    data_uV = np.zeros((n_ch, n_samp), dtype=np.float32)
    times = np.arange(n_samp, dtype=np.float32) / float(sfreq)
    return data_uV, times, float(sfreq), ch_names


def save_upload_to_temp(uploaded_file) -> str:
    # Save upload bytes to a temp file (reduces RAM issues)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def read_edf_safe(uploaded_file, resample_hz=None):
    if not MNE_AVAILABLE:
        st.error("Missing dependency: mne. Add it to requirements.txt: mne")
        st.stop()

    tmp_path = save_upload_to_temp(uploaded_file)
    raw = mne.io.read_raw_edf(tmp_path, preload=False, verbose="ERROR")

    if resample_hz:
        try:
            raw = raw.copy().resample(float(resample_hz), npad="auto")
        except Exception as e:
            st.warning(f"Resample failed; continuing without resampling. ({e})")

    return raw, tmp_path


def pick_channels(raw, max_channels: int):
    # Prefer EEG channels if present
    try:
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(picks) == 0:
            picks = np.arange(len(raw.ch_names))
    except Exception:
        picks = np.arange(len(raw.ch_names))

    picks = picks[:max_channels]
    ch_names = [raw.ch_names[i] for i in picks]
    return picks, ch_names


def load_window_uV(raw, picks, t0, window_s):
    sfreq = float(raw.info["sfreq"])
    n_total = int(raw.n_times)

    start = int(max(0, np.floor(t0 * sfreq)))
    stop = int(min(n_total, np.ceil((t0 + window_s) * sfreq)))
    if stop <= start + 2:
        stop = min(n_total, start + 3)

    data_V = raw.get_data(picks=picks, start=start, stop=stop)  # window only
    data_uV = (data_V * 1e6).astype(np.float32)
    times = (np.arange(start, stop, dtype=np.float32) / sfreq).astype(np.float32)
    return data_uV, times, sfreq


def detect_events_in_window(window_data_uV, sfreq, win_s=2.0, hop_s=0.5, z_thresh=2.7):
    """
    Demo heuristic (NOT clinical).
    Returns event intervals relative to the window start (seconds).
    """
    n_ch, n_samp = window_data_uV.shape
    win = int(win_s * sfreq)
    hop = int(hop_s * sfreq)

    if win < 8 or hop < 1 or n_samp < win:
        return []

    scores = []
    centers = []

    for s in range(0, n_samp - win + 1, hop):
        seg = window_data_uV[:, s:s + win]

        # Score: line-length + power
        ll = float(np.mean(np.sum(np.abs(np.diff(seg, axis=1)), axis=1)))
        pwr = float(np.mean(seg ** 2))
        score = float(np.log1p(ll) + 0.5 * np.log1p(pwr))

        scores.append(score)
        centers.append((s + win // 2) / sfreq)

    scores = np.asarray(scores, dtype=np.float32)
    if scores.size < 3 or float(np.std(scores)) < 1e-8:
        return []

    z = (scores - float(np.mean(scores))) / float(np.std(scores))
    hits = z > float(z_thresh)
    if not np.any(hits):
        return []

    idx = np.where(hits)[0].tolist()
    groups = []
    cur = [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)

    events = []
    for g in groups:
        a = centers[g[0]] - win_s / 2
        b = centers[g[-1]] + win_s / 2
        a = float(max(0.0, a))
        b = float(min(n_samp / sfreq, b))
        if (b - a) >= max(1.0, win_s):
            events.append((a, b))

    return events


def bandpower_simple(window_data_uV, sfreq):
    # Lightweight FFT bandpower estimate (demo)
    bands = {
        "delta_1_4": (1, 4),
        "theta_4_8": (4, 8),
        "alpha_8_13": (8, 13),
        "beta_13_30": (13, 30),
    }

    n_ch, n_samp = window_data_uV.shape
    if n_samp < int(sfreq * 1.5):
        return {k: None for k in bands}

    x = window_data_uV - np.mean(window_data_uV, axis=1, keepdims=True)
    freqs = np.fft.rfftfreq(n_samp, d=1.0 / sfreq)
    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2

    out = {}
    for name, (f0, f1) in bands.items():
        mask = (freqs >= f0) & (freqs < f1)
        out[name] = float(np.mean(spec[:, mask])) if np.any(mask) else None
    return out


def build_eeg_figure(window_data_uV, times, ch_names, t0, window_s, scale_uV, show_labels, event_intervals_global):
    n_ch, _ = window_data_uV.shape

    # Baseline offsets (stacked)
    spacing_uV = float(scale_uV) * 3.0
    offsets_uV = (np.arange(n_ch)[::-1] * spacing_uV).astype(np.float32)

    fig = go.Figure()

    for ci in range(n_ch):
        y = (window_data_uV[ci] / max(1e-6, float(scale_uV))) + (offsets_uV[ci] / max(1e-6, float(scale_uV)))
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
        # Baseline dotted line
        fig.add_hline(
            y=(offsets_uV[ci] / max(1e-6, float(scale_uV))),
            line_width=1,
            line_dash="dot",
            opacity=0.25,
        )

    # Marker overlay (shaded vertical regions)
    for (a, b) in event_intervals_global:
        # only if intersects view
        if b < t0 or a > (t0 + window_s):
            continue
        fig.add_vrect(x0=max(a, t0), x1=min(b, t0 + window_s), opacity=0.15, line_width=0)

    if show_labels:
        tickvals = (offsets_uV / max(1e-6, float(scale_uV))).tolist()
        ticktext = ch_names[::-1]
        fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
    else:
        fig.update_yaxes(showticklabels=False)

    fig.update_layout(
        height=720,
        margin=dict(l=75, r=30, t=30, b=50),
        showlegend=False,
        plot_bgcolor="#060b14",
        paper_bgcolor="#0b1220",
        font=dict(color="rgba(255,255,255,0.85)"),
    )
    fig.update_xaxes(range=[t0, t0 + window_s], showgrid=True, minor=dict(showgrid=True), zeroline=False, title="Time (s)")
    fig.update_yaxes(showgrid=True, minor=dict(showgrid=True), zeroline=False, title="")

    return fig


def get_openai_client():
    if not OPENAI_AVAILABLE:
        return None
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def gpt_summarize_window(payload: dict, model_name: str):
    """
    Never crashes the app.
    Returns either summary text or a helpful error message.
    """
    client = get_openai_client()
    if client is None:
        return "GPT is not configured. Add OPENAI_API_KEY to Streamlit secrets, and ensure openai is in requirements.txt."

    # Build prompt safely (no triple quotes)
    prompt = "\n".join([
    "You are assisting with an EEG review summary for research/prototyping.",
    "Base your summary ONLY on the provided EEG features and algorithm outputs.",
    "Do NOT diagnose. Use cautious language like 'algorithm-flagged' or 'may be consistent with'.",
    "",
    "REQUIRED OUTPUT STRUCTURE:",
    "1) Background EEG description (brief).",
    "2) Algorithm-flagged activity:",
    "   - State clearly whether seizure-like activity was flagged in this window.",
    "   - If flagged, list the time intervals and describe what features suggest this",
    "     (rhythmic activity, increased amplitude, evolving frequency, spikes, etc.).",
    "3) Channel involvement: name the most involved channels from the payload.",
    "4) Impression:",
    "   - If seizure_likelihood_window is 'moderate' or 'high', say:",
    "     'Seizure-like activity suspected by algorithm.'",
    "   - Otherwise say activity is not strongly suggestive of seizure.",
    "5) Recommendation: Clinician review recommended.",
    "",
    "FEATURES PAYLOAD:",
    str(payload),
])

    # Compatibility attempts
    attempts = [
        {"model": model_name, "input": prompt, "reasoning": {"effort": "minimal"}, "text": {"verbosity": "low"}},
        {"model": model_name, "input": prompt},
    ]

    last_err = None
    for kwargs in attempts:
        try:
            resp = client.responses.create(**kwargs)
            return resp.output_text
        except Exception as e:
            last_err = e

    return f"GPT summary failed: {type(last_err).__name__}: {last_err}"


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Input")
    use_edf = st.toggle("Upload EDF", value=True)
    uploaded = st.file_uploader("Upload .edf", type=["edf"]) if use_edf else None

    st.divider()
    st.header("Performance")
    resample_hz = st.selectbox("Resample for display", [None, 128, 256, 512], index=2)
    max_channels = st.slider("Max channels displayed", 4, 32, 19, 1)

    st.divider()
    st.header("View")
    window_s = st.slider("Window length (s)", 5, 60, 15, 1)
    scale_uV = st.slider("Scale (µV per division)", 10, 300, 75, 5)
    show_labels = st.toggle("Show channel labels", value=True)

    st.divider()
    st.header("AI markers (demo)")
    enable_markers = st.toggle("Overlay markers", value=True)
    z_thresh = st.slider("Sensitivity (z-threshold)", 1.5, 5.0, 2.7, 0.1)
    det_win_s = st.slider("Detection window (s)", 1.0, 6.0, 2.0, 0.5)
    det_hop_s = st.slider("Detection hop (s)", 0.25, 2.0, 0.5, 0.25)

    st.divider()
    st.header("GPT Summary")
    enable_gpt = st.toggle("Enable GPT summary", value=False)
    model_name = st.text_input("Model name", value="gpt-5.2")
    summarize_btn = st.button("Summarize this window")


# -----------------------------
# Load EDF or blank data
# -----------------------------
new_file_key = None
if uploaded is not None:
    new_file_key = f"{uploaded.name}:{uploaded.size}"

if new_file_key != st.session_state.file_key:
    cleanup_tmp_file()
    st.session_state.file_key = new_file_key

if use_edf and uploaded is not None:
    if st.session_state.raw is None:
        with st.spinner("Reading EDF (stable mode: preload=False)…"):
            raw, tmp_path = read_edf_safe(uploaded, resample_hz=resample_hz)
            picks, ch_names = pick_channels(raw, max_channels=max_channels)

            st.session_state.raw = raw
            st.session_state.tmp_path = tmp_path
            st.session_state.picks = picks
            st.session_state.ch_names = ch_names
            st.session_state.sfreq = float(raw.info["sfreq"])
            st.session_state.total_dur = float(raw.n_times / st.session_state.sfreq)

    raw = st.session_state.raw
    picks = st.session_state.picks
    ch_names = st.session_state.ch_names
    sfreq = st.session_state.sfreq
    total_dur = st.session_state.total_dur
else:
    raw = None
    picks = None
    data_uV, times_all, sfreq, ch_names = make_blank_eeg(DEFAULT_CHANNELS, sfreq=256.0, duration_s=90.0)
    total_dur = float(times_all[-1]) if len(times_all) else 90.0


# -----------------------------
# Header metrics + scroll slider
# -----------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Sampling rate", f"{sfreq:.1f} Hz")
with c2:
    st.metric("Channels", f"{len(ch_names)}")
with c3:
    st.metric("Duration", f"{total_dur:.1f} s")

max_t0 = max(0.0, float(total_dur) - float(window_s))
t0 = st.slider("Scroll (start time, seconds)", 0.0, float(max_t0), 0.0, 0.1)


# -----------------------------
# Load visible window only
# -----------------------------
if raw is not None:
    window_data_uV, times, sfreq = load_window_uV(raw, picks, float(t0), float(window_s))
else:
    # slice from blank data
    i0 = int(float(t0) * float(sfreq))
    i1 = int((float(t0) + float(window_s)) * float(sfreq))
    i1 = min(data_uV.shape[1], max(i1, i0 + 3))
    window_data_uV = data_uV[:, i0:i1]
    times = np.arange(i0, i1, dtype=np.float32) / float(sfreq)


# -----------------------------
# Per-window markers
# -----------------------------
event_intervals_global = []
if enable_markers and raw is not None:
    local_events = detect_events_in_window(
        window_data_uV=window_data_uV,
        sfreq=float(sfreq),
        win_s=float(det_win_s),
        hop_s=float(det_hop_s),
        z_thresh=float(z_thresh),
    )
    event_intervals_global = [(float(t0) + a, float(t0) + b) for (a, b) in local_events]
    st.caption("Shaded regions are algorithm-flagged (demo heuristic). Not a diagnosis.")


# -----------------------------
# Plot
# -----------------------------
fig = build_eeg_figure(
    window_data_uV=window_data_uV,
    times=times,
    ch_names=ch_names,
    t0=float(t0),
    window_s=float(window_s),
    scale_uV=float(scale_uV),
    show_labels=bool(show_labels),
    event_intervals_global=event_intervals_global,
)
st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# GPT summary (features only)
# -----------------------------
if enable_gpt and summarize_btn:
    bp = bandpower_simple(window_data_uV, float(sfreq))
    rms_uV = float(np.sqrt(np.mean(window_data_uV ** 2)))
    peak_uV = float(np.max(np.abs(window_data_uV)))
    line_len = float(np.mean(np.sum(np.abs(np.diff(window_data_uV, axis=1)), axis=1)))

    # ---- Seizure likelihood estimate (based on markers in view) ----
flagged_total_s = float(sum((b - a) for (a, b) in event_intervals_global))
likelihood = "low"
if flagged_total_s >= 3.0:
    likelihood = "moderate"
if flagged_total_s >= 6.0:
    likelihood = "high"

# ---- Channel involvement (most active channels) ----
per_ch_line_len = np.sum(np.abs(np.diff(window_data_uV, axis=1)), axis=1)
top_idx = np.argsort(per_ch_line_len)[-5:][::-1]
top_channels_ll = [{"channel": ch_names[i], "line_length": float(per_ch_line_len[i])} for i in top_idx]

    payload = {
    "window_start_s": float(t0),
    "window_length_s": float(window_s),
    "sampling_rate_hz": float(sfreq),
    "channels_displayed": ch_names,

    "signal_features": {
        "rms_uV": rms_uV,
        "peak_abs_uV": peak_uV,
        "mean_line_length": line_len,
        "bandpower_simple": bp,
    },

    "algorithm_flagged_intervals_s": [
        {"start": float(a), "end": float(b), "duration": float(b - a)}
        for (a, b) in event_intervals_global
    ],

    # 🔹 New seizure-awareness fields
    "seizure_likelihood_window": likelihood,

    "marker_summary": {
        "flagged_total_seconds_in_view": flagged_total_s,
        "flagged_count": len(event_intervals_global),
    },

    "channel_involvement": {
        "top_by_line_length": top_channels_ll,
    },

    "limitations": [
        "Prototype only. Not for clinical diagnosis.",
        "Markers are a demo heuristic unless replaced with a trained model."
    ],
}
        "seizure_likelihood_window": likelihood,
"marker_summary": {
    "flagged_total_seconds_in_view": flagged_total_s,
    "flagged_count": len(event_intervals_global),
},
"channel_involvement": {
    "top_by_line_length": top_channels_ll,
},

    with st.spinner("Generating summary…"):
        text = gpt_summarize_window(payload, model_name=model_name)

    st.subheader("GPT Summary (Prototype)")
    st.write(text)

    st.download_button(
        "Download summary (.txt)",
        data=str(text).encode("utf-8"),
        file_name="eeg_summary.txt",
        mime="text/plain",
    )


# -----------------------------
# Troubleshooting panel
# -----------------------------
with st.expander("Troubleshooting", expanded=False):
    st.write("If EDF upload crashes: reduce Max channels, reduce Window length, and resample to 128 or 256.")
    st.write("If GPT says not configured: add OPENAI_API_KEY to Streamlit secrets and ensure openai is installed.")
    st.write("This app intentionally avoids multi-line triple-quote prompts to prevent syntax errors.")
