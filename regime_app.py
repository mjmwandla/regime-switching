"""
Optimal Conditional TE Policy in Regime-Switching Markets
=========================================================
Interactive research companion app — Mzobe & Flint (SAFA 2026)

Three steps:
  1. GMM Regime Discovery
  2. HSMM Duration Modelling
  3. Optimal TE Allocation & Monte-Carlo Simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from scipy.stats import nbinom, norm
from scipy.optimize import minimize
import io
import os
import warnings
warnings.filterwarnings("ignore")

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "MSCI ACWI Monthly REturn CSV VIX_cleaned.csv",
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regime-Switching TE Policy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 4px 4px 0 0;
    }
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 12px 16px;
        border-radius: 8px;
    }
    .explanation-box {
        background: #f0f7ff;
        border-left: 4px solid #1a73e8;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.92em;
    }
    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette for regimes
# ─────────────────────────────────────────────────────────────────────────────
REGIME_COLOURS = {
    0: "#2ecc71",  # green  — Bull
    1: "#f39c12",  # amber  — Tranquil
    2: "#e74c3c",  # red    — Bear
    3: "#9b59b6",  # purple — extra
    4: "#3498db",  # blue   — extra
}
REGIME_NAMES_DEFAULT = {0: "Bull", 1: "Tranquil", 2: "Bear", 3: "Regime 4", 4: "Regime 5"}


def hex_to_rgba(hex_colour: str, alpha: float = 0.5) -> str:
    """Convert a hex colour string (e.g. '#2ecc71') to an rgba() string."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load & cache data
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(_DATA_PATH)
    except FileNotFoundError:
        st.error(
            f"Data file not found at: {_DATA_PATH}\n"
            "Please ensure the CSV is in the same folder as regime_app.py."
        )
        st.stop()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)
    df.columns = ["Date", "Return", "CSV", "VIX", "Turbulence"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helper: fit GMM
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fitting GMM…")
def fit_gmm(X_raw, n_components, cov_type, random_state=42, n_init=20):
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=cov_type,
        n_init=n_init,
        random_state=random_state,
        max_iter=500,
    )
    gmm.fit(X)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)
    return gmm, scaler, labels, probs


def order_regimes_by_return(labels, returns):
    """Re-order regime labels so that the regime with the highest
    mean return is labelled 0 (Bull), next is 1, etc."""
    unique = np.unique(labels)
    means = {k: returns[labels == k].mean() for k in unique}
    ordered = sorted(means, key=means.get, reverse=True)
    mapping = {old: new for new, old in enumerate(ordered)}
    return np.array([mapping[l] for l in labels]), mapping


# ─────────────────────────────────────────────────────────────────────────────
# Helper: fit HMM / HSMM
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fitting HMM…")
def fit_hmm(X_raw, n_components, n_iter=200, random_state=42):
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X)
    labels = model.predict(X)
    return model, scaler, labels


def compute_sojourn_times(labels):
    """Compute list of (regime, duration) for consecutive runs."""
    sojourns = []
    current = labels[0]
    count = 1
    for i in range(1, len(labels)):
        if labels[i] == current:
            count += 1
        else:
            sojourns.append((current, count))
            current = labels[i]
            count = 1
    sojourns.append((current, count))
    return sojourns


def fit_negbin_durations(sojourns, n_components):
    """Fit Negative Binomial to sojourn times per regime."""
    params = {}
    for k in range(n_components):
        durs = [d for (r, d) in sojourns if r == k]
        if len(durs) < 3:
            params[k] = {"r": 1, "p": 0.5, "mean": float(np.mean(durs)) if len(durs) > 0 else 1.0, "durations": list(durs)}
            continue
        durs_arr = np.array(durs)
        mean_d = float(durs_arr.mean())
        var_d = float(durs_arr.var())
        if var_d <= mean_d:
            var_d = mean_d * 1.1  # NegBin requires var > mean; 10% margin is scale-invariant
        p_hat = mean_d / var_d
        r_hat = mean_d * p_hat / (1 - p_hat)
        r_hat = max(r_hat, 0.5)
        p_hat = min(max(p_hat, 0.01), 0.99)
        params[k] = {"r": r_hat, "p": p_hat, "mean": mean_d, "durations": list(durs)}
    return params


def hsmm_viterbi_duration(hmm_labels, negbin_params, n_components):
    """Refine HMM labels using duration-dependent transition probabilities.
    Uses a simple forward pass that penalises staying too long or too short."""
    refined = hmm_labels.copy()
    sojourns = compute_sojourn_times(hmm_labels)

    idx = 0
    for regime, dur in sojourns:
        nb = negbin_params.get(regime, {"r": 1, "p": 0.5})
        r, p = nb["r"], nb["p"]
        # If duration is extremely unlikely under the NegBin, split or merge
        if dur > 1:
            survival_prob = 1 - nbinom.cdf(dur, r, p)
            if survival_prob < 0.01 and dur > nb["mean"] * 2.5:
                # Run is implausibly long — reassign second half to next regime
                split = idx + dur // 2
                next_regime = (regime + 1) % n_components
                refined[split: idx + dur] = next_regime
        idx += dur

    return refined


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Optimal TE
# ─────────────────────────────────────────────────────────────────────────────
def compute_optimal_te(ic_k, csv_k, pi_k, te_total, te_floor):
    """Closed-form optimal TE per regime (floored)."""
    K = len(ic_k)
    theta_k = np.array(ic_k) * np.array(csv_k)
    pi = np.array(pi_k)

    # Unconstrained solution
    pi_theta = pi * theta_k
    norm_val = np.sqrt(np.sum(pi_theta**2))
    if norm_val < 1e-12:
        return np.full(K, te_total / np.sqrt(K)), theta_k, 0.0

    te_star = (pi_theta / norm_val) * te_total

    # Apply floor
    te_star = np.maximum(te_star, te_floor)

    # Rescale to meet budget (approximate)
    current_norm = np.sqrt(np.sum(te_star**2))
    if current_norm > 0:
        te_star = te_star * (te_total / current_norm)
        te_star = np.maximum(te_star, te_floor)

    # IR
    alpha = np.sum(pi * theta_k * te_star)
    te_actual = np.sqrt(np.sum(te_star**2))
    ir = alpha / te_actual if te_actual > 0 else 0

    return te_star, theta_k, ir


def simulate_paths(
    returns, regime_labels, ic_k, csv_k, pi_k,
    te_fixed, te_dynamic, n_paths=500, seed=42,
    progress_callback=None,
):
    """Monte-Carlo: generate n_paths by sampling regime sequences and returns.

    Units / dimensional contract
    ----------------------------
    ic_k   : decimal (e.g. 0.069 for 6.9% IC)
    csv_k  : cross-sectional volatility in the same units as returns column
    te_fixed / te_dynamic : annualised tracking error in %; divided by 100
                            inside this function to obtain a decimal fraction
    theta_k = IC × CSV    : expected active return per unit of TE (per period)
    alpha per period      : theta_k × (TE / 100)
    noise std per period  : (TE / 100) × 0.5  — represents unexplained active-
                            return variance (the non-IC component of TE)
    """
    rng = np.random.RandomState(seed)
    K = len(ic_k)
    T = len(returns)

    # Regime-conditional return stats
    regime_stats = {}
    for k in range(K):
        mask = regime_labels == k
        if mask.sum() > 0:
            regime_stats[k] = {"mean": returns[mask].mean(), "std": returns[mask].std()}
        else:
            regime_stats[k] = {"mean": 0, "std": 0.01}

    theta_k = np.array(ic_k) * np.array(csv_k)

    results_fixed = []
    results_dynamic = []
    ts_fixed = []    # shape (n_paths, T) — cumulative alpha at each period
    ts_dynamic = []

    for i in range(n_paths):
        # Random regime sequence (draw from stationary probabilities)
        regimes = rng.choice(K, size=T, p=pi_k)

        cum_fixed = 0.0
        cum_dynamic = 0.0
        cum_fixed_series = []
        cum_dynamic_series = []

        for t in range(T):
            k = regimes[t]
            # Base return (random draw from regime distribution)
            base = rng.normal(regime_stats[k]["mean"], regime_stats[k]["std"])

            # Active return = IC_k * CSV_k * TE + noise
            noise_fixed = rng.normal(0, te_fixed / 100)
            noise_dynamic = rng.normal(0, te_dynamic[k] / 100)

            alpha_fixed = theta_k[k] * (te_fixed / 100) + noise_fixed * 0.5
            alpha_dynamic = theta_k[k] * (te_dynamic[k] / 100) + noise_dynamic * 0.5

            cum_fixed += alpha_fixed
            cum_dynamic += alpha_dynamic
            cum_fixed_series.append(cum_fixed)
            cum_dynamic_series.append(cum_dynamic)

        results_fixed.append(cum_fixed)
        results_dynamic.append(cum_dynamic)
        ts_fixed.append(cum_fixed_series)
        ts_dynamic.append(cum_dynamic_series)
        if progress_callback is not None:
            progress_callback((i + 1) / n_paths)

    return (
        np.array(results_fixed),
        np.array(results_dynamic),
        np.array(ts_fixed),    # (n_paths, T)
        np.array(ts_dynamic),  # (n_paths, T)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────
st.title("Optimal Conditional TE Policy in Regime-Switching Markets")
st.caption("Interactive research companion — Mzobe & Flint | SAFA Conference 2026")

st.markdown("""
<div class="explanation-box">
<b>What is this app?</b> This tool accompanies a research paper that argues fund managers
should <i>not</i> keep their risk level (Tracking Error) fixed. Instead, they should
adjust it based on market conditions — taking more risk when their skill and market
opportunities are high, and less when conditions are poor. The app walks you through
the three analytical steps used to build and test this strategy.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3 = st.tabs([
    "📈 Data Explorer",
    "🔬 Step 1: GMM Regime Discovery",
    "⏱️ Step 2: HSMM Duration Model",
    "🎯 Step 3: Optimal TE & Simulation",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 0 — Data Explorer
# ═════════════════════════════════════════════════════════════════════════════
with tab0:
    st.header("Raw Data Explorer")
    st.markdown("""
    <div class="explanation-box">
    <b>The dataset</b> contains monthly observations of the MSCI All-Country World Index
    (ACWI) from January 2000 to December 2025 — roughly 25 years of global equity data.
    Each row has four variables:

    - **Return**: Monthly return of the MSCI ACWI index (how much the market moved)
    - **CSV** (Cross-Sectional Volatility): How spread out individual stock returns are.
      High CSV = stocks behaving very differently = more opportunity for stock pickers.
    - **VIX**: The "fear gauge" — market's expectation of volatility over the next 30 days.
    - **Turbulence Index**: A statistical measure (Mahalanobis distance) of how unusual
      today's market is compared to history. High = stressed markets.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", len(df))
    c2.metric("Date range", f"{df['Date'].min():%b %Y} – {df['Date'].max():%b %Y}")
    c3.metric("Avg monthly return", f"{df['Return'].mean():.2%}")
    c4.metric("Avg CSV", f"{df['CSV'].mean():.4f}")

    # Time-series plots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=["Monthly Return", "Cross-Sectional Volatility (CSV)",
                                        "VIX Index", "Turbulence Index"])
    fig.add_trace(go.Bar(x=df["Date"], y=df["Return"], marker_color=np.where(df["Return"] >= 0, "#2ecc71", "#e74c3c"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["CSV"], mode="lines", line=dict(color="#3498db", width=1.5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["VIX"], mode="lines", line=dict(color="#f39c12", width=1.5), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Turbulence"], mode="lines", line=dict(color="#9b59b6", width=1.5), showlegend=False), row=4, col=1)
    fig.update_layout(height=700, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View raw data table"):
        st.dataframe(df.style.format({"Return": "{:.4f}", "CSV": "{:.6f}", "VIX": "{:.2f}", "Turbulence": "{:.4f}"}), use_container_width=True)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    st.markdown("""
    <div class="explanation-box">
    Correlations tell us how the variables move together. A high correlation between VIX
    and Turbulence (both measure "stress") is expected. If CSV correlates with VIX, it
    means stock dispersion rises when fear rises — which makes sense because panicky
    markets create more divergence between winners and losers.
    </div>
    """, unsafe_allow_html=True)
    corr = df[["Return", "CSV", "VIX", "Turbulence"]].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, aspect="auto")
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — GMM Regime Discovery
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Step 1: Gaussian Mixture Model (GMM) Regime Discovery")
    st.markdown("""
    <div class="explanation-box">
    <b>What is a GMM?</b> A Gaussian Mixture Model is a statistical method that assumes
    your data comes from a <i>mixture</i> of several different "bell curve" (Gaussian)
    distributions. Each distribution represents a market regime (e.g., Bull, Tranquil, Bear).

    <b>What does it do here?</b> The GMM looks at each month's Return, CSV, VIX, and
    Turbulence values and asks: "Which of K regimes most likely generated these numbers?"
    It then assigns each month a probability of belonging to each regime.

    <b>Key limitation:</b> GMM is <i>memoryless</i> — it classifies each month independently,
    with no knowledge of what happened last month. That's why we need Step 2 (HSMM).
    </div>
    """, unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Step 1: GMM Settings")
        n_regimes = st.slider("Number of regimes (K)", 2, 5, 3, key="gmm_k",
                              help="The paper uses K=3 (Bull, Tranquil, Bear)")
        cov_type = st.selectbox("Covariance type", ["full", "tied", "diag", "spherical"],
                                index=0, key="gmm_cov",
                                help="'full' lets each regime have its own shape. 'diag' forces axis-aligned clusters.")
        features = st.multiselect("Features for clustering",
                                  ["Return", "CSV", "VIX", "Turbulence"],
                                  default=["Return", "CSV", "VIX", "Turbulence"],
                                  key="gmm_features")
        winsor_pct = st.slider("Winsorisation percentile", 0, 10, 1, key="gmm_winsor",
                               help="Clip extreme values at this percentile (paper uses 1st/99th)")

    if len(features) < 1:
        st.warning("Select at least one feature.")
        st.stop()

    # Prepare data
    X_raw = df[features].values.copy()
    if winsor_pct > 0:
        for col in range(X_raw.shape[1]):
            lo, hi = np.percentile(X_raw[:, col], [winsor_pct, 100 - winsor_pct])
            X_raw[:, col] = np.clip(X_raw[:, col], lo, hi)

    # Fit
    gmm, scaler, labels_raw, probs = fit_gmm(X_raw, n_regimes, cov_type)
    labels, mapping = order_regimes_by_return(labels_raw, df["Return"].values)

    # Reorder probs columns too
    probs_ordered = np.zeros_like(probs)
    for old, new in mapping.items():
        probs_ordered[:, new] = probs[:, old]

    regime_names = {i: REGIME_NAMES_DEFAULT.get(i, f"Regime {i}") for i in range(n_regimes)}

    # Store in session state
    st.session_state["gmm_labels"] = labels
    st.session_state["gmm_probs"] = probs_ordered
    st.session_state["n_regimes"] = n_regimes
    st.session_state["regime_names"] = regime_names

    # ── Metrics ──
    st.subheader("Model Fit")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("BIC", f"{gmm.bic(scaler.transform(X_raw)):,.0f}",
               help="Bayesian Information Criterion — lower is better")
    mc2.metric("AIC", f"{gmm.aic(scaler.transform(X_raw)):,.0f}",
               help="Akaike Information Criterion — lower is better")
    mc3.metric("Log-likelihood", f"{gmm.score(scaler.transform(X_raw)) * len(X_raw):,.1f}")

    st.markdown("""
    <div class="explanation-box">
    <b>BIC & AIC</b> help you decide how many regimes to use. They balance model fit
    against complexity — adding more regimes always fits better, but at some point the
    extra complexity isn't worth it. <b>Lower values = better model.</b> Try changing
    K in the sidebar and watch these numbers.
    </div>
    """, unsafe_allow_html=True)

    # ── BIC comparison across K ──
    st.subheader("Optimal Number of Regimes")
    bics, aics = [], []
    for k_test in range(2, 6):
        gmm_test, sc_test, _, _ = fit_gmm(X_raw, k_test, cov_type)
        X_sc = sc_test.transform(X_raw)
        bics.append(gmm_test.bic(X_sc))
        aics.append(gmm_test.aic(X_sc))

    fig_bic = go.Figure()
    fig_bic.add_trace(go.Scatter(x=list(range(2, 6)), y=bics, mode="lines+markers", name="BIC", line=dict(color="#e74c3c")))
    fig_bic.add_trace(go.Scatter(x=list(range(2, 6)), y=aics, mode="lines+markers", name="AIC", line=dict(color="#3498db")))
    fig_bic.update_layout(xaxis_title="Number of Regimes (K)", yaxis_title="Score (lower = better)", height=350, title="BIC & AIC vs Number of Regimes")
    st.plotly_chart(fig_bic, use_container_width=True)

    # ── Regime timeline ──
    st.subheader("Regime Classification Over Time")
    fig_timeline = go.Figure()
    for k in range(n_regimes):
        mask = labels == k
        fig_timeline.add_trace(go.Bar(
            x=df["Date"][mask], y=df["Return"][mask],
            name=regime_names[k],
            marker_color=REGIME_COLOURS[k],
        ))
    fig_timeline.update_layout(barmode="overlay", height=400, yaxis_title="Monthly Return",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Background shading version
    st.subheader("Regime Periods (Background Shading)")
    fig_shade = go.Figure()
    fig_shade.add_trace(go.Scatter(x=df["Date"], y=df["Return"], mode="lines", line=dict(color="black", width=1), name="Return"))
    # Add regime backgrounds
    sojourns = compute_sojourn_times(labels)
    idx = 0
    for regime, dur in sojourns:
        start_date = df["Date"].iloc[idx]
        end_date = df["Date"].iloc[min(idx + dur - 1, len(df) - 1)]
        colour = REGIME_COLOURS[regime]
        fig_shade.add_vrect(x0=start_date, x1=end_date, fillcolor=colour, opacity=0.2, line_width=0)
        idx += dur
    fig_shade.update_layout(height=400, yaxis_title="Monthly Return")
    st.plotly_chart(fig_shade, use_container_width=True)

    # ── Regime statistics ──
    st.subheader("Regime Statistics")
    stats_rows = []
    for k in range(n_regimes):
        mask = labels == k
        r = df["Return"][mask]
        c = df["CSV"][mask]
        v = df["VIX"][mask]
        t = df["Turbulence"][mask]
        stats_rows.append({
            "Regime": regime_names[k],
            "Count": int(mask.sum()),
            "Probability": f"{mask.mean():.1%}",
            "Mean Return": f"{r.mean():.2%}",
            "Std Return": f"{r.std():.2%}",
            "Mean CSV": f"{c.mean():.4f}",
            "Mean VIX": f"{v.mean():.1f}",
            "Mean Turbulence": f"{t.mean():.2f}",
        })
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="explanation-box">
    <b>What to look for:</b> The Bull regime should have the highest mean return and the
    lowest volatility/turbulence. The Bear regime should have negative returns and high
    CSV (dispersion). High CSV in Bear markets is the key insight — it means there's more
    room for skilled managers to outperform when markets are stressed.
    </div>
    """, unsafe_allow_html=True)

    # ── Probability heatmap ──
    st.subheader("Regime Membership Probabilities")
    st.markdown("Each month gets a probability of belonging to each regime. Hard boundaries are rare — transitions are gradual.")
    fig_prob = go.Figure()
    for k in range(n_regimes):
        fig_prob.add_trace(go.Scatter(
            x=df["Date"], y=probs_ordered[:, k],
            mode="lines", fill="tonexty" if k > 0 else "tozeroy",
            name=regime_names[k], line=dict(width=0.5),
            fillcolor=hex_to_rgba(REGIME_COLOURS[k], 0.5),
            stackgroup="one",
        ))
    fig_prob.update_layout(height=350, yaxis_title="Probability", yaxis_range=[0, 1])
    st.plotly_chart(fig_prob, use_container_width=True)

    # ── Feature distributions by regime ──
    st.subheader("Feature Distributions by Regime")
    feat_to_show = st.selectbox("Select feature", features, key="gmm_feat_dist")
    fig_dist = go.Figure()
    for k in range(n_regimes):
        mask = labels == k
        fig_dist.add_trace(go.Histogram(
            x=df[feat_to_show][mask], name=regime_names[k],
            marker_color=REGIME_COLOURS[k], opacity=0.6, nbinsx=30,
        ))
    fig_dist.update_layout(barmode="overlay", height=350, xaxis_title=feat_to_show, yaxis_title="Count")
    st.plotly_chart(fig_dist, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — HSMM Duration Model
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Step 2: Hidden Semi-Markov Model (HSMM)")
    st.markdown("""
    <div class="explanation-box">
    <b>Why do we need this?</b> The GMM in Step 1 is "memoryless" — it looks at each month
    in isolation. But in reality, regimes <i>persist</i>: if we're in a bear market this month,
    we'll probably still be in one next month. The HSMM adds this sense of time.

    <b>How does it work?</b> Two layers:
    1. A <b>Hidden Markov Model (HMM)</b> learns transition probabilities between regimes
       (e.g., "if we're in Bull, there's a 5% chance of switching to Bear next month").
    2. A <b>Negative Binomial duration model</b> is fitted to how long each regime lasts.
       Unlike a simple geometric distribution (which the HMM assumes), the Negative Binomial
       can capture the fact that bear markets, for example, tend to last 6–12 months —
       not just randomly flip each month.

    <b>Think of it like weather:</b> The HMM says "it's winter." The duration model adds
    "and we've been in winter for 2 months, so we probably have 1–2 months left."
    </div>
    """, unsafe_allow_html=True)

    if "gmm_labels" not in st.session_state:
        st.warning("Run Step 1 first (click on the GMM tab).")
        st.stop()

    n_k = st.session_state["n_regimes"]
    regime_names = st.session_state["regime_names"]

    with st.sidebar:
        st.header("⚙️ Step 2: HSMM Settings")
        hmm_n_iter = st.slider("HMM iterations", 50, 500, 200, step=50, key="hmm_iter")

    # Features
    feats_hmm = [f for f in ["Return", "CSV", "VIX", "Turbulence"] if f in df.columns]
    X_hmm = df[feats_hmm].values.copy()

    # Fit HMM
    hmm_model, hmm_scaler, hmm_labels_raw = fit_hmm(X_hmm, n_k, n_iter=hmm_n_iter)
    hmm_labels, hmm_mapping = order_regimes_by_return(hmm_labels_raw, df["Return"].values)

    st.session_state["hmm_labels"] = hmm_labels

    # Transition matrix
    st.subheader("Transition Probability Matrix")
    st.markdown("""
    <div class="explanation-box">
    This matrix shows the probability of moving from one regime to another in a single month.
    Read it as: "If I'm in row regime this month, what's the probability I'll be in column regime
    next month?" High diagonal values mean regimes are persistent (sticky).
    </div>
    """, unsafe_allow_html=True)

    # Reorder transition matrix
    trans_raw = hmm_model.transmat_
    trans = np.zeros_like(trans_raw)
    inv_map = {v: k for k, v in hmm_mapping.items()}
    for i in range(n_k):
        for j in range(n_k):
            trans[i, j] = trans_raw[inv_map[i], inv_map[j]]

    trans_df = pd.DataFrame(trans,
                            index=[regime_names[i] for i in range(n_k)],
                            columns=[regime_names[i] for i in range(n_k)])

    fig_trans = px.imshow(trans_df, text_auto=".1%", color_continuous_scale="Blues",
                          zmin=0, zmax=1, aspect="auto")
    fig_trans.update_layout(height=350, title="Monthly Transition Probabilities")
    st.plotly_chart(fig_trans, use_container_width=True)

    # Stationary probabilities
    eigenvalues, eigenvectors = np.linalg.eig(trans.T)
    idx_one = np.argmin(np.abs(eigenvalues - 1))
    stationary = np.real(eigenvectors[:, idx_one])
    stationary = stationary / stationary.sum()

    st.subheader("Stationary (Long-Run) Regime Probabilities")
    stat_cols = st.columns(n_k)
    for k in range(n_k):
        stat_cols[k].metric(regime_names[k], f"{stationary[k]:.1%}")

    # Sojourn / Duration analysis
    st.subheader("Regime Duration Analysis")
    sojourns = compute_sojourn_times(hmm_labels)
    negbin_params = fit_negbin_durations(sojourns, n_k)

    # Duration stats table
    dur_rows = []
    for k in range(n_k):
        nb = negbin_params[k]
        durs = nb["durations"]
        dur_rows.append({
            "Regime": regime_names[k],
            "# Episodes": len(durs),
            "Mean duration (months)": f"{nb['mean']:.1f}",
            "Min": min(durs) if len(durs) > 0 else 0,
            "Max": max(durs) if len(durs) > 0 else 0,
            "NegBin r": f"{nb['r']:.2f}",
            "NegBin p": f"{nb['p']:.2f}",
        })
    st.dataframe(pd.DataFrame(dur_rows), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="explanation-box">
    <b>Reading the duration table:</b>
    - <b>Mean duration</b> tells you how long each regime typically lasts.
    - <b>NegBin r, p</b> are the parameters of the Negative Binomial distribution fitted to
      each regime's durations. Higher <i>r</i> = more peaked around the mean; lower <i>p</i> = more spread out.
    - Bull markets tend to last longest; Bear markets are shorter but more intense.
    </div>
    """, unsafe_allow_html=True)

    # Duration histogram with fitted NegBin overlay
    st.subheader("Duration Distributions (Observed vs Fitted Negative Binomial)")
    dur_cols = st.columns(min(n_k, 3))
    for k in range(n_k):
        nb = negbin_params[k]
        durs = nb["durations"]
        with dur_cols[k % len(dur_cols)]:
            fig_dur = go.Figure()
            if len(durs) > 0:
                max_d = max(durs) + 5
                fig_dur.add_trace(go.Histogram(
                    x=durs, nbinsx=max(5, max_d // 2),
                    name="Observed", marker_color=REGIME_COLOURS[k], opacity=0.7,
                    histnorm="probability",
                ))
                # NegBin overlay
                x_nb = np.arange(1, max_d + 1)
                y_nb = nbinom.pmf(x_nb, nb["r"], nb["p"])
                fig_dur.add_trace(go.Scatter(
                    x=x_nb, y=y_nb, mode="lines",
                    name="NegBin fit", line=dict(color="black", width=2, dash="dash"),
                ))
            fig_dur.update_layout(
                title=f"{regime_names[k]}",
                xaxis_title="Duration (months)", yaxis_title="Probability",
                height=300, showlegend=True,
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_dur, use_container_width=True)

    # Compare GMM vs HMM regimes
    st.subheader("GMM vs HMM Regime Comparison")
    st.markdown("How often do the two methods agree on the regime classification?")

    gmm_l = st.session_state["gmm_labels"]
    agreement = (gmm_l == hmm_labels).mean()
    st.metric("Agreement rate", f"{agreement:.1%}")

    fig_comp = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=["GMM Regimes", "HMM Regimes"],
                             vertical_spacing=0.08)
    for k in range(n_k):
        mask_g = gmm_l == k
        mask_h = hmm_labels == k
        fig_comp.add_trace(go.Bar(x=df["Date"][mask_g], y=df["Return"][mask_g],
                                  name=regime_names[k], marker_color=REGIME_COLOURS[k],
                                  showlegend=True if k == 0 else False, legendgroup=str(k)), row=1, col=1)
        fig_comp.add_trace(go.Bar(x=df["Date"][mask_h], y=df["Return"][mask_h],
                                  name=regime_names[k], marker_color=REGIME_COLOURS[k],
                                  showlegend=False, legendgroup=str(k)), row=2, col=1)
    fig_comp.update_layout(barmode="overlay", height=500)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("""
    <div class="explanation-box">
    <b>Why do they differ?</b> The HMM incorporates <i>sequential information</i> — it knows
    that being in a regime last month makes it likely to be in the same regime this month.
    The GMM doesn't know this. So the HMM tends to produce "smoother" regime sequences
    with fewer rapid switches. This is more realistic for financial markets.
    </div>
    """, unsafe_allow_html=True)

    # Regime survival curves
    st.subheader("Regime Survival Curves")
    st.markdown("""
    <div class="explanation-box">
    A survival curve shows the probability that a regime lasts <i>at least</i> D months.
    It starts at 100% (the regime just started) and decays over time. Steeper decay = shorter
    regime. The Negative Binomial fit (dashed) should roughly match the observed data (solid).
    </div>
    """, unsafe_allow_html=True)

    fig_surv = go.Figure()
    for k in range(n_k):
        nb = negbin_params[k]
        durs = nb["durations"]
        if len(durs) < 2:
            continue
        max_d = max(durs) + 5
        # Empirical survival
        x_vals = np.arange(1, max_d + 1)
        surv_emp = np.array([np.mean(np.array(durs) >= d) for d in x_vals])
        fig_surv.add_trace(go.Scatter(x=x_vals, y=surv_emp, mode="lines",
                                       name=f"{regime_names[k]} (observed)",
                                       line=dict(color=REGIME_COLOURS[k], width=2)))
        # NegBin survival
        surv_nb = 1 - nbinom.cdf(x_vals, nb["r"], nb["p"])
        fig_surv.add_trace(go.Scatter(x=x_vals, y=surv_nb, mode="lines",
                                       name=f"{regime_names[k]} (NegBin)",
                                       line=dict(color=REGIME_COLOURS[k], width=2, dash="dash")))
    fig_surv.update_layout(height=400, xaxis_title="Duration (months)",
                           yaxis_title="Survival Probability", yaxis_range=[0, 1.05])
    st.plotly_chart(fig_surv, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Optimal TE & Simulation
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Step 3: Optimal TE Allocation & Monte-Carlo Simulation")
    st.markdown("""
    <div class="explanation-box">
    <b>This is where everything comes together.</b> Now that we know the market regimes
    (Step 1) and how long they last (Step 2), we can answer the key question:

    <i>How much risk (Tracking Error) should a fund manager take in each regime?</i>

    The answer comes from a mathematical formula:
    <b>Optimal TE in regime k = (probability × skill × opportunity) / normalising constant × total risk budget</b>

    Where:
    - <b>Skill</b> = Information Coefficient (IC) — how good are the manager's stock picks in this regime?
    - <b>Opportunity</b> = Cross-Sectional Volatility (CSV) — how spread out are stock returns?
    - <b>Theta (θ)</b> = IC × CSV = the "skill-opportunity product"

    When θ is high → take more risk. When θ is low or negative → minimise risk.
    </div>
    """, unsafe_allow_html=True)

    if "hmm_labels" not in st.session_state:
        st.warning("Complete Steps 1 and 2 first.")
        st.stop()

    n_k = st.session_state["n_regimes"]
    regime_names = st.session_state["regime_names"]
    labels_for_te = st.session_state["hmm_labels"]

    # Compute actual regime stats
    pi_actual = np.array([np.mean(labels_for_te == k) for k in range(n_k)])
    csv_actual = np.array([df["CSV"][labels_for_te == k].mean() for k in range(n_k)])

    with st.sidebar:
        st.header("⚙️ Step 3: TE Settings")
        st.markdown("**Information Coefficients (IC)**")
        st.caption("Skill measure per regime: -1 (worst) to +1 (best). Paper values: Bull=-1.75%, Tranquil=-0.05%, Bear=+6.9%")

        ic_values = []
        default_ics = [-0.0175, -0.0005, 0.069, 0.0, 0.0]
        for k in range(n_k):
            ic_val = st.slider(
                f"IC: {regime_names[k]}",
                -0.15, 0.15, default_ics[k] if k < len(default_ics) else 0.0,
                step=0.005, format="%.3f", key=f"ic_{k}",
            )
            ic_values.append(ic_val)

        st.markdown("---")
        te_total = st.slider("Total TE budget (%)", 1.0, 20.0, 6.0, step=0.5, key="te_total",
                             help="The overall risk budget. Paper uses 6%.")
        te_floor = st.slider("TE floor (%)", 0.0, 5.0, 2.0, step=0.5, key="te_floor",
                             help="Minimum TE in any regime. Paper uses 2%.")
        te_fixed = st.slider("Fixed TE for comparison (%)", 1.0, 20.0, 6.0, step=0.5, key="te_fixed",
                             help="The static TE strategy to compare against.")
        n_simulations = st.slider("Number of simulation paths", 100, 2000, 500, step=100, key="n_sims")

    # ── Compute optimal TE ──
    te_star, theta_k, ir_dynamic = compute_optimal_te(
        ic_values, csv_actual, pi_actual, te_total, te_floor
    )

    # Fixed IR
    theta_arr = np.array(ic_values) * csv_actual
    ir_fixed = np.sum(pi_actual * theta_arr * (te_fixed / 100)) / (te_fixed / 100)

    # ── Display regime parameters ──
    st.subheader("Regime Parameters & Optimal TE")

    param_rows = []
    for k in range(n_k):
        param_rows.append({
            "Regime": regime_names[k],
            "π (probability)": f"{pi_actual[k]:.1%}",
            "IC (skill)": f"{ic_values[k]:.3f}",
            "CSV (opportunity)": f"{csv_actual[k]:.4f}",
            "θ = IC × CSV": f"{theta_k[k]:.5f}",
            "Optimal TE (%)": f"{te_star[k]:.2f}",
            "Fixed TE (%)": f"{te_fixed:.2f}",
        })
    st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)

    # Visual comparison
    fig_te = go.Figure()
    x_labels = [regime_names[k] for k in range(n_k)]
    fig_te.add_trace(go.Bar(x=x_labels, y=te_star, name="Optimal (Dynamic) TE",
                            marker_color="#2ecc71", text=[f"{v:.1f}%" for v in te_star], textposition="auto"))
    fig_te.add_trace(go.Bar(x=x_labels, y=[te_fixed] * n_k, name="Fixed TE",
                            marker_color="#95a5a6", text=[f"{te_fixed:.1f}%"] * n_k, textposition="auto"))
    fig_te.update_layout(barmode="group", height=400, yaxis_title="Tracking Error (%)",
                         title="Optimal vs Fixed TE by Regime")
    st.plotly_chart(fig_te, use_container_width=True)

    st.markdown("""
    <div class="explanation-box">
    <b>Key insight:</b> Notice how the dynamic strategy allocates much more TE to the
    Bear regime (where skill is high and opportunities are abundant) and minimises TE in
    Bull/Tranquil regimes (where skill is negative — stock picks actually hurt performance).
    This is the core of the paper: <i>be brave at the right time, not all the time</i>.
    </div>
    """, unsafe_allow_html=True)

    # ── Theta decomposition ──
    st.subheader("Skill × Opportunity Decomposition (θ = IC × CSV)")
    fig_theta = make_subplots(rows=1, cols=3, subplot_titles=["IC (Skill)", "CSV (Opportunity)", "θ (Product)"])
    for k in range(n_k):
        fig_theta.add_trace(go.Bar(x=[regime_names[k]], y=[ic_values[k]],
                                    marker_color=REGIME_COLOURS[k], showlegend=False), row=1, col=1)
        fig_theta.add_trace(go.Bar(x=[regime_names[k]], y=[csv_actual[k]],
                                    marker_color=REGIME_COLOURS[k], showlegend=False), row=1, col=2)
        fig_theta.add_trace(go.Bar(x=[regime_names[k]], y=[theta_k[k]],
                                    marker_color=REGIME_COLOURS[k], showlegend=False), row=1, col=3)
    fig_theta.update_layout(height=350)
    st.plotly_chart(fig_theta, use_container_width=True)

    # ── IR comparison ──
    st.subheader("Information Ratio Comparison")
    ir_cols = st.columns(3)
    ir_cols[0].metric("IR (Fixed TE)", f"{ir_fixed:.4f}")
    ir_cols[1].metric("IR (Dynamic TE)", f"{ir_dynamic:.4f}")
    ir_cols[2].metric("IR Improvement", f"{ir_dynamic - ir_fixed:.4f}",
                      delta=f"{ir_dynamic - ir_fixed:+.4f}")

    st.markdown("""
    <div class="explanation-box">
    <b>Information Ratio (IR)</b> = expected active return ÷ tracking error. It's the
    "miles per gallon" of investing — how much return per unit of risk. A higher IR means
    the manager is more efficient with their risk budget. The dynamic strategy should
    show a meaningfully higher IR because it avoids wasting risk in bad environments.
    </div>
    """, unsafe_allow_html=True)

    # ── Monte-Carlo Simulation ──
    st.subheader("Monte-Carlo Simulation")
    st.markdown(f"Simulating **{n_simulations}** market paths to test the strategy...")

    _progress = st.progress(0, text="Starting simulation...")

    def _update(frac):
        n_done = int(frac * n_simulations)
        _progress.progress(frac, text=f"Simulating path {n_done} / {n_simulations}…")

    results_fixed, results_dynamic, ts_fixed, ts_dynamic = simulate_paths(
        df["Return"].values, labels_for_te, ic_values, csv_actual, pi_actual,
        te_fixed, te_star, n_paths=n_simulations, seed=42,
        progress_callback=_update,
    )
    _progress.empty()

    win_rate = (results_dynamic > results_fixed).mean()

    sim_cols = st.columns(3)
    sim_cols[0].metric("Win rate (Dynamic > Fixed)", f"{win_rate:.1%}")
    sim_cols[1].metric("Mean cumulative alpha (Dynamic)", f"{results_dynamic.mean():.4f}")
    sim_cols[2].metric("Mean cumulative alpha (Fixed)", f"{results_fixed.mean():.4f}")

    p5_d, p95_d = np.percentile(results_dynamic, [5, 95])
    p5_f, p95_f = np.percentile(results_fixed, [5, 95])
    pct_cols = st.columns(4)
    pct_cols[0].metric("5th %ile (Dynamic)", f"{p5_d:.4f}")
    pct_cols[1].metric("95th %ile (Dynamic)", f"{p95_d:.4f}")
    pct_cols[2].metric("5th %ile (Fixed)", f"{p5_f:.4f}")
    pct_cols[3].metric("95th %ile (Fixed)", f"{p95_f:.4f}")

    # Distribution of outcomes
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(x=results_dynamic, name="Dynamic TE",
                                   marker_color="#2ecc71", opacity=0.6, nbinsx=50))
    fig_mc.add_trace(go.Histogram(x=results_fixed, name="Fixed TE",
                                   marker_color="#95a5a6", opacity=0.6, nbinsx=50))
    fig_mc.update_layout(barmode="overlay", height=400,
                         xaxis_title="Cumulative Active Return",
                         yaxis_title="Frequency",
                         title="Distribution of Outcomes: Dynamic vs Fixed TE")
    st.plotly_chart(fig_mc, use_container_width=True)

    # IR distribution across paths
    fig_ir_dist = go.Figure()
    ir_diff = results_dynamic - results_fixed
    fig_ir_dist.add_trace(go.Histogram(x=ir_diff, nbinsx=50,
                                        marker_color=np.where(ir_diff >= 0, "#2ecc71", "#e74c3c").tolist(),
                                        showlegend=False))
    fig_ir_dist.add_vline(x=0, line_dash="dash", line_color="black")
    fig_ir_dist.update_layout(height=350,
                              xaxis_title="Dynamic Alpha − Fixed Alpha",
                              yaxis_title="Frequency",
                              title=f"Alpha Differential (Dynamic wins {win_rate:.0%} of paths)")
    st.plotly_chart(fig_ir_dist, use_container_width=True)

    # ── Fan chart: path distribution over time ──
    st.subheader("Cumulative Alpha Path Distribution Over Time")
    st.markdown("Each band shows how the spread of outcomes widens as paths diverge. The median lines show the central tendency.")
    months = np.arange(1, ts_dynamic.shape[1] + 1)
    pct_d = np.percentile(ts_dynamic, [5, 25, 50, 75, 95], axis=0)
    pct_f = np.percentile(ts_fixed,   [5, 25, 50, 75, 95], axis=0)

    fig_fan = go.Figure()

    # Dynamic outer band (5th–95th)
    fig_fan.add_trace(go.Scatter(
        x=np.concatenate([months, months[::-1]]),
        y=np.concatenate([pct_d[4], pct_d[0][::-1]]),
        fill='toself', fillcolor=hex_to_rgba('#2ecc71', 0.15),
        line=dict(color='rgba(0,0,0,0)'),
        name='Dynamic 5–95th %ile', legendgroup='dynamic',
    ))
    # Dynamic inner band (25th–75th)
    fig_fan.add_trace(go.Scatter(
        x=np.concatenate([months, months[::-1]]),
        y=np.concatenate([pct_d[3], pct_d[1][::-1]]),
        fill='toself', fillcolor=hex_to_rgba('#2ecc71', 0.35),
        line=dict(color='rgba(0,0,0,0)'),
        name='Dynamic 25–75th %ile', legendgroup='dynamic',
    ))
    # Dynamic median
    fig_fan.add_trace(go.Scatter(
        x=months, y=pct_d[2],
        line=dict(color='#2ecc71', width=2),
        name='Dynamic median', legendgroup='dynamic',
    ))

    # Fixed outer band (5th–95th)
    fig_fan.add_trace(go.Scatter(
        x=np.concatenate([months, months[::-1]]),
        y=np.concatenate([pct_f[4], pct_f[0][::-1]]),
        fill='toself', fillcolor=hex_to_rgba('#95a5a6', 0.15),
        line=dict(color='rgba(0,0,0,0)'),
        name='Fixed 5–95th %ile', legendgroup='fixed',
    ))
    # Fixed inner band (25th–75th)
    fig_fan.add_trace(go.Scatter(
        x=np.concatenate([months, months[::-1]]),
        y=np.concatenate([pct_f[3], pct_f[1][::-1]]),
        fill='toself', fillcolor=hex_to_rgba('#95a5a6', 0.35),
        line=dict(color='rgba(0,0,0,0)'),
        name='Fixed 25–75th %ile', legendgroup='fixed',
    ))
    # Fixed median
    fig_fan.add_trace(go.Scatter(
        x=months, y=pct_f[2],
        line=dict(color='#7f8c8d', width=2),
        name='Fixed median', legendgroup='fixed',
    ))

    fig_fan.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.4)
    fig_fan.update_layout(
        height=450, xaxis_title='Month', yaxis_title='Cumulative Active Alpha',
        title='Path Fan Chart — Green: Dynamic TE, Grey: Fixed TE',
    )
    st.plotly_chart(fig_fan, use_container_width=True)

    # ── Download buttons ──
    dl_col1, dl_col2 = st.columns(2)

    sim_df = pd.DataFrame({
        "path_id": range(1, len(results_dynamic) + 1),
        "cumulative_alpha_dynamic": results_dynamic,
        "cumulative_alpha_fixed": results_fixed,
        "alpha_differential": results_dynamic - results_fixed,
    })
    buf = io.StringIO()
    sim_df.to_csv(buf, index=False)
    dl_col1.download_button(
        "⬇ Download simulation results",
        data=buf.getvalue(),
        file_name="simulation_results.csv",
        mime="text/csv",
    )

    gmm_labels_dl = st.session_state["gmm_labels"]
    regime_df = pd.DataFrame({
        "Date": df["Date"].dt.strftime("%Y-%m-%d"),
        "Return": df["Return"],
        "GMM_regime": gmm_labels_dl,
        "GMM_regime_name": [regime_names[k] for k in gmm_labels_dl],
        "HMM_regime": labels_for_te,
        "HMM_regime_name": [regime_names[k] for k in labels_for_te],
    })
    buf2 = io.StringIO()
    regime_df.to_csv(buf2, index=False)
    dl_col2.download_button(
        "⬇ Download regime classifications",
        data=buf2.getvalue(),
        file_name="regime_classifications.csv",
        mime="text/csv",
    )

    st.markdown("""
    <div class="explanation-box">
    <b>Reading the simulation results:</b>
    - The <b>win rate</b> tells you in what percentage of simulated futures the dynamic
      strategy outperformed the fixed one.
    - The <b>histogram</b> shows the full distribution of outcomes. If the green (dynamic)
      distribution is shifted to the right of the grey (fixed), the dynamic strategy is
      adding value on average.
    - The <b>differential chart</b> shows the gap directly: values to the right of zero
      mean the dynamic strategy won; left means the fixed one did.
    </div>
    """, unsafe_allow_html=True)

    # ── Closed-form equation display ──
    st.subheader("The Mathematics")
    st.markdown("The closed-form solution for optimal TE in regime *k*:")
    st.latex(r"\mathrm{TE}_k^* = \frac{\pi_k \cdot \theta_k}{\left[\sum_{j=1}^{K} (\pi_j \theta_j)^2\right]^{1/2}} \cdot \overline{\mathrm{TE}}")
    st.markdown("Where θ_k = IC_k × σ_{D,k} is the skill-opportunity product.")
    st.markdown("The resulting optimal Information Ratio is:")
    st.latex(r"\mathrm{IR}^* = \left[\sum_{k=1}^{K} (\pi_k \cdot \mathrm{IC}_k \cdot \sigma_{D,k})^2\right]^{1/2}")

    st.markdown("""
    <div class="explanation-box">
    <b>In plain English:</b> The optimal amount of risk to take in any regime is proportional
    to your skill in that regime times the opportunity available, weighted by how often that
    regime occurs. The formula ensures you're not wasting risk budget in regimes where you
    can't add value, and concentrating it where you can.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.caption("Research companion app | Mzobe & Flint — *Optimal Conditional TE Policy in Regime-Switching Markets* | SAFA Conference 2026")
