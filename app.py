import os
import pathlib

import altair as alt
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from astro import AstroClass  # your original class, untouched

BASE = pathlib.Path(__file__).parent
GAMMA_COLOR, HADRON_COLOR = "#4C9AFF", "#FF6B6B"
COLOR_SCALE = alt.Scale(domain=["Gamma", "Hadron"], range=[GAMMA_COLOR, HADRON_COLOR])

FEATURE_INFO = {
    "fLength": "Major axis of the ellipse (mm)",
    "fWidth": "Minor axis of the ellipse (mm)",
    "fSize": "log10 of the sum of all pixel contents (photon count)",
    "fConc": "Ratio of the two highest pixels over fSize",
    "fConc1": "Ratio of the highest pixel over fSize",
    "fAsym": "Distance from highest pixel to centre, along the major axis (mm)",
    "fM3Long": "3rd root of third moment along the major axis (mm)",
    "fM3Trans": "3rd root of third moment along the minor axis (mm)",
    "fAlpha": "Angle of the major axis with the vector to the origin (deg)",
    "fDist": "Distance from the origin to the ellipse centre (mm)",
}

CSS = """
<style>
.hero {padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
       background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);}
.hero h1 {color: #fff; margin: 0; font-size: 2.1rem;}
.hero p {color: #cfe3ee; margin: .4rem 0 0 0; font-size: 1.05rem;}
div[data-testid="stMetric"] {background: rgba(128,128,128,.10); padding: 14px 18px;
                             border-radius: 12px;}
</style>
"""


class AstroApp(AstroClass):
    """Streamlit front-end built on top of AstroClass.

    Everything from AstroClass (df, divide, TTsplit, trainer_pipeline, ...) is
    inherited as-is; this subclass only adds the page builders.
    """

    def __init__(self):
        super().__init__()
        self.features = [c for c in self.numeric_features]
        # plotting copy with readable labels (self.df stays exactly as AstroClass made it)
        self.viz_df = self.df.assign(label=self.df["class"].map({"g": "Gamma", "h": "Hadron"}))

        # same split as in your __main__ (80/20, seed 42, stratified)
        self.x_train, self.x_test, self.y_train, self.y_test = self.TTsplit()
        self.model = self.load_or_train()
        self.proba_test = self.model.predict_proba(self.x_test)[:, 1]
        self.y_true = self.y_test.to_numpy()

    # ------------------------------------------------------------------ model
    def load_or_train(self):
        """Use the saved 'astroWX' pipeline; retrain with your pipeline if it
        is missing or can't be unpickled (e.g. scikit-learn version mismatch)."""
        path = BASE / "astroWX"
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                pass
        return self.trainer_pipeline(self.x_train, self.y_train)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def hero(title, subtitle):
        st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                    unsafe_allow_html=True)

    # ------------------------------------------------------------------ pages
    def page_overview(self, threshold):
        self.hero("Gamma vs Hadron Classification",
                  "Separating gamma-ray showers from cosmic-ray background in "
                  "Cherenkov telescope images.")

        df = self.df
        n_g, n_h = (df["class"] == "g").sum(), (df["class"] == "h").sum()
        pred = (self.proba_test >= 0.5).astype(int)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Events", f"{len(df):,}")
        c2.metric("Gamma / Hadron", f"{n_g:,} / {n_h:,}")
        c3.metric("Test accuracy", f"{accuracy_score(self.y_true, pred):.1%}")
        c4.metric("Test ROC-AUC", f"{roc_auc_score(self.y_true, self.proba_test):.3f}")

        left, right = st.columns([3, 2])
        with left:
            st.subheader("What is this about?")
            st.markdown(
                """
                The **MAGIC telescope** records the faint blue flash (Cherenkov light) produced
                when high-energy particles hit the atmosphere. Each flash leaves an elliptical
                image in the camera:

                - **Gamma rays** (signal) produce compact, regular, well-aligned ellipses.
                - **Hadrons** (background, cosmic rays) produce wider, messier, more scattered images.

                Ten image parameters (Hillas parameters) describe each event. The model
                learns to tell the two apart so astronomers can keep the real gamma signal.
                """
            )
            st.subheader("Pipeline")
            st.code("PowerTransformer(standardize=True)  →  LogisticRegression(C=0.8, saga)",
                    language="text")
        with right:
            st.subheader("Class balance")
            bal = self.viz_df["label"].value_counts().rename_axis("Class").reset_index(name="Events")
            st.altair_chart(
                alt.Chart(bal).mark_arc(innerRadius=60).encode(
                    theta="Events:Q",
                    color=alt.Color("Class:N", scale=COLOR_SCALE),
                    tooltip=["Class", "Events"],
                ).properties(height=300),
                use_container_width=True,
            )

        st.subheader("Feature glossary")
        st.dataframe(
            pd.DataFrame({"Feature": list(FEATURE_INFO), "Meaning": list(FEATURE_INFO.values())}),
            hide_index=True, use_container_width=True,
        )

    def page_explorer(self, threshold):
        self.hero("Data Explorer", "How do the image parameters differ between gammas and hadrons?")
        vdf = self.viz_df

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Distributions", "Feature vs feature", "Correlations", "Pairplot", "Data quality"])

        with tab1:
            feat = st.selectbox("Feature", self.features, index=self.features.index("fAlpha"))
            st.caption(FEATURE_INFO.get(feat, ""))
            dens = (
                alt.Chart(vdf)
                .transform_density(
                    feat, groupby=["label"], as_=[feat, "density"],
                    extent=[float(vdf[feat].quantile(0.005)), float(vdf[feat].quantile(0.995))])
                .mark_area(opacity=0.5, line=True)
                .encode(x=alt.X(f"{feat}:Q"), y="density:Q",
                        color=alt.Color("label:N", scale=COLOR_SCALE, title="Class"))
                .properties(height=350)
            )
            st.altair_chart(dens, use_container_width=True)
            st.dataframe(
                vdf.groupby("label")[feat].describe()[["mean", "std", "min", "50%", "max"]].round(3),
                use_container_width=True)

        with tab2:
            c1, c2, c3 = st.columns(3)
            fx = c1.selectbox("X axis", self.features, index=self.features.index("fLength"))
            fy = c2.selectbox("Y axis", self.features, index=self.features.index("fWidth"))
            n = c3.slider("Sample size", 200, 5000, 1500, 100)
            sample = vdf.sample(n, random_state=42)
            st.altair_chart(
                alt.Chart(sample).mark_circle(size=45, opacity=0.55).encode(
                    x=alt.X(f"{fx}:Q", scale=alt.Scale(zero=False)),
                    y=alt.Y(f"{fy}:Q", scale=alt.Scale(zero=False)),
                    color=alt.Color("label:N", scale=COLOR_SCALE, title="Class"),
                    tooltip=[fx, fy, "label"],
                ).properties(height=480).interactive(),
                use_container_width=True)

        with tab3:
            corr = self.df[self.features].corr().round(2)
            long = corr.reset_index().melt("index", var_name="other", value_name="corr")
            base = alt.Chart(long).encode(x=alt.X("other:N", title=None),
                                          y=alt.Y("index:N", title=None))
            heat = base.mark_rect().encode(
                color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])))
            text = base.mark_text(fontSize=11).encode(text="corr:Q")
            st.altair_chart((heat + text).properties(height=520), use_container_width=True)

        with tab4:
            st.caption("Same idea as your display_pairplot(), but on a feature subset so it renders fast.")
            chosen = st.multiselect("Features", self.features,
                                    default=["fLength", "fWidth", "fAlpha", "fDist"])
            if len(chosen) >= 2 and st.button("Draw pairplot"):
                sample = vdf.sample(n=min(500, len(vdf)), random_state=42)
                g = sns.pairplot(sample, vars=chosen, hue="label", palette="Set1", diag_kind="kde")
                st.pyplot(g.fig)
                plt.close("all")

        with tab5:
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows × columns", f"{self.df.shape[0]:,} × {self.df.shape[1]}")
            c2.metric("Missing values", int(self.df.isnull().sum().sum()))
            c3.metric("Duplicated rows", int(self.df.duplicated().sum()))
            st.dataframe(self.df.describe(include="all").T, use_container_width=True)
            st.dataframe(self.df.head(500), use_container_width=True)
            st.download_button("Download CSV", self.df.to_csv(index=False),
                               "magic04.csv", "text/csv")

    def page_performance(self, threshold):
        self.hero("Model Performance",
                  f"Evaluated on the held-out 20% test set ({len(self.y_true):,} events) "
                  f"at threshold {threshold:.2f}.")

        y, proba = self.y_true, self.proba_test
        pred = (proba >= threshold).astype(int)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{accuracy_score(y, pred):.3f}")
        c2.metric("Precision (hadron)", f"{precision_score(y, pred):.3f}")
        c3.metric("Recall (hadron)", f"{recall_score(y, pred):.3f}")
        c4.metric("F1", f"{f1_score(y, pred):.3f}")
        c5.metric("ROC-AUC", f"{roc_auc_score(y, proba):.3f}")

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion matrix")
            cm = confusion_matrix(y, pred)
            names = ["Gamma", "Hadron"]
            cm_df = pd.DataFrame(
                [(names[i], names[j], int(cm[i, j])) for i in range(2) for j in range(2)],
                columns=["Actual", "Predicted", "Count"])
            base = alt.Chart(cm_df).encode(
                x=alt.X("Predicted:N", axis=alt.Axis(labelAngle=0)), y="Actual:N")
            heat = base.mark_rect().encode(
                color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")))
            text = base.mark_text(fontSize=22, fontWeight="bold").encode(
                text="Count:Q",
                color=alt.condition(alt.datum.Count > cm.max() / 2,
                                    alt.value("white"), alt.value("black")))
            st.altair_chart((heat + text).properties(height=340), use_container_width=True)

        with right:
            st.subheader("ROC curve")
            fpr, tpr, _ = roc_curve(y, proba)
            line = alt.Chart(pd.DataFrame({"FPR": fpr, "TPR": tpr})).mark_line(
                color=HADRON_COLOR, strokeWidth=3).encode(
                x=alt.X("FPR:Q", title="False positive rate"),
                y=alt.Y("TPR:Q", title="True positive rate"))
            diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
                strokeDash=[5, 5], color="gray").encode(x="x", y="y")
            st.altair_chart((line + diag).properties(height=340), use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Predicted probability by true class")
            pdf = pd.DataFrame({"P(hadron)": proba,
                                "Class": np.where(y == 1, "Hadron", "Gamma")})
            hist = alt.Chart(pdf).mark_bar(opacity=0.65).encode(
                x=alt.X("P(hadron):Q", bin=alt.Bin(maxbins=30)),
                y=alt.Y("count():Q", stack=None, title="Events"),
                color=alt.Color("Class:N", scale=COLOR_SCALE))
            rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(
                color="white", strokeDash=[4, 4], size=2).encode(x="t:Q")
            st.altair_chart((hist + rule).properties(height=320), use_container_width=True)

        with right:
            st.subheader("What the model relies on")
            coefs = self.model.named_steps["model"].coef_[0]
            cdf = pd.DataFrame({"Feature": self.features, "Coefficient": coefs})
            cdf["Pushes toward"] = np.where(cdf["Coefficient"] > 0, "Hadron", "Gamma")
            st.altair_chart(
                alt.Chart(cdf).mark_bar().encode(
                    x="Coefficient:Q", y=alt.Y("Feature:N", sort="-x"),
                    color=alt.Color("Pushes toward:N", scale=COLOR_SCALE),
                    tooltip=["Feature", alt.Tooltip("Coefficient:Q", format=".3f")],
                ).properties(height=320),
                use_container_width=True)
            st.caption("Coefficients act on the Yeo-Johnson transformed, standardised "
                       "features, so magnitudes are directly comparable.")

    @staticmethod
    def _load_preset(row, features):
        for f in features:
            st.session_state[f"in_{f}"] = float(row[f])

    def page_predictor(self, threshold):
        self.hero("Live Predictor", "Enter the image parameters of an event and classify it.")
        df, feats = self.df, self.features

        for f in feats:
            st.session_state.setdefault(f"in_{f}", float(df[f].median()))

        b1, b2, b3, _ = st.columns([1, 1, 1, 3])
        b1.button("🎲 Random gamma", on_click=self._load_preset,
                  args=(df[df["class"] == "g"].sample().iloc[0], feats))
        b2.button("🎲 Random hadron", on_click=self._load_preset,
                  args=(df[df["class"] == "h"].sample().iloc[0], feats))
        b3.button("↺ Reset to median", on_click=self._load_preset,
                  args=(df[feats].median(), feats))

        cols = st.columns(2)
        values = {}
        for i, f in enumerate(feats):
            lo, hi = float(df[f].min()), float(df[f].max())
            with cols[i % 2]:
                values[f] = st.slider(f, lo, hi, key=f"in_{f}", step=(hi - lo) / 500,
                                      format="%.3f", help=FEATURE_INFO.get(f))

        x_new = pd.DataFrame([values])[feats]
        p_h = float(self.model.predict_proba(x_new)[0, 1])

        st.divider()
        r1, r2 = st.columns([1, 2])
        with r1:
            if p_h >= threshold:
                st.error("### Hadron (background)")
            else:
                st.success("### Gamma (signal)")
            st.metric("P(hadron)", f"{p_h:.1%}")
            st.metric("P(gamma)", f"{1 - p_h:.1%}")
        with r2:
            st.write("Probability of being a hadron")
            st.progress(p_h)
            st.caption(f"Decision threshold: {threshold:.2f}")
            coefs = self.model.named_steps["model"].coef_[0]
            z = self.model.named_steps["transformer"].transform(x_new)[0]
            contrib = pd.DataFrame({"Feature": feats, "Contribution": coefs * z})
            contrib["Pushes toward"] = np.where(contrib["Contribution"] > 0, "Hadron", "Gamma")
            st.altair_chart(
                alt.Chart(contrib).mark_bar().encode(
                    x=alt.X("Contribution:Q", title="Contribution to log-odds (hadron)"),
                    y=alt.Y("Feature:N", sort="-x"),
                    color=alt.Color("Pushes toward:N", scale=COLOR_SCALE),
                ).properties(height=300, title="Why this prediction?"),
                use_container_width=True)

    # ------------------------------------------------------------------- run
    def run(self):
        st.markdown(CSS, unsafe_allow_html=True)

        pages = {
            "🏠 Overview": self.page_overview,
            "📊 Data Explorer": self.page_explorer,
            "📈 Model Performance": self.page_performance,
            "🔮 Live Predictor": self.page_predictor,
        }
        st.sidebar.title("🔭 MAGIC Classifier")
        page = st.sidebar.radio("Navigate", list(pages))
        st.sidebar.divider()
        threshold = st.sidebar.slider(
            "Decision threshold (P[hadron] ≥ t → hadron)", 0.05, 0.95, 0.50, 0.01,
            help="Used on the Performance and Predictor pages.")
        st.sidebar.caption(
            "Logistic Regression + PowerTransformer on the "
            "[MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope) dataset.")

        pages[page](threshold)


@st.cache_resource(show_spinner="Loading data and model...")
def build_app() -> AstroApp:
    os.chdir(BASE)  # AstroClass reads "magic04.data" relatively
    return AstroApp()


if __name__ == "__main__":
    st.set_page_config(page_title="MAGIC Gamma vs Hadron", page_icon="🔭",
                       layout="wide", initial_sidebar_state="expanded")
    build_app().run()
