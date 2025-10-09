import streamlit as st
import pandas as pd
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import joblib
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


try:
    from sklearn.compose._column_transformer import _RemainderColsList
except ImportError:
    class _RemainderColsList(list):
        pass


module_name = 'sklearn.compose._column_transformer'
if module_name not in sys.modules:
    try:
        import importlib
        importlib.import_module('sklearn.compose')
    except Exception:
        pass

try:
    sys.modules[module_name]._RemainderColsList = _RemainderColsList
except Exception:
    import types
    fake_mod = types.ModuleType(module_name)
    fake_mod._RemainderColsList = _RemainderColsList
    sys.modules[module_name] = fake_mod


def cast_to_str(A):
    try:
        return A.astype(str)
    except Exception:
        if isinstance(A, pd.DataFrame):
            return A.apply(lambda col: col.astype(str))
        return np.asarray(A).astype(str)


# ---------------------------
# Styling used by both pages
# ---------------------------
CARD_STYLE = """
<style>
.card { display:flex; gap:12px; margin-bottom:12px; }
.kpi { padding:12px; border-radius:8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);}
.kpi .value { font-size:20px; font-weight:700; }
.kpi .small-muted { color:#666; font-size:12px; }
</style>
"""


@st.cache_resource
def load_model():
    """Load the USA model used in the app (cached)."""
    with open("xgboost_filtered_data_model.pkl", "rb") as f:
        xgb_model = joblib.load(f)
    return xgb_model


@st.cache_data
def load_feature_names():
    with open("feature_names_in_order_filtered_data.json", "r") as f:
        feature_names = json.load(f)
        print(f"Loaded {len(feature_names)} feature names.")
    if not isinstance(feature_names, (list, tuple)):
        raise ValueError("feature_names_in_order must be a JSON array")
    return feature_names


@st.cache_resource
def load_japan_model(model_path="best_model_japan.pkl"):
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load Japan model from {model_path}: {e}")
        return None


@st.cache_data
def load_japan_features(features_path="best_model_japan.json"):
    try:
        with open(features_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load Japan feature list from {features_path}: {e}")
        return None


# ---------------------------
# Helper utilities
# ---------------------------

def safe_read_csv(uploaded):
    try:
        return pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return None


def run_predict_with_fallback(model, X):
    try:
        return model.predict(X)
    except Exception:
        # Attempt to bypass pipeline preprocessor and call final estimator
        if hasattr(model, "named_steps"):
            try:
                steps = list(model.named_steps.items())
                final_estimator = steps[-1][1]
                X_num = X.select_dtypes(include=["number"]).copy()
                return final_estimator.predict(X_num)
            except Exception as e:
                raise e
        raise


def bring_in_reportlab():
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        return SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, getSampleStyleSheet, colors
    except ImportError:
        st.error("Need 'reportlab' to make PDF reports. Add it to requirements.txt and redeploy.")
        st.stop()


# ---------------------------
# app layout
# ---------------------------
st.set_page_config(page_title="CAD Prediction App", page_icon="🫀", layout="wide")
st.sidebar.title("🫀 CAD Prediction App")

page = st.sidebar.radio(
    "Navigate to:",
    ["Home", "Japan Gut Prediction", "USA Custom Page", "Datasets", "About"]
)

st.markdown(CARD_STYLE, unsafe_allow_html=True)

# ---------------------------
# Home Page
# ---------------------------
if page == "Home":
    st.markdown("<h1 style='text-align:center;'>🫀 CAD Prediction using Gut Microbiome</h1>", unsafe_allow_html=True)
    st.write("Welcome! Use the sidebar to navigate to Japan or USA dataset pages.")

# ---------------------------
# Japan Gut Prediction Page (now unified look)
# ---------------------------
elif page == "Japan Gut Prediction":
    st.markdown("## 🇯🇵 Japan Gut Dataset Prediction")

    uploaded = st.file_uploader("Upload Japan dataset (CSV) with column `Status`", type="csv", key="japan_file")

    run_btn = st.button("🚀 Run Prediction", key="run_japan")
    pdf_btn = st.button("📄 Generate PDF Report", key="pdf_japan")

    if uploaded:
        df = safe_read_csv(uploaded)
        if df is None:
            st.stop()

        if "Status" not in df.columns:
            st.error("CSV must contain 'Status' column with 'cad'/'control' values.")
            st.stop()

        # KPI cards
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown("<div class='kpi'><div class='value'>{}</div><div class='small-muted'>Rows</div></div>".format(df.shape[0]), unsafe_allow_html=True)
        with k2:
            st.markdown("<div class='kpi'><div class='value'>{}</div><div class='small-muted'>Features</div></div>".format(df.shape[1]), unsafe_allow_html=True)
        with k3:
            class_balance = df["Status"].astype(str).str.lower().str.strip().value_counts().to_dict()
            balance_str = " / ".join([f"{k}:{v}" for k, v in class_balance.items()])
            st.markdown("<div class='kpi'><div class='value'>{}</div><div class='small-muted'>Class balance</div></div>".format(balance_str), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Tabs
        preview_tab, chart_tab, raw_tab = st.tabs(["Preview", "Charts", "Raw"])
        with preview_tab:
            st.subheader("Dataset preview")
            st.dataframe(df.head(100), use_container_width=True)
        with chart_tab:
            st.subheader("Feature & label overview")
            st.write("Run the prediction to view results and charts.")
        with raw_tab:
            st.subheader("Raw CSV head")
            st.code(df.head(10).to_csv(index=False))

        # Load model & feature names
        with st.spinner("Loading Japan model and features..."):
            japan_model = load_japan_model()
            japan_features = None
            try:
                japan_features = load_japan_features()
                if japan_features:
                    st.success(f"Loaded {len(japan_features)} Japan model features.")
            except Exception as e:
                st.warning(f"Could not load Japan features: {e}")

        # Run predictions
        if run_btn:
            if japan_model is None:
                st.error("Japan model not loaded. Add best_model_japan.pkl to the app folder.")
                st.stop()

            # Prepare X and y
            X = df.drop(columns=["Status"], errors="ignore")
            y_true = df["Status"].astype(str).str.lower().str.strip()

            # Factorize object columns
            for col in X.select_dtypes(include="object").columns:
                X[col] = pd.factorize(X[col])[0]

            # Fill missing features if model expects a list
            if isinstance(japan_features, (list, tuple)):
                missing_cols = [c for c in japan_features if c not in X.columns]
                if missing_cols:
                    st.warning(f"Missing columns filled with zeros: {missing_cols[:10]}")
                    for c in missing_cols:
                        X[c] = 0
                # Reorder safely
                X = X.reindex(columns=japan_features, fill_value=0)

            # Predict (with fallback)
            try:
                preds = run_predict_with_fallback(japan_model, X)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            mapping = {0: "cad", 1: "control"}
            pred_labels = [mapping.get(p, str(p)) for p in preds]

            df_res = df.copy()
            df_res["Predicted_Status"] = pred_labels
            df_res["Actual_Status"] = y_true

            st.success("✅ Predictions completed!")
            st.dataframe(df_res.head())
            
            res_tab, cm_tab = st.tabs(["Prediction Breakdown", "Confusion Matrix"])
            with res_tab:
                st.subheader("Predictions preview")
                st.dataframe(df_res[["Predicted_Status", "Actual_Status"]].head(200), use_container_width=True)
                    
                # Plots and confusion matrix
                pred_counts = df_res["Predicted_Status"].value_counts()
                fig1, ax1 = plt.subplots(figsize=(8, 4.5))
                sns.barplot(x=pred_counts.index, y=pred_counts.values, ax=ax1)
                ax1.set_title("Predicted Labels Distribution")
                ax1.set_xlabel("Predicted Status")
                ax1.set_ylabel("Count")
                st.pyplot(fig1)

            with cm_tab:
                # Confusion
                cm = confusion_matrix(y_true, df_res["Predicted_Status"], labels=["cad", "control"]) if set(["cad","control"]).issubset(set(y_true.unique())) else confusion_matrix(y_true, df_res["Predicted_Status"])
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
                ax2.set_xlabel("Predicted")
                ax2.set_ylabel("Actual")
                st.pyplot(fig2)

            acc = accuracy_score(y_true, df_res["Predicted_Status"]) if len(set(y_true))>0 else None
            if acc is not None:
                st.markdown(f"### 🎯 Model Accuracy: **{acc:.2f}**")

            # Save session state for PDF
            st.session_state["japan_data_out"] = df_res
            st.session_state["japan_fig1"] = fig1
            st.session_state["japan_fig2"] = fig2
            st.session_state["japan_acc"] = acc

    else:
        st.info("Upload a Japan CSV to get started. You can use the sample dataset from the Datasets page.")

    # Japan PDF generation
    if pdf_btn:
        if "japan_data_out" not in st.session_state:
            st.error("Please run predictions first.")
            st.stop()

        (Doc, Para, Spacer, RLImg, Table, TStyle, getStyles, colors) = bring_in_reportlab()

        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = Doc(temp_pdf.name)
        styles = getStyles()
        story = [Para("Japan Gut CAD Prediction Report", styles["Title"]), Spacer(1, 16)]

        if "japan_acc" in st.session_state and st.session_state["japan_acc"] is not None:
            story.append(Para(f"Model Accuracy: {st.session_state['japan_acc']:.2f}", styles["Normal"]))
            story.append(Spacer(1, 10))

        report = classification_report(
            st.session_state["japan_data_out"]["Actual_Status"],
            st.session_state["japan_data_out"]["Predicted_Status"],
            output_dict=True
        )
        rep_df = pd.DataFrame(report).transpose().round(2)

        rows = [["Class", "Precision", "Recall", "F1", "Support"]]
        for label in rep_df.index:
            if label == "accuracy":
                continue
            row = [
                label,
                rep_df.loc[label, "precision"],
                rep_df.loc[label, "recall"],
                rep_df.loc[label, "f1-score"],
                int(rep_df.loc[label, "support"])
            ]
            rows.append(row)

        table = Table(rows, hAlign="CENTER")
        table.setStyle(TStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ]))

        story.append(Para("Classification Report", styles["Heading2"]))
        story.append(table)
        story.append(Spacer(1, 16))

        for fig_obj, fig_title in [("japan_fig1", "Prediction Distribution"), ("japan_fig2", "Confusion Matrix")]:
            fig = st.session_state.get(fig_obj)
            if fig:
                tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(tmp_img.name, bbox_inches="tight", facecolor="white")
                story.append(Para(fig_title, styles["Heading2"]))
                story.append(RLImg(tmp_img.name, width=400, height=250))
                story.append(Spacer(1, 16))
                tmp_img.close()
            
        # Add prediction data table
        df_out = st.session_state["japan_data_out"][["Sample ID", "clade_name", "Status", "Predicted_Status"]]
        story.append(Para("Prediction Report", styles["Heading2"]))
        data_rows = [list(df_out.columns)] + df_out.values.tolist()

        pred_table = Table(data_rows, hAlign="CENTER")
        pred_table.setStyle(TStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))

        story.append(pred_table)
        story.append(Spacer(1, 16))
        
        doc.build(story)

        with open(temp_pdf.name, "rb") as f:
            st.download_button("📥 Download PDF", data=f, file_name="Japan_CAD_Report.pdf", mime="application/pdf")


# ---------------------------
# USA Custom Page
# ---------------------------
elif page == "USA Custom Page":
    st.markdown("Upload a CSV that includes a `CARDIOVASCULAR_DISEASE` column with 'Yes'/'No' values.")
    st.markdown("</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload your gut microbiome CSV", type="csv", help="CSV must include CARDIOVASCULAR_DISEASE (Yes/No).", key="usa_file")
    run_btn = st.button("🚀 Run Prediction", key="run_usa")
    pdf_btn = st.button("📄 Generate PDF Report", key="pdf_usa")

    df = None
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as err:
            st.error(f"Could not read file: {err}")
            st.stop()

        if "CARDIOVASCULAR_DISEASE" not in df.columns:
            st.error("CSV must contain 'CARDIOVASCULAR_DISEASE' column with 'Yes'/'No' values.")
            st.stop()

        if "VIOSCREEN_BMI" in df.columns:
            df = df.drop(columns=["VIOSCREEN_BMI"])

        # cards
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown("<div class='kpi'><div class='value'>{}</div><div class='small-muted'>Rows</div></div>".format(df.shape[0]), unsafe_allow_html=True)
        with k2:
            st.markdown("<div class='kpi'><div class='value'>{}</div><div class='small-muted'>Features</div></div>".format(df.shape[1]), unsafe_allow_html=True)
        with k3:
            class_balance = df["CARDIOVASCULAR_DISEASE"].astype(str).str.lower().str.strip().value_counts().to_dict()
            
            balance_str = " / ".join([f"{k}:{v}" for k, v in class_balance.items()])
            st.markdown("<div class='kpi'><div class='value'>{}</div><div class='small-muted'>Class balance</div></div>".format(balance_str), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Tabs
        preview_tab, chart_tab, raw_tab = st.tabs(["Preview", "Charts", "Raw"])
        with preview_tab:
            st.subheader("Dataset preview")
            st.dataframe(df.head(100), use_container_width=True)
        with chart_tab:
            st.subheader("Feature & label overview")
            st.write("If you want richer charts, run the prediction and inspect results in the Results area.")
        with raw_tab:
            st.subheader("Raw CSV head")
            st.code(df.head(10).to_csv(index=False))

        # Load model & features
        with st.spinner("Loading model and features..."):
            model = load_model()
            feature_names = None
            try:
                feature_names = load_feature_names()
                st.success(f"Loaded {len(feature_names)} model features.")
            except Exception as e:
                st.warning(f"Could not load feature names: {e}")

        if run_btn:
            preds = None
            try:
                preds = model.predict(df)
                st.success("Prediction successful (fully).")
            except Exception as e:
                st.warning(f"Pipeline preprocessing failed: {e}")
                st.info("Retrying by bypassing preprocessor...")

                if hasattr(model, "named_steps"):
                    try:
                        steps = list(model.named_steps.items())
                        final_step_name, final_estimator = steps[-1]
                        st.write(f"Detected final estimator: `{final_step_name}`")
                        df_numeric = df.select_dtypes(include=["number"]).copy()
                        preds = final_estimator.predict(df_numeric)
                        st.success("Prediction successful (bypassed pipeline).")
                    except Exception as e2:
                        st.error(f"Final estimator also failed: {e2}")
                else:
                    st.error("Model is not a Pipeline; cannot bypass preprocessing.")

            if preds is not None:
                df_result = df.copy()
                df_result["prediction"] = preds

                label_map = {0: "cad", 1: "control"}
                pred_labels = [label_map.get(p, str(p).lower()) for p in preds]

                df["Predicted_Status"] = pred_labels
                y_real = df["CARDIOVASCULAR_DISEASE"].astype(str).str.lower().str.strip()
                df["Actual_Status"] = y_real

                # Save to session for PDF/export
                st.session_state.update({
                    "data_out": df,
                    "acc": None
                })

                # Results & confusion
                res_tab, cm_tab = st.tabs(["Prediction Breakdown", "Confusion Matrix"])
                with res_tab:
                    st.subheader("Predictions preview")
                    st.dataframe(df[["Predicted_Status", "Actual_Status"]].head(200), use_container_width=True)

                    pred_dist = pd.Series(pred_labels).value_counts()
                    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
                    sns.barplot(x=pred_dist.index, y=pred_dist.values, ax=ax1)
                    ax1.set_title("Predicted Labels Distribution")
                    ax1.set_xlabel("Predicted Status")
                    ax1.set_ylabel("Count")
                    st.pyplot(fig1)

                with cm_tab:
                    def normalize_label(x):
                        x = str(x).strip().lower()
                        if x in ["yes", "1", "cad"]:
                            return "yes"
                        elif x in ["no", "0", "control"]:
                            return "no"
                        return x

                    y_real_norm = [normalize_label(v) for v in y_real]
                    pred_labels_norm = [normalize_label(v) for v in pred_labels]
                    unique_labels = sorted(set(y_real_norm) | set(pred_labels_norm))
                    cm = confusion_matrix(y_real_norm, pred_labels_norm, labels=unique_labels)

                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=unique_labels, yticklabels=unique_labels, ax=ax2)
                    ax2.set_xlabel("Predicted")
                    ax2.set_ylabel("Actual")
                    st.pyplot(fig2)

                acc = accuracy_score(y_real_norm, pred_labels_norm)
                st.markdown(f"### 🎯 Accuracy: **{acc:.2f}**")
                
                st.session_state.update({
                    "fig1": fig1,
                    "fig2": fig2,
                    "acc": acc
                })

    else:
        st.info("Upload a CSV to get started. You can use the sample dataset from the Datasets page.")

    # USA PDF generation
    if pdf_btn:
        if "data_out" not in st.session_state:
            st.error("Please run predictions first.")
            st.stop()

        (Doc, Para, Spacer, RLImg, Table, TStyle, getStyles, colors) = bring_in_reportlab()

        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = Doc(temp_pdf.name)
        styles = getStyles()
        story = [Para("CAD Prediction Report (USA Gut)", styles["Title"]), Spacer(1, 16)]

        if "acc" in st.session_state and st.session_state["acc"] is not None:
            story.append(Para(f"Model Accuracy: {st.session_state['acc']:.2f}", styles["Normal"]))
            story.append(Spacer(1, 10))

        report = classification_report(
            st.session_state["data_out"]["Actual_Status"],
            st.session_state["data_out"]["Predicted_Status"],
            output_dict=True
        )
        rep_df = pd.DataFrame(report).transpose().round(2)

        rows = [["Class", "Precision", "Recall", "F1", "Support"]]
        for label in rep_df.index:
            if label == "accuracy":
                continue
            row = [
                label,
                rep_df.loc[label, "precision"],
                rep_df.loc[label, "recall"],
                rep_df.loc[label, "f1-score"],
                int(rep_df.loc[label, "support"])
            ]
            rows.append(row)

        table = Table(rows, hAlign="CENTER")
        table.setStyle(TStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ]))

        story.append(Para("Classification Report", styles["Heading2"]))
        story.append(table)
        story.append(Spacer(1, 16))

        for fig_obj, fig_title in [("fig1", "Prediction Breakdown"), ("fig2", "Confusion Matrix")]:
            fig = st.session_state.get(fig_obj)
            if fig:
                tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(tmp_img.name, bbox_inches="tight", facecolor="white")
                story.append(Para(fig_title, styles["Heading2"]))
                story.append(RLImg(tmp_img.name, width=400, height=250))
                story.append(Spacer(1, 16))
                tmp_img.close()

        # Add prediction data table
        df_out = st.session_state["data_out"][["#SampleID", "COUNTRY_OF_BIRTH", "SEX", "LIVER_DISEASE", "AGE_YEARS", "DIABETES", "Actual_Status", "Predicted_Status"]]
        story.append(Para("Prediction Report", styles["Heading2"]))
        data_rows = [list(df_out.columns)] + df_out.values.tolist()

        pred_table = Table(data_rows[:60], hAlign="CENTER")
        pred_table.setStyle(TStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))

        story.append(pred_table)
        story.append(Spacer(1, 16))
        
        doc.build(story)

        with open(temp_pdf.name, "rb") as f:
            st.download_button("📥 Download PDF", data=f, file_name="CAD_Report.pdf", mime="application/pdf")


# ---------------------------
# Datasets Page
# ---------------------------
elif page == "Datasets":
    st.markdown("## 📂 Sample Datasets")
    sample_files = {
        "🍣 Japan Gut Microbiome": "PRJDB6472.csv",
        "🧬 USA Gut Microbiome": "dataset_filtered.csv",
    }
    for name, file in sample_files.items():
        if os.path.exists(file):
            with open(file, "rb") as f:
                st.download_button(label=f"⬇️ Download {name}", data=f, file_name=file, mime="text/csv")
        else:
            st.warning(f"{file} not found. Please add it to the app folder.")

# ---------------------------
# About Page
# ---------------------------
elif page == "About":
    st.markdown("## About this Project")
    st.write("""
    - **Japan Dataset** → Uses the existing Japan prediction code  
    - **USA Dataset** → Use the 'USA Custom Page' to paste your own processing code
    """)

