from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Cardiovascular Risk Explorer",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "cardiovascular_risk_dataset.csv"
TARGET_COL = "risk_category"
ID_COL = "Patient_ID"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df[TARGET_COL] = pd.Categorical(
        df[TARGET_COL], categories=["Low", "Medium", "High"], ordered=True
    )
    return df


@st.cache_data
def prepare_modeling_table(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    smoking_map = {"Never": 0, "Former": 1, "Current": 2}
    family_map = {"No": 0, "Yes": 1}
    risk_map = {"Low": 0, "Medium": 1, "High": 2}

    encoded["smoking_status_code"] = encoded["smoking_status"].map(smoking_map)
    encoded["family_history_code"] = encoded["family_history_heart_disease"].map(family_map)
    encoded["risk_code"] = encoded[TARGET_COL].map(risk_map)
    return encoded


@st.cache_data
def correlation_with_target(df: pd.DataFrame) -> pd.DataFrame:
    encoded = prepare_modeling_table(df)
    corr = (
        encoded.drop(columns=[ID_COL]).corr(numeric_only=True)["heart_disease_risk_score"]
        .drop("heart_disease_risk_score")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .reset_index()
    )
    corr.columns = ["variable", "correlation"]
    return corr


@st.cache_data
def aggregate_dashboard_data(df: pd.DataFrame):
    risk_distribution = (
        df[TARGET_COL].value_counts().rename_axis(TARGET_COL).reset_index(name="count")
    )
    smoking_risk = (
        df.groupby(["smoking_status", TARGET_COL], observed=False)
        .size()
        .reset_index(name="count")
    )
    family_risk = (
        df.groupby(["family_history_heart_disease", TARGET_COL], observed=False)
        .size()
        .reset_index(name="count")
    )
    return risk_distribution, smoking_risk, family_risk


@st.cache_data
def get_feature_lists(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in [ID_COL, TARGET_COL, "heart_disease_risk_score"]]
    categorical_cols = [c for c in feature_cols if df[c].dtype == "object"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    return feature_cols, numeric_cols, categorical_cols


@st.cache_resource
def train_models(df: pd.DataFrame):
    feature_cols, numeric_cols, categorical_cols = get_feature_lists(df)
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model_specs = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    trained_models = {}
    metrics_rows = []
    predictions = {}

    for model_name, estimator in model_specs.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        trained_models[model_name] = pipe
        predictions[model_name] = {
            "y_true": y_test,
            "y_pred": y_pred,
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"]),
            "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        }
        metrics_rows.append(
            {
                "Modèle": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1 macro": f1_score(y_test, y_pred, average="macro"),
                "F1 weighted": f1_score(y_test, y_pred, average="weighted"),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by=["F1 macro", "Accuracy"], ascending=False
    )
    best_model_name = metrics_df.iloc[0]["Modèle"]

    rf_pipeline = trained_models["Random Forest"]
    preprocessor_fitted = rf_pipeline.named_steps["preprocessor"]
    rf_model = rf_pipeline.named_steps["model"]
    feature_names = preprocessor_fitted.get_feature_names_out()
    feature_importance_df = (
        pd.DataFrame(
            {"feature": feature_names, "importance": rf_model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .head(15)
    )

    return {
        "metrics_df": metrics_df,
        "trained_models": trained_models,
        "predictions": predictions,
        "best_model_name": best_model_name,
        "feature_importance_df": feature_importance_df,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "feature_cols": feature_cols,
    }



def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtres globaux")

    age_range = st.sidebar.slider(
        "Âge",
        int(df["age"].min()),
        int(df["age"].max()),
        (int(df["age"].min()), int(df["age"].max())),
    )
    risk_options = st.sidebar.multiselect(
        "Catégories de risque",
        options=list(df[TARGET_COL].cat.categories),
        default=list(df[TARGET_COL].cat.categories),
    )
    smoking_options = st.sidebar.multiselect(
        "Statut tabagique",
        options=sorted(df["smoking_status"].unique()),
        default=sorted(df["smoking_status"].unique()),
    )
    family_options = st.sidebar.multiselect(
        "Antécédents familiaux",
        options=sorted(df["family_history_heart_disease"].unique()),
        default=sorted(df["family_history_heart_disease"].unique()),
    )

    filtered = df[
        (df["age"].between(age_range[0], age_range[1]))
        & (df[TARGET_COL].isin(risk_options))
        & (df["smoking_status"].isin(smoking_options))
        & (df["family_history_heart_disease"].isin(family_options))
    ].copy()

    st.sidebar.markdown("---")
    st.sidebar.metric("Lignes après filtrage", f"{len(filtered):,}")
    return filtered



def show_kpis(df: pd.DataFrame):
    missing_rate = df.isna().mean().mean() * 100
    cols = st.columns(6)
    cols[0].metric("Patients", f"{len(df):,}")
    cols[1].metric("Colonnes", df.shape[1])
    cols[2].metric("Âge moyen", f"{df['age'].mean():.1f} ans")
    cols[3].metric("BMI moyen", f"{df['bmi'].mean():.1f}")
    cols[4].metric("Risque moyen", f"{df['heart_disease_risk_score'].mean():.1f}")
    cols[5].metric("Valeurs manquantes", f"{missing_rate:.1f}%")



def page_home(df: pd.DataFrame, modeling_results: dict):
    st.title("❤️ Cardiovascular Risk Explorer")
    st.subheader("Projet personnel Streamlit - Parcours B")

    st.markdown(
        """
        Cette application explore un dataset médical synthétique portant sur le risque cardiovasculaire.
        L'objectif principal est de comprendre comment des variables de mode de vie et de santé
        comme l'âge, l'IMC, la tension artérielle, le cholestérol, l'activité physique, le tabagisme
        ou encore le sommeil sont associées à un score de risque cardiaque.

        En plus de l'analyse exploratoire, l'application intègre maintenant une **partie prédiction**
        pour estimer la **catégorie de risque cardiovasculaire** d'un patient à partir de ses variables
        de santé et de comportement.
        """
    )

    show_kpis(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Meilleur modèle", modeling_results["best_model_name"])
    c2.metric("Accuracy test", f"{modeling_results['metrics_df'].iloc[0]['Accuracy']:.3f}")
    c3.metric("F1 macro test", f"{modeling_results['metrics_df'].iloc[0]['F1 macro']:.3f}")

    st.markdown("### Aperçu des données")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### Description des colonnes principales")
    description = pd.DataFrame(
        {
            "Colonne": df.columns,
            "Description": [
                "Identifiant unique du patient",
                "Âge du patient",
                "Indice de masse corporelle",
                "Pression artérielle systolique",
                "Pression artérielle diastolique",
                "Taux de cholestérol",
                "Fréquence cardiaque au repos",
                "Statut tabagique",
                "Nombre moyen de pas par jour",
                "Niveau de stress",
                "Heures d'activité physique par semaine",
                "Heures de sommeil par nuit",
                "Présence d'antécédents familiaux",
                "Score de qualité alimentaire",
                "Unités d'alcool consommées par semaine",
                "Score numérique de risque cardiaque",
                "Catégorie de risque finale",
            ],
            "Type": [str(df[c].dtype) for c in df.columns],
        }
    )
    st.dataframe(description, use_container_width=True, hide_index=True)



def page_exploration(df: pd.DataFrame):
    st.title("📊 Exploration et visualisations")
    st.markdown(
        "Cette page présente les indicateurs descriptifs clés et plusieurs visualisations interactives du dataset filtré."
    )

    show_kpis(df)

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            df,
            x="heart_disease_risk_score",
            nbins=30,
            title="Distribution du score de risque cardiaque",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        risk_dist = df[TARGET_COL].value_counts().rename_axis(TARGET_COL).reset_index(name="count")
        fig = px.pie(
            risk_dist,
            names=TARGET_COL,
            values="count",
            title="Répartition des catégories de risque",
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.box(
            df,
            x=TARGET_COL,
            y="systolic_bp",
            color=TARGET_COL,
            title="Tension systolique selon la catégorie de risque",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        activity = (
            df.groupby("smoking_status", observed=False)["heart_disease_risk_score"]
            .mean()
            .reset_index()
            .sort_values("heart_disease_risk_score", ascending=False)
        )
        fig = px.bar(
            activity,
            x="smoking_status",
            y="heart_disease_risk_score",
            title="Score moyen de risque selon le statut tabagique",
            text_auto=".1f",
        )
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)

    with c5:
        fig = px.scatter(
            df,
            x="age",
            y="heart_disease_risk_score",
            color=TARGET_COL,
            size="cholesterol_mg_dl",
            hover_data=["bmi", "daily_steps", "smoking_status"],
            title="Âge et score de risque cardiaque",
            trendline="ols",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        corr_df = prepare_modeling_table(df).drop(columns=[ID_COL, "risk_code"])
        corr_matrix = corr_df.corr(numeric_only=True)
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            title="Heatmap des corrélations",
        )
        st.plotly_chart(fig, use_container_width=True)



def page_analysis(df: pd.DataFrame):
    st.title("🔎 Analyse approfondie")
    st.markdown(
        """
        **Question de recherche :**
        *Quels facteurs semblent les plus associés à une hausse du risque cardiovasculaire dans ce dataset ?*
        """
    )

    corr = correlation_with_target(df)
    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            corr.head(10),
            x="correlation",
            y="variable",
            orientation="h",
            title="Top 10 des variables les plus corrélées au score de risque",
            text_auto=".2f",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        agg = (
            df.groupby(TARGET_COL, observed=False)[
                ["age", "bmi", "systolic_bp", "cholesterol_mg_dl", "daily_steps"]
            ]
            .mean()
            .reset_index()
        )
        long = agg.melt(id_vars=TARGET_COL, var_name="metric", value_name="mean_value")
        fig = px.line(
            long,
            x="metric",
            y="mean_value",
            color=TARGET_COL,
            markers=True,
            title="Profils moyens par catégorie de risque",
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.box(
            df,
            x="family_history_heart_disease",
            y="heart_disease_risk_score",
            color="family_history_heart_disease",
            title="Impact des antécédents familiaux sur le score de risque",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        sleep_bins = pd.cut(df["sleep_hours"], bins=[0, 5, 6, 7, 8, 10], include_lowest=True)
        sleep_view = (
            df.assign(sleep_bin=sleep_bins)
            .groupby("sleep_bin", observed=False)["heart_disease_risk_score"]
            .mean()
            .reset_index()
        )
        sleep_view["sleep_bin"] = sleep_view["sleep_bin"].astype(str)
        fig = px.bar(
            sleep_view,
            x="sleep_bin",
            y="heart_disease_risk_score",
            title="Score moyen selon les heures de sommeil",
            text_auto=".1f",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Insights et conclusions")
    st.markdown(
        """
        1. Dans le dataset filtré, les variables les plus liées au score de risque sont la **tension systolique**,
        le **cholestérol**, la **tension diastolique**, l'**IMC** et l'**âge**.

        2. Les patients de catégorie **High** présentent en moyenne des niveaux plus élevés de tension artérielle,
        de cholestérol et d'IMC, tandis que leurs niveaux de pas quotidiens et d'activité physique sont plus faibles.

        3. Le **tabagisme actif** et la **présence d'antécédents familiaux** sont aussi associés à un risque plus élevé.

        4. Cette analyse reste **exploratoire**. Elle met en évidence des associations, pas une causalité clinique.
        """
    )



def page_prediction(df: pd.DataFrame, modeling_results: dict):
    st.title("🤖 Prédiction du risque cardiovasculaire")
    st.markdown(
        """
        Cette section entraîne plusieurs modèles de **classification** pour prédire la variable cible
        **risk_category** (`Low`, `Medium`, `High`).

        Les modèles comparés sont :
        - **Logistic Regression**
        - **Random Forest**
        - **Gradient Boosting**
        """
    )

    metrics_df = modeling_results["metrics_df"].copy()
    best_model_name = modeling_results["best_model_name"]
    predictions = modeling_results["predictions"]
    trained_models = modeling_results["trained_models"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Taille train", f"{modeling_results['X_train_shape'][0]:,}")
    c2.metric("Taille test", f"{modeling_results['X_test_shape'][0]:,}")
    c3.metric("Meilleur modèle", best_model_name)

    st.markdown("### Comparaison des performances")
    st.dataframe(
        metrics_df.style.format(
            {"Accuracy": "{:.3f}", "F1 macro": "{:.3f}", "F1 weighted": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig_perf = px.bar(
        metrics_df.melt(id_vars="Modèle", var_name="Metric", value_name="Score"),
        x="Modèle",
        y="Score",
        color="Metric",
        barmode="group",
        title="Comparaison Accuracy / F1 des modèles",
        text_auto=".3f",
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    selected_model = st.selectbox(
        "Voir le détail d'un modèle",
        options=metrics_df["Modèle"].tolist(),
        index=0,
    )

    cm = predictions[selected_model]["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Low", "Medium", "High"], columns=["Low", "Medium", "High"])
    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        title=f"Matrice de confusion - {selected_model}",
        labels=dict(x="Prédit", y="Réel", color="Effectif"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    report_df = pd.DataFrame(predictions[selected_model]["report"]).T.reset_index().rename(columns={"index": "classe"})
    st.markdown("### Classification report")
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    st.markdown("### Variables les plus importantes du Random Forest")
    fig_imp = px.bar(
        modeling_results["feature_importance_df"].sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 des importances de variables",
        text_auto=".3f",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("### Faire une prédiction sur un nouveau patient")
    st.caption(f"La prédiction ci-dessous utilise par défaut le meilleur modèle : {best_model_name}.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Âge", min_value=int(df["age"].min()), max_value=int(df["age"].max()), value=int(df["age"].median()))
            bmi = st.number_input("BMI", min_value=float(df["bmi"].min()), max_value=float(df["bmi"].max()), value=float(df["bmi"].median()))
            systolic_bp = st.number_input("Pression systolique", min_value=int(df["systolic_bp"].min()), max_value=int(df["systolic_bp"].max()), value=int(df["systolic_bp"].median()))
            diastolic_bp = st.number_input("Pression diastolique", min_value=int(df["diastolic_bp"].min()), max_value=int(df["diastolic_bp"].max()), value=int(df["diastolic_bp"].median()))
        with c2:
            cholesterol_mg_dl = st.number_input("Cholestérol (mg/dL)", min_value=int(df["cholesterol_mg_dl"].min()), max_value=int(df["cholesterol_mg_dl"].max()), value=int(df["cholesterol_mg_dl"].median()))
            resting_heart_rate = st.number_input("Fréquence cardiaque au repos", min_value=int(df["resting_heart_rate"].min()), max_value=int(df["resting_heart_rate"].max()), value=int(df["resting_heart_rate"].median()))
            daily_steps = st.number_input("Pas quotidiens", min_value=int(df["daily_steps"].min()), max_value=int(df["daily_steps"].max()), value=int(df["daily_steps"].median()))
            stress_level = st.number_input("Stress level", min_value=int(df["stress_level"].min()), max_value=int(df["stress_level"].max()), value=int(df["stress_level"].median()))
        with c3:
            physical_activity_hours_per_week = st.number_input("Activité physique / semaine", min_value=float(df["physical_activity_hours_per_week"].min()), max_value=float(df["physical_activity_hours_per_week"].max()), value=float(df["physical_activity_hours_per_week"].median()))
            sleep_hours = st.number_input("Heures de sommeil", min_value=float(df["sleep_hours"].min()), max_value=float(df["sleep_hours"].max()), value=float(df["sleep_hours"].median()))
            diet_quality_score = st.number_input("Diet quality score", min_value=int(df["diet_quality_score"].min()), max_value=int(df["diet_quality_score"].max()), value=int(df["diet_quality_score"].median()))
            alcohol_units_per_week = st.number_input("Alcool / semaine", min_value=float(df["alcohol_units_per_week"].min()), max_value=float(df["alcohol_units_per_week"].max()), value=float(df["alcohol_units_per_week"].median()))

        c4, c5 = st.columns(2)
        with c4:
            smoking_status = st.selectbox("Statut tabagique", sorted(df["smoking_status"].dropna().unique().tolist()))
        with c5:
            family_history = st.selectbox(
                "Antécédents familiaux",
                sorted(df["family_history_heart_disease"].dropna().unique().tolist()),
            )

        prediction_model_name = st.selectbox(
            "Modèle utilisé pour la prédiction",
            options=metrics_df["Modèle"].tolist(),
            index=metrics_df["Modèle"].tolist().index(best_model_name),
        )
        submitted = st.form_submit_button("Prédire la catégorie de risque")

    if submitted:
        new_patient = pd.DataFrame(
            [
                {
                    "age": age,
                    "bmi": bmi,
                    "systolic_bp": systolic_bp,
                    "diastolic_bp": diastolic_bp,
                    "cholesterol_mg_dl": cholesterol_mg_dl,
                    "resting_heart_rate": resting_heart_rate,
                    "smoking_status": smoking_status,
                    "daily_steps": daily_steps,
                    "stress_level": stress_level,
                    "physical_activity_hours_per_week": physical_activity_hours_per_week,
                    "sleep_hours": sleep_hours,
                    "family_history_heart_disease": family_history,
                    "diet_quality_score": diet_quality_score,
                    "alcohol_units_per_week": alcohol_units_per_week,
                }
            ]
        )

        model = trained_models[prediction_model_name]
        pred = model.predict(new_patient)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(new_patient)[0]
            class_names = model.classes_
            proba_df = pd.DataFrame({"Classe": class_names, "Probabilité": proba}).sort_values(
                "Probabilité", ascending=False
            )

        if pred == "High":
            st.error(f"Catégorie prédite : **{pred}**")
        elif pred == "Medium":
            st.warning(f"Catégorie prédite : **{pred}**")
        else:
            st.success(f"Catégorie prédite : **{pred}**")

        st.dataframe(new_patient, use_container_width=True, hide_index=True)

        if proba is not None:
            fig_proba = px.bar(
                proba_df,
                x="Classe",
                y="Probabilité",
                title="Probabilités par classe",
                text_auto=".3f",
            )
            st.plotly_chart(fig_proba, use_container_width=True)

        st.info(
            "Cette prédiction est issue d'un dataset synthétique et a une valeur pédagogique. Elle ne doit pas être interprétée comme un avis médical."
        )



def page_dashboard(df: pd.DataFrame):
    st.title("📈 Dashboard interactif")
    st.markdown("Tableau de bord synthétique avec téléchargement des données filtrées.")

    risk_distribution, smoking_risk, family_risk = aggregate_dashboard_data(df)

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.bar(
            risk_distribution,
            x=TARGET_COL,
            y="count",
            title="Effectifs par catégorie de risque",
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            smoking_risk,
            x="smoking_status",
            y="count",
            color=TARGET_COL,
            barmode="group",
            title="Risque par statut tabagique",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        fig = px.bar(
            family_risk,
            x="family_history_heart_disease",
            y="count",
            color=TARGET_COL,
            barmode="group",
            title="Risque et antécédents familiaux",
        )
        st.plotly_chart(fig, use_container_width=True)

    c4, c5 = st.columns(2)
    with c4:
        fig = px.scatter(
            df,
            x="daily_steps",
            y="heart_disease_risk_score",
            color="smoking_status",
            title="Pas quotidiens et score de risque",
            trendline="ols",
            hover_data=["age", "bmi", TARGET_COL],
        )
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        fig = px.scatter(
            df,
            x="physical_activity_hours_per_week",
            y="heart_disease_risk_score",
            color="family_history_heart_disease",
            title="Activité physique et score de risque",
            trendline="ols",
            hover_data=["age", "bmi", TARGET_COL],
        )
        st.plotly_chart(fig, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Télécharger les données filtrées en CSV",
        data=csv_bytes,
        file_name="cardio_filtered_data.csv",
        mime="text/csv",
    )

    st.dataframe(df, use_container_width=True, height=320)



def main():
    df = load_data()
    filtered_df = apply_filters(df)
    modeling_results = train_models(df)

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Accueil",
            "📊 Exploration et visualisations",
            "🔎 Analyse approfondie",
            "🤖 Prédiction",
            "📈 Dashboard interactif",
        ],
    )

    if filtered_df.empty:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés. Modifie les filtres dans la barre latérale.")
        return

    if page == "🏠 Accueil":
        page_home(filtered_df, modeling_results)
    elif page == "📊 Exploration et visualisations":
        page_exploration(filtered_df)
    elif page == "🔎 Analyse approfondie":
        page_analysis(filtered_df)
    elif page == "🤖 Prédiction":
        page_prediction(df, modeling_results)
    else:
        page_dashboard(filtered_df)


if __name__ == "__main__":
    main()
