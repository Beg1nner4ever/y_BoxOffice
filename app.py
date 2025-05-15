import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# For preprocessing (ensure these are in your requirements.txt)
from sentence_transformers import SentenceTransformer
# import tiktoken # Uncomment if token_count is a feature and tiktoken is used

# --- Application Configuration ---
APP_VERSION = "2.0 Advanced Model"
MODELS_BASE_DIR = "saved_models_from_notebook"
PREFERRED_MODEL_FILENAME = "xgboost_notebook_untuned.pkl"  # Your best performing model
SCALER_FILENAME = "data_scaler.pkl"
TRAINING_COLS_FILENAME = "training_columns_list.json"
SCALED_COLS_FILENAME = "scaled_columns_list.json"
MLB_FILES = {  # Map feature name to MLB filename
    "genres": "mlb_genres.pkl",
    "keywords": "mlb_keywords.pkl",
    # Add other MLBed features here if any, e.g. 'production_companies': "mlb_production_companies.pkl"
}
# TIKTOKEN_ENCODER_NAME = "cl100k_base" # Uncomment if using tiktoken

# --- Page Configuration ---
st.set_page_config(
    page_title="Y_BoxOffice Pro Revenue Predictor",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styling (Optional: Simple CSS for better aesthetics) ---
st.markdown(
    """
<style>
    .stMetricValue {
        font-size: 2.5rem !important;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
    }
    .stExpander {
        border: 1px solid #2C3A47;
        border-radius: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- Caching Functions for Loading Resources ---
@st.cache_resource  # For objects that are expensive to create (models, encoders)
def load_artifacts():
    """Loads the ML model, scaler, SBERT model, and MLBs."""
    artifacts = {
        "model": None,
        "scaler": None,
        "sbert_model": None,
        "mlbs": {},
        # "tiktoken_encoder": None # Uncomment if using
    }
    critical_load_error = False

    try:
        # SBERT Model
        artifacts["sbert_model"] = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"💥 Critical Error: Failed to load SentenceTransformer model: {e}")
        critical_load_error = True

    # ML Model
    model_path = os.path.join(MODELS_BASE_DIR, PREFERRED_MODEL_FILENAME)
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            artifacts["model"] = pickle.load(f)
    else:
        st.error(
            f"💥 Critical Error: Model '{PREFERRED_MODEL_FILENAME}' not found in '{MODELS_BASE_DIR}'."
        )
        critical_load_error = True

    # Scaler
    scaler_path = os.path.join(MODELS_BASE_DIR, SCALER_FILENAME)
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            artifacts["scaler"] = pickle.load(f)
    else:
        st.error(
            f"💥 Critical Error: Scaler '{SCALER_FILENAME}' not found. Predictions will be inaccurate."
        )
        critical_load_error = True  # Scaler is essential

    # MLBs
    for feature_name, mlb_filename in MLB_FILES.items():
        mlb_path = os.path.join(MODELS_BASE_DIR, mlb_filename)
        if os.path.exists(mlb_path):
            with open(mlb_path, "rb") as f:
                artifacts["mlbs"][feature_name] = pickle.load(f)
        else:
            st.warning(
                f"⚠️ MLB for '{feature_name}' ('{mlb_filename}') not found. '{feature_name.capitalize()}' features will be limited."
            )
            artifacts["mlbs"][feature_name] = None  # Mark as None if not found

    # Tiktoken Encoder (Optional)
    # try:
    #     artifacts["tiktoken_encoder"] = tiktoken.get_encoding(TIKTOKEN_ENCODER_NAME)
    # except Exception as e:
    #     st.warning(f"⚠️ Tiktoken encoder '{TIKTOKEN_ENCODER_NAME}' not loaded: {e}. 'token_count' feature might not be available.")

    if critical_load_error:
        return None  # Signal that critical components are missing
    return artifacts


@st.cache_data  # For data that doesn't change (column lists, feature importance)
def load_metadata():
    """Loads training columns, scaled columns, and feature importance."""
    metadata = {
        "training_cols": None,
        "scaled_cols": None,
        "feature_importance_df": None,
    }
    critical_load_error = False

    training_cols_path = os.path.join(MODELS_BASE_DIR, TRAINING_COLS_FILENAME)
    if os.path.exists(training_cols_path):
        with open(training_cols_path, "r") as f:
            metadata["training_cols"] = json.load(f)
    else:
        st.error(
            f"💥 Critical Error: Training columns list '{TRAINING_COLS_FILENAME}' not found."
        )
        critical_load_error = True

    scaled_cols_path = os.path.join(MODELS_BASE_DIR, SCALED_COLS_FILENAME)
    if os.path.exists(scaled_cols_path):
        with open(scaled_cols_path, "r") as f:
            metadata["scaled_cols"] = json.load(f)
    else:
        st.error(
            f"💥 Critical Error: Scaled columns list '{SCALED_COLS_FILENAME}' not found."
        )
        critical_load_error = True

    # Feature Importance
    importance_filename = (
        f"{PREFERRED_MODEL_FILENAME.split('.')[0]}_feature_importance.csv"
    )
    importance_path = os.path.join(MODELS_BASE_DIR, importance_filename)
    if os.path.exists(importance_path):
        metadata["feature_importance_df"] = pd.read_csv(importance_path)
    else:
        st.warning(f"⚠️ Feature importance file '{importance_filename}' not found.")

    if critical_load_error:
        return None
    return metadata


# --- Load Resources ---
artifacts = load_artifacts()
metadata = load_metadata()

if (
    not artifacts
    or not metadata
    or not artifacts["model"]
    or not artifacts["sbert_model"]
    or not metadata["training_cols"]
    or not metadata["scaled_cols"]
    or not artifacts["scaler"]
):
    st.error(
        "🔴 Critical application components failed to load. Please check the logs and ensure all necessary files are in the `saved_models_from_notebook` directory."
    )
    st.stop()


# --- Helper Functions ---
def format_currency(value):
    return f"${value:,.0f}"  # Simpler formatting for this app


def preprocess_input_data(
    raw_inputs_dict, sbert_model, mlbs, training_cols, scaler, scaled_cols
):  # , tiktoken_encoder):
    """
    Transforms raw user inputs into a feature vector ready for the model.
    """
    input_df = pd.DataFrame([raw_inputs_dict])

    # 1. Text for embedding
    input_df["text_to_embed"] = (
        (
            input_df["title"].fillna("")
            + " "
            + input_df["tagline"].fillna("")
            + " "
            + input_df["overview"].fillna("")
        )
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # 2. Token count (if it was a feature in training_cols)
    # if 'token_count' in training_cols and tiktoken_encoder:
    #     input_df['token_count'] = input_df['text_to_embed'].apply(lambda x: len(tiktoken_encoder.encode(x)))
    # elif 'token_count' in training_cols:
    #     input_df['token_count'] = 0 # Fallback if encoder not loaded but feature expected
    #     st.sidebar.warning("Token count defaulted to 0 as encoder missing.")

    # 3. Embeddings
    text_embeddings = sbert_model.encode(
        input_df["text_to_embed"].tolist(), convert_to_numpy=True
    )
    embedding_df = pd.DataFrame(text_embeddings, index=input_df.index)
    embedding_df.columns = [f"embed_{i}" for i in range(embedding_df.shape[1])]

    # 4. Simple Categorical Features (e.g., original_language, adult)
    # We will use get_dummies and then reindex with all training columns.
    # The training_cols list will already have the dummified names like 'original_language_en'.
    # So, we create dummies for what the user selected, then reindex.
    simple_categorical_to_dummy = {}
    if "original_language" in input_df.columns:  # Check if user provided it
        lang_prefix = "original_language_"
        user_lang_col = lang_prefix + input_df["original_language"].iloc[0]
        simple_categorical_to_dummy[user_lang_col] = 1
    if "adult" in input_df.columns:
        adult_prefix = "adult_"
        user_adult_col = adult_prefix + str(
            input_df["adult"].iloc[0]
        )  # 'adult_0' or 'adult_1'
        simple_categorical_to_dummy[user_adult_col] = 1

    simple_cats_df = pd.DataFrame([simple_categorical_to_dummy], index=input_df.index)

    # 5. Multi-Label Binarized Features (genres, keywords)
    list_features_dfs = []
    for feature_name, mlb_instance in mlbs.items():
        if mlb_instance:  # Check if MLB was loaded
            user_selection = raw_inputs_dict.get(
                f"selected_{feature_name}", []
            )  # e.g., raw_inputs_dict['selected_genres']
            if not isinstance(user_selection, list):  # e.g. keywords from text input
                user_selection = [
                    k.strip() for k in str(user_selection).split(",") if k.strip()
                ]

            transformed_data = mlb_instance.transform([user_selection])
            feature_df = pd.DataFrame(
                transformed_data,
                columns=[f"{feature_name}_{c}" for c in mlb_instance.classes_],
                index=input_df.index,
            )
            list_features_dfs.append(feature_df)
        # else: Create empty columns for this feature set if MLB missing but features in training_cols? Handled by final reindex.

    # 6. Combine all base and derived features
    # Select numerical features that were directly input and are in training_cols
    direct_numerical_features = [
        col
        for col in [
            "runtime",
            "budget",
            "release_year",
            "release_month",
        ]  # Add 'token_count' if used
        if col in training_cols and col in input_df.columns
    ]

    combined_df = pd.concat(
        [input_df[direct_numerical_features], simple_cats_df, embedding_df]
        + list_features_dfs,
        axis=1,
    )

    # 7. Reindex to match training columns (CRITICAL)
    # This adds any missing columns (e.g., unselected dummy variables) with fill_value=0
    # and ensures correct order.
    final_df = combined_df.reindex(columns=training_cols, fill_value=0)

    # 8. Scale features
    # Ensure only existing columns in final_df that need scaling are passed to scaler
    cols_to_scale_for_input = [col for col in scaled_cols if col in final_df.columns]
    if cols_to_scale_for_input:
        final_df[cols_to_scale_for_input] = scaler.transform(
            final_df[cols_to_scale_for_input]
        )

    return final_df


# --- UI Layout ---
st.title("🎬 Y_BoxOffice Pro Revenue Predictor")
st.markdown(
    f"Predict movie revenue with an advanced machine learning model. (App Version: {APP_VERSION})"
)
st.markdown("---")

# Sidebar for Inputs
with st.sidebar:
    st.header(" ✨ Movie Input Details")
    with st.form(key="movie_input_form"):
        st.subheader("📝 Textual Information")
        title = st.text_input(
            "Movie Title",
            "Avatar: The Way of Water",
            help="The official title of the movie.",
        )
        tagline = st.text_input(
            "Tagline", "Return to Pandora.", help="The movie's catchy tagline, if any."
        )
        overview = st.text_area(
            "Overview / Synopsis",
            "Jake Sully lives with his newfound family formed on the extrasolar moon Pandora. Once a familiar threat returns to finish what was previously started, Jake must work with Neytiri and the army of the Na'vi race to protect their home.",
            height=120,
            help="A brief summary of the movie's plot.",
        )

        st.subheader("🔢 Core Numbers")
        budget = st.number_input(
            "Budget (USD)",
            min_value=0,
            max_value=1_000_000_000,
            value=350_000_000,
            step=1_000_000,
            format="%d",
            help="Estimated production budget in US dollars.",
        )
        runtime = st.slider(
            "Runtime (minutes)",
            min_value=30,
            max_value=300,
            value=192,
            help="Total runtime of the movie in minutes.",
        )
        # token_count (Calculated automatically if it was a feature)

        st.subheader("🗂️ Categories & Creative")
        # Dynamically get genre options from loaded MLB if available and has reasonable number of classes
        genre_mlb = artifacts["mlbs"].get("genres")
        genre_options_default = [
            "Action",
            "Adventure",
            "Animation",
            "Comedy",
            "Crime",
            "Drama",
            "Family",
            "Fantasy",
            "Horror",
            "Mystery",
            "Romance",
            "Science Fiction",
            "Thriller",
        ]
        genre_options = (
            sorted(list(genre_mlb.classes_))
            if genre_mlb and len(genre_mlb.classes_) < 50
            else genre_options_default
        )
        selected_genres = st.multiselect(
            "Genres",
            genre_options,
            default=["Action", "Adventure", "Science Fiction"],
            help="Select one or more genres.",
        )

        keywords_input = st.text_input(
            "Keywords (comma-separated)",
            "pandora, alien planet, sequel, 3d",
            help="Up to 5-7 important keywords, e.g., 'superhero, time travel, based on comic'.",
        )

        # Prepare original language options (top N from your EDA + Other)
        # Example: These should ideally come from your analysis or training_cols prefixes
        lang_options_map = {
            "English": "en",
            "French": "fr",
            "Spanish": "es",
            "German": "de",
            "Japanese": "ja",
            "Korean": "ko",
            "Chinese": "zh",
            "Hindi": "hi",
            "Italian": "it",
            "Russian": "ru",
            "Other": "other_lang",
        }
        selected_lang_display = st.selectbox(
            "Original Language",
            list(lang_options_map.keys()),
            index=0,
            help="The primary language of the movie.",
        )
        original_language = lang_options_map[selected_lang_display]

        adult = st.checkbox(
            "Adult Movie (Rated R/18+ or equivalent)",
            False,
            help="Is this an adult-rated movie?",
        )

        st.subheader("🗓️ Release Timing")
        current_year = datetime.now().year
        release_year = st.slider(
            "Release Year",
            current_year - 30,
            current_year + 5,
            current_year - 1,
            help="The year the movie was (or will be) released.",
        )
        release_month = st.slider(
            "Release Month", 1, 12, 12, help="The month of release."
        )

        submit_button = st.form_submit_button(label="🚀 Predict Revenue!")

# --- Main Area for Output ---
if submit_button:
    raw_inputs = {
        "title": title,
        "tagline": tagline,
        "overview": overview,
        "budget": budget,
        "runtime": runtime,
        "selected_genres": selected_genres,  # Pass as list
        "selected_keywords": keywords_input,  # Pass as string, will be parsed in preprocess
        "original_language": original_language,  # Pass the code 'en', 'fr' etc.
        "adult": 1 if adult else 0,
        "release_year": release_year,
        "release_month": release_month,
    }

    with st.spinner(
        "⚙️ Preprocessing data and running model... This might take a moment."
    ):
        try:
            model_input_df = preprocess_input_data(
                raw_inputs,
                artifacts["sbert_model"],
                artifacts["mlbs"],
                metadata["training_cols"],
                artifacts["scaler"],
                metadata["scaled_cols"],
                # artifacts["tiktoken_encoder"] # Uncomment if used
            )

            # Prediction
            prediction_log = artifacts["model"].predict(model_input_df)[0]
            predicted_revenue = np.expm1(
                prediction_log
            )  # Assuming log1p was used for target
            predicted_revenue = max(0, predicted_revenue)  # Ensure non-negative

            st.success("✅ Prediction Complete!")
            st.markdown("---")

            col1, col2 = st.columns([0.6, 0.4])
            with col1:
                st.subheader(f"💰 Predicted Revenue for '{title}'")
                st.metric(
                    label="Estimated Gross Revenue",
                    value=format_currency(predicted_revenue),
                )
                st.caption(
                    f"Prediction based on model: {PREFERRED_MODEL_FILENAME.split('.')[0]}"
                )

            with col2:
                st.image(
                    "https://www.transparentpng.com/thumb/money/money-stack-clipart-transparent-yAgHkD.png",
                    width=150,
                )  # Placeholder image
                st.markdown(
                    f"<div style='text-align: center; font-size: small;'>Model Confidence: Moderate <br>(Varies by input similarity to training data)</div>",
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"🤕 An error occurred during prediction: {e}")
            st.exception(e)  # Shows traceback for debugging

    st.markdown("---")
    # Feature Importance Display
    if metadata["feature_importance_df"] is not None:
        with st.expander("🔍 View Top Feature Importances", expanded=False):
            st.markdown(
                "Features that most influenced this type of prediction (based on the trained model):"
            )
            top_n_features = 15
            fi_df_display = metadata["feature_importance_df"].head(top_n_features)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x="Importance",
                y="Feature",
                data=fi_df_display,
                palette="viridis",
                ax=ax,
            )
            ax.set_title(f"Top {top_n_features} Feature Importances")
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("Feature importance data is not available for display.")

else:
    st.info(
        "✨ Fill in the movie details in the sidebar and click 'Predict Revenue!' to see the magic happen."
    )


# --- Additional Information Sections ---
st.markdown("---")
col_about1, col_about2 = st.columns(2)

with col_about1:
    with st.expander("ℹ️ About This Predictor"):
        st.markdown(f"""
        This application utilizes a **{PREFERRED_MODEL_FILENAME.split("_")[0].upper()}** model, trained on a dataset
        from The Movie Database (TMDB). It considers a variety of features, including:
        - Textual data (title, tagline, overview) processed into numerical embeddings.
        - Core numerical inputs (budget, runtime).
        - Categorical aspects (original language, adult rating).
        - Creative elements (genres, keywords).
        - Release timing (year, month).

        The revenue predictions are log-transformed during training and then converted back for display.
        This tool is intended for illustrative and educational purposes.
        """)

with col_about2:
    with st.expander("⚠️ Limitations & Considerations"):
        st.markdown("""
        - **Data Scope:** Predictions are based on the patterns learned from the TMDB dataset. Accuracy may vary for movies significantly different from the training data.
        - **Market Dynamics:** External factors not in the dataset (e.g., marketing spend, competition, global events, star power not captured by text) heavily influence revenue and are not modeled here.
        - **Feature Representation:** The way features like 'keywords' are input can impact results.
        - **Model Generalization:** While efforts are made to build a generalizable model, it's a simplified representation of a complex reality.
        - **Not Financial Advice:** Predictions should not be taken as definitive financial forecasts.
        """)

st.markdown("<hr style='border:1px solid #2C3A47'>", unsafe_allow_html=True)
st.caption(
    f"Y_BoxOffice Pro Predictor | Version {APP_VERSION} | Albert School GDA Project | Powered by Streamlit & Scikit-learn/XGBoost"
)
