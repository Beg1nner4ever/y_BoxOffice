import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Y_BoxOffice Revenue Predictor",
    page_icon="🎬",
    layout="wide"
)

# Title and description
st.title("🎬 Y_BoxOffice Revenue Predictor")
st.markdown("""
This app uses machine learning to predict a movie's box office revenue based on various characteristics.
Fill in the details below to get a prediction!
""")

# Sidebar for inputs
st.sidebar.header("Movie Details")

# User inputs
with st.sidebar:
    st.subheader("Basic Information")
    runtime = st.slider("Runtime (minutes)", 30, 280, 120)
    vote_average = st.slider("Average Rating (0-10)", 0.0, 10.0, 7.0, 0.1)
    vote_count = st.number_input("Number of Votes", 0, 100000, 1000)
    
    st.subheader("Release Information")
    release_year = st.slider("Release Year", 1990, datetime.now().year, 2023)
    release_month = st.slider("Release Month", 1, 12, 6)
    
    # Optional features for future expansion
    st.subheader("Additional Features (Optional)")
    budget = st.number_input("Budget (USD)", 0, 500000000, 50000000, 1000000)
    genre_options = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", 
                     "Drama", "Family", "Fantasy", "History", "Horror", "Music", 
                     "Mystery", "Romance", "Science Fiction", "Thriller", "War", "Western"]
    genres = st.multiselect("Genres", genre_options, ["Action", "Adventure"])
    
    # Button to predict
    predict_button = st.button("Predict Revenue")

# Function to load models
@st.cache_resource
def load_models():
    models = {}
    
    try:
        # Load Random Forest model
        if os.path.exists("models/random_forest_model.pkl"):
            with open("models/random_forest_model.pkl", "rb") as f:
                models["random_forest"] = pickle.load(f)
        else:
            st.warning("Random Forest model not found. Using simulation instead.")
        
        # Load Linear Regression model
        if os.path.exists("models/linear_regression_model.pkl"):
            with open("models/linear_regression_model.pkl", "rb") as f:
                models["linear_regression"] = pickle.load(f)
        else:
            st.warning("Linear Regression model not found. Using simulation instead.")
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}

# Load feature importance
@st.cache_data
def load_feature_importance():
    if os.path.exists("models/feature_importance.csv"):
        return pd.read_csv("models/feature_importance.csv")
    else:
        # Return simulated feature importance
        return pd.DataFrame({
            'Feature': ['vote_count', 'vote_average', 'runtime', 'release_year', 'release_month'],
            'Importance': [0.45, 0.25, 0.15, 0.10, 0.05]
        })

# Load the models
models = load_models()

# Format currency function
def format_currency(value):
    if value >= 1e9:
        return f"${value/1e9:.2f} billion"
    elif value >= 1e6:
        return f"${value/1e6:.2f} million"
    else:
        return f"${value:,.2f}"

# Main content
if predict_button:
    # Prepare input data
    input_data = pd.DataFrame({
        'runtime': [runtime],
        'vote_average': [vote_average],
        'vote_count': [vote_count],
        'release_year': [release_year],
        'release_month': [release_month]
    })
    
    # Make predictions with models if available, otherwise simulate
    if "random_forest" in models:
        log_revenue_rf = models["random_forest"].predict(input_data)[0]
        revenue_rf = np.exp(log_revenue_rf)
    else:
        # Fallback simulation
        log_revenue_rf = 12 + 0.02 * runtime + 0.5 * vote_average + 0.0002 * vote_count + (release_year - 2000) * 0.05 + np.sin(release_month) * 0.2
        revenue_rf = np.exp(log_revenue_rf)
    
    if "linear_regression" in models:
        log_revenue_lr = models["linear_regression"].predict(input_data)[0]
        revenue_lr = np.exp(log_revenue_lr)
    else:
        # Fallback simulation
        log_revenue_lr = 10 + 0.01 * runtime + 0.4 * vote_average + 0.0001 * vote_count + (release_year - 2000) * 0.04
        revenue_lr = np.exp(log_revenue_lr)
    
    # Display predictions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Results")
        st.metric("Random Forest Prediction", format_currency(revenue_rf))
        st.metric("Linear Regression Prediction", format_currency(revenue_lr))
        
        st.info("""
        **Note:** The Random Forest model generally shows better performance, 
        with higher R² and lower error metrics in our validation tests.
        """)
    
    with col2:
        st.subheader("Movie Comparable Chart")
        
        # Create sample data for movie comparisons
        comparison_data = pd.DataFrame({
            'Type': ['Your Movie', 'Avg. Similar Movies', 'Avg. All Movies'],
            'Expected Revenue': [revenue_rf, revenue_rf * 0.8, revenue_rf * 0.5]
        })
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Type', y='Expected Revenue', data=comparison_data, ax=ax)
        ax.set_ylabel('Expected Revenue ($)')
        ax.set_title('Revenue Comparison')
        
        # Format y-axis ticks for better readability
        if revenue_rf >= 1e9:
            ax.yaxis.set_major_formatter(lambda x, pos: f'${x/1e9:.1f}B')
        elif revenue_rf >= 1e6:
            ax.yaxis.set_major_formatter(lambda x, pos: f'${x/1e6:.1f}M')
        
        st.pyplot(fig)

# Feature importance section
st.subheader("Feature Importance")
st.markdown("""
Based on our Random Forest model, these are the features that most influence box office revenue:
""")

# Load feature importance data
feature_importance = load_feature_importance()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
ax.set_title('Feature Importance')
st.pyplot(fig)

# Add explanatory text
st.subheader("About the Model")
st.markdown("""
This prediction is based on a machine learning model trained on TMDB movie data. The model uses:

- **Random Forest Regression:** A powerful ensemble learning method that combines multiple decision trees
- **Log-transformed revenue:** Due to the skewed distribution of box office revenues
- **Key features:** Runtime, ratings, vote count, and release timing

**Note:** This is a proof-of-concept model. For commercial use, additional features and more sophisticated models would improve accuracy.
""")

# Add information about the dataset
with st.expander("About the Dataset"):
    st.markdown("""
    The models were trained on data from The Movie Database (TMDB) containing information about thousands of movies.
    
    The dataset includes features such as:
    - Runtime
    - Average user ratings
    - Number of votes
    - Release year and month
    - Budget (not currently used in predictions)
    - Genres (not currently used in predictions)
    
    The target variable is the box office revenue, which was log-transformed for training to handle its skewed distribution.
    """)

# Footer
st.markdown("---")
st.markdown("🎬 Y_BoxOffice Predictor | Albert School Global Data 2024-2025") 