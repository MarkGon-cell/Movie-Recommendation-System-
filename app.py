import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Movie Dashboard", layout="wide")

st.write("✅ App Started Successfully")

# ---------------- HEADER ----------------
st.markdown("""
<div style='background: linear-gradient(to right, #ff7e5f, #6a82fb);
padding: 20px; border-radius:10px'>
<h1 style='color:white; text-align:center;'>🎬 Movie Recommendation Dashboard</h1>
<p style='color:white; text-align:center;'>K-Means + Random Forest</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")

    # 🔥 LIMIT DATA FOR SPEED
    df = df.head(800)

    df = df[['title', 'genres', 'overview', 'vote_average']]

    # Fix genres JSON
    def extract_genres(x):
        try:
            genres = ast.literal_eval(x)
            return " ".join([g['name'] for g in genres])
        except:
            return ""

    df['genres'] = df['genres'].apply(extract_genres)

    df['overview'] = df['overview'].fillna("")
    df['genres'] = df['genres'].fillna("")

    # Remove invalid ratings
    df = df[df['vote_average'] > 0]

    df['features'] = df['overview'] + " " + df['genres']

    return df

movies = load_data()

# ---------------- MODEL TRAINING ----------------
with st.spinner("⏳ Training models... please wait"):
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(movies['features'])

    # KMeans (lighter)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    movies['cluster'] = kmeans.fit_predict(X)

    # Random Forest (lighter)
    y = movies['vote_average']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)

# ---------------- SIDEBAR ----------------
st.sidebar.header("🎯 Choose Movie")
selected_movie = st.sidebar.selectbox("Select a movie", movies['title'].values)

# ---------------- RECOMMEND ----------------
def recommend(movie):
    idx = movies[movies['title'] == movie].index[0]
    
    vec = tfidf.transform([movies.iloc[idx]['features']])
    similarities = (X @ vec.T).toarray().flatten()

    similar_indices = similarities.argsort()[-7:][::-1]

    return movies.iloc[similar_indices[1:]]

# ---------------- STATS ----------------
st.markdown("## 📊 Dashboard Stats")

col1, col2, col3 = st.columns(3)

col1.metric("Total Movies", len(movies))
col2.metric("Clusters", movies['cluster'].nunique())
col3.metric("Avg Rating", round(movies['vote_average'].mean(), 2))

# ---------------- RECOMMENDATIONS ----------------
st.markdown("## 🎥 Recommended Movies")

if st.button("Get Recommendations"):
    recs = recommend(selected_movie)

    cols = st.columns(3)

    for i, (_, row) in enumerate(recs.iterrows()):
        with cols[i % 3]:

            vec = tfidf.transform([row['features']])
            predicted_rating = round(rf_model.predict(vec)[0], 2)

            st.markdown(f"""
            <div style='background:#1e1e1e;padding:15px;border-radius:10px'>
            <h4 style='color:white'>{row['title']}</h4>
            <p style='color:gray'>{row['genres']}</p>
            <p style='color:orange'>⭐ Actual: {round(row['vote_average'],1)}</p>
            <p style='color:lightgreen'>🤖 Predicted: {predicted_rating}</p>
            </div>
            """, unsafe_allow_html=True)

# ---------------- ANALYTICS DASHBOARD ----------------
st.markdown("## 📊 Advanced Movie Analytics")

import matplotlib.pyplot as plt
import seaborn as sns

# -------- Rating Distribution --------
st.subheader("⭐ Rating Distribution")
fig, ax = plt.subplots()
sns.histplot(movies['vote_average'], bins=20, kde=True, ax=ax)
st.pyplot(fig)

# -------- Top Genres --------
st.subheader("🎭 Top Genres")
all_genres = movies['genres'].str.split().explode()
genre_counts = all_genres.value_counts().head(10)

fig, ax = plt.subplots()
genre_counts.plot(kind='bar', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# -------- Cluster Distribution --------
st.subheader("📊 Cluster Distribution")
fig, ax = plt.subplots()
sns.countplot(x=movies['cluster'], ax=ax)
st.pyplot(fig)

# -------- Ratings per Cluster --------
st.subheader("📈 Ratings by Cluster")
fig, ax = plt.subplots()
sns.boxplot(x='cluster', y='vote_average', data=movies, ax=ax)
st.pyplot(fig)

# -------- Top Rated Movies --------
st.subheader("🔥 Top Rated Movies")
top_movies = movies.sort_values(by='vote_average', ascending=False).head(10)
st.dataframe(top_movies[['title', 'genres', 'vote_average']])