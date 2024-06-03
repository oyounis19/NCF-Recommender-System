import os
os.sys.path.append(os.path.abspath('app/model/'))
abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app/model/')

import pandas as pd
import streamlit as st

from utils.model import NCF, __model_version__
from utils.utils import Utils, cols_dict, occupation, genre, css

@st.cache_data
def load_data():
    users_exp = pd.read_csv(abs_path + 'data/users_exp.csv').values
    users_imp = pd.read_csv(abs_path + 'data/users_imp.csv').values
    movies = pd.read_csv(abs_path + 'data/movies.csv')
    movies_og = pd.read_csv(abs_path + 'data/movies.dat', sep='::', names=cols_dict['items'], encoding='latin-1', engine='python')
    ratings = pd.read_csv(abs_path + 'data/ratings.dat', sep='::', names=cols_dict['ratings'], engine='python')

    return users_exp, users_imp, movies, movies_og, ratings

@st.cache_resource
def load_models():
    model_exp = NCF('explicit', gpu=False)
    model_exp.load_weights(abs_path + 'weights/explicit.pth', eval=True)

    model_imp = NCF('implicit', gpu=False)
    model_imp.load_weights(abs_path + 'weights/implicit.pth', eval=True)

    return model_exp, model_imp

users_exp, users_imp, movies, movies_og, ratings = load_data()
model_exp, model_imp = load_models()

# GUI
st.title('NCF Recommender System')

st.write(f'Models version: {__model_version__}')

model_type = st.radio('Select model type', ['Implicit', 'Explicit'])

new_user = st.checkbox('New user? (no user ID needed)', value=True)

# Input number of recommendations
top_k = st.number_input('Number of recommendations', min_value=1, max_value=20, value=10, step=1)

# User ID input
user_id = st.number_input('User ID (MAX: 6040)', min_value=1, max_value=6040, value=3000, step=1, disabled=new_user)

# New user inputs
user_gender = st.selectbox('Gender', ['M', 'F'], disabled=not new_user)
user_age = st.number_input('Age', min_value=1, max_value=99, value=25, step=1, disabled=not new_user)
user_occupation = st.selectbox('Job', occupation, disabled=not new_user, help='Select your job', index=17)
user_genres = st.multiselect('Favourite genres', genre, disabled=not new_user, help='Select at least 3 genres', default=['Comedy', 'Children', 'Animation'], max_selections=5)

# Get recommendations button
recommend = st.button('Get Recommendations')

# create the user dict
user = {
    'top_k': top_k,
    'id': user_id if not new_user else 9000,
    'age': user_age if new_user else None,
    'gender': user_gender if new_user else None,
    'occupation': user_occupation if new_user else None,
    'genres': user_genres if new_user else None
}

# Get recommendations
if recommend and 5 >= len(user_genres) >= 3:
    pred_movies, top_n_genres = Utils.pipeline(
        request=user,
        model=model_exp if model_type == 'Explicit' else model_imp,
        users=users_exp if model_type == 'Explicit' else users_imp,
        movies=movies,
        movies_og=movies_og,
        ratings=ratings,
        weights=[model_exp.user_embedding_mlp.weight.data.cpu().numpy(), model_exp.user_embedding_mf.weight.data.cpu().numpy()] if model_type == 'Explicit' else [model_imp.user_embedding_mlp.weight.data.cpu().numpy(), model_imp.user_embedding_mf.weight.data.cpu().numpy()],
        mode=model_type.lower()
    )

    # Display recommendations
    st.write(f'Top {top_k} recommendations for user {user_id}:' if not new_user else f'Top {top_k} recommendations for new user:')
    # st.write(pred_movies, unsafe_allow_html=True)
    if not new_user:
        st.write(f'Top genres user with ID {user_id} like: {", ".join(top_n_genres)}')

    pred = 'rating' if model_type == 'Explicit' else 'score'

    html = """<div class="card-container">"""
    for i, movie in enumerate(pred_movies):
        # create the movie card
        html += f"""<div class="card">
                <h5 class="card-title">{i + 1}</h5>
                <p class="card-text">Title: <b style="font-size: 1.2em;">{movie['title']}</b></p>
                <p class="card-text">Genres: {movie['genre']}</p>
                <p class="card-text">Predicted {pred}: {movie['predicted_score'] if model_type == 'Implicit' else movie['predicted_rating']}</p>
            </div>"""

    st.markdown(
        html + '</div>',
        unsafe_allow_html=True
    )

elif recommend and len(user_genres) < 3:
    st.write('Please select 3 to 5 genres')

# Fixed footer
st.markdown(
    css,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="footer">
        Made with ❤️ by <a href="https://www.linkedin.com/in/omar-younis-3b57a8230">Omar Younis</a>
    </div>
    """,
    unsafe_allow_html=True
)