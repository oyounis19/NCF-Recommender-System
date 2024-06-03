import os
os.sys.path.append(os.path.abspath('app/model/'))
abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app/model/')

import pandas as pd
import streamlit as st

from utils.model import NCF, __model_version__
from utils.utils import Utils, cols_dict, occupation, genre

# Load data
users_exp = pd.read_csv(abs_path + 'data/users_exp.csv').values
users_imp = pd.read_csv(abs_path + 'data/users_imp.csv').values
movies = pd.read_csv(abs_path + 'data/movies.csv')
movies_og = pd.read_csv(abs_path + 'data/movies.dat', sep='::', names=cols_dict['items'], encoding='latin-1', engine='python')
ratings = pd.read_csv(abs_path + 'data/ratings.dat', sep='::', names=cols_dict['ratings'], engine='python')

model_exp = NCF('explicit', gpu=False)
model_exp.load_weights(abs_path + 'weights/explicit.pth', eval=True)

model_imp = NCF('implicit', gpu=False)
model_imp.load_weights(abs_path + 'weights/implicit.pth', eval=True)

# GUI
st.title('NCF Recommender System')

st.write(f'Models version: {__model_version__}')

model_type = st.radio('Select model type', ['Implicit', 'Explicit'])

# Input number of recommendations
top_k = st.number_input('Number of recommendations', min_value=1, max_value=20, value=10, step=1)

# Input user ID
user_id = st.number_input('User ID (old user <= 6040)', min_value=1, max_value=10_000, value=1, step=1)

# New user Input user data
st.write('If new user:')

user_gender = st.selectbox('Gender (optional)', ['M', 'F'])
user_age = st.number_input('Age (optional)', min_value=1, max_value=99, value=25, step=1)
user_occupation = st.selectbox('Job (optional)', occupation)
user_genres = st.multiselect('Favourite genres (optional)', genre)

# create the user dict
user = {
    'top_k': top_k,
    'id': user_id,
    'age': user_age,
    'gender': user_gender,
    'occupation': user_occupation,
    'genres': user_genres
}

# Get recommendations
if st.button('Get Recommendations'):
    pred_movies = Utils.pipeline(
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
    st.write(f'Top {top_k} recommendations for user {user_id}:')
    st.write(pred_movies, unsafe_allow_html=True)

