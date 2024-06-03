import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from typing import Literal, Union
import matplotlib.pyplot as plt

from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

class Utils:
    @staticmethod
    def extract_year(items_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Extracts:
            - Year from title

        returns: Dataframes with extracted features
        '''
        # Extract year from title
        items_df['year'] = items_df['title'].str.extract(r'\((\d{4})\)').astype(int)

        return items_df

    @staticmethod
    def extract_category_avg_ratings(users_df: pd.DataFrame, items_df: pd.DataFrame, ratings_df: pd.DataFrame, k=0.6) -> pd.DataFrame:
        '''
        Extracts penalized average ratings for each category for each user using exponential decay penalty.
        
        users_df: DataFrame containing user information.
        items_df: DataFrame containing item information with categories as binary columns.
        ratings_df: DataFrame containing user-item interactions and ratings.
        k: Control factor for penalty steepness. Default is 0.6.
        
        Returns a DataFrame with users and their penalized average ratings per category.
        '''
        # Create a copy of users_df to store features
        features_df = users_df.copy()

        # Define exponential penalty function
        def exp_penalty(n, k=0.6):
            return 1 / np.exp(k * n)

        # Iterate over each category in the items_df (excluding non-categorical columns)
        for category in items_df.columns[2:]:
            # Get item IDs in the current category
            category_items = items_df[items_df[category] == 1]['movie_id']

            # Filter ratings_df to include only ratings for items in the current category
            category_ratings = ratings_df[ratings_df['movie_id'].isin(category_items)]

            # Group by user_id and calculate the average rating and count of ratings for the current category
            user_stats = category_ratings.groupby('user_id')['rating'].agg(['mean', 'count']).reset_index()
            user_stats.columns = ['user_id', f'user_avg_rating_{category}', f'count_rating_{category}']

            # Merge the user stats into features_df
            features_df = pd.merge(features_df, user_stats, on='user_id', how='left')

            # Apply exponential penalty and calculate the penalized average rating for each user
            features_df[f'user_avg_rating_{category}'] = (
                (1 - exp_penalty(features_df[f'count_rating_{category}'], k)) * features_df[f'user_avg_rating_{category}']
            )

            # Fill missing values with 0
            features_df[f'user_avg_rating_{category}'] = features_df[f'user_avg_rating_{category}'].fillna(0)

        # Select only the relevant columns
        cols = features_df.columns[:32].tolist() + [col for col in features_df.columns if col.startswith('user_avg_rating_')]
        result_df = features_df[cols]

        return result_df

    @staticmethod
    def extract_category_freq(users_df: pd.DataFrame, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        # Copy users_df to avoid modifying the original dataframe
        freq_df = users_df.copy()

        # Get total interactions for each user
        total_interactions = ratings_df.groupby('user_id').size().reset_index(name='total_interactions')
        freq_df = pd.merge(freq_df, total_interactions, on='user_id', how='left').fillna(0)
        
        # Iterate over each category in the items_df (assuming categories are from the 3rd column onward)
        for category in items_df.columns[2:]:
            # Get movie_ids that belong to the current category
            movie_ids_in_category = items_df[items_df[category] == 1]['movie_id']
            
            # Count interactions in the current category for each user
            category_interactions = ratings_df[ratings_df['movie_id'].isin(movie_ids_in_category)].groupby('user_id').size().reset_index(name=f'{category}_count')
            
            # Merge category_interactions with freq_df
            freq_df = pd.merge(freq_df, category_interactions, on='user_id', how='left').fillna(0)
            
            # Calculate frequency of interactions for the current category
            freq_df[f'freq_{category}'] = freq_df[f'{category}_count'] / freq_df['total_interactions']
            
            # Drop the intermediate category count column
            freq_df.drop(columns=[f'{category}_count'], inplace=True)
        
        return freq_df.fillna(0).drop(columns=['total_interactions'])

    @staticmethod
    def extend_users_items(users_df: pd.DataFrame, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        Extends users and items dataframes to match the ratings dataframe
        '''
        # Extend users dataframe
        users_df = pd.merge(users_df, ratings_df[['user_id']], on='user_id', how='right')

        # Extend items dataframe
        items_df = pd.merge(items_df, ratings_df[['movie_id']], on='movie_id', how='right')
        
        return users_df, items_df
    
    @staticmethod
    def multi_hot_encode(df: pd.DataFrame, col: str, delimiter='|') -> pd.DataFrame:
        '''
        Multi hot encodes columns in a dataframe
        '''
        df_ = df.copy(deep=True)

        # Change Children's to Children to match the other genres
        df_[col] = df_[col].str.replace("Children's", 'Children') if col == 'genre' else df_[col]

        # split genres
        df_[col] = df_[col].str.split(delimiter)

        # Create a pivot table
        pivot_df = df_.explode(col).pivot_table(index='movie_id', columns=col, aggfunc='size', fill_value=0).reset_index()

        # Merge the pivot table with the original DataFrame on 'movie_id'
        result = pd.merge(df, pivot_df, on='movie_id', how='left')
        
        return result.drop(columns=[col])
    
    @staticmethod
    def one_hot_encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        '''
        One hot encodes columns in a dataframe
        '''
        return pd.get_dummies(df, columns=cols) * 1
    
    @staticmethod
    def move_column(df: pd.DataFrame, col: list[str], pos: int) -> pd.DataFrame:
        '''
        Moves a column to a specific position in a DataFrame
        '''
        cols = df.columns.tolist()
        for i in reversed(col):
            cols.insert(pos, cols.pop(cols.index(i)))
        return df[cols]

    @staticmethod
    def preprocess_user(user: dict, num_items: int, users: np.ndarray, weights: list[np.ndarray]=None, topk: int=3, verbose=False) -> tuple[torch.IntTensor, torch.FloatTensor, Union[list[np.ndarray], None]]:
        '''
        Preprocesses user data for model input
        '''
        if 'age' not in user or user['age'] == None:
            user_ = users[user['id'] - 1]
            user_ = np.insert(user_, 0, user['id'])
            print(f"User id: {user['id']} top {topk} genres: {np.array(genre)[np.argsort(user_[-18:])[-topk:][::-1]]}") if verbose else None
            user_ = np.tile(user_, (num_items, 1))
            return torch.IntTensor(user_[:, 0]), torch.FloatTensor(user_[:, 1:]), None

        user_ = np.zeros(31, dtype=float)

        user_[0] = user['id']

        user_[1 if user['gender'] == 'M' else 2] = 1

        user_[3 + occupation.index(user['occupation'])] = 1

        # map age to bins
        user['age'] = 1 if user['age'] < 18 else 18 if user['age'] < 25 else 25 if user['age'] < 35 else 35 if user['age'] < 45 else 45 if user['age'] < 56 else 56

        user_[3 + len(occupation) + age.index(user['age'])] = 1

        avg_ratings = np.zeros(len(genre), dtype=float) # 18 genres

        for genre_ in user['genres']:
            avg_ratings[genre.index(genre_)] = 1.0

        user_ = np.concatenate((user_, avg_ratings))

        # Get top 10 users ids of users with similar intrests (cosine similarity)
        similar_users_ids = cosine_similarity(user_[1:].reshape(1, -1), users).argsort()[0][-10:]

        # Get the mean embeddings of the top 10 similar users
        mlp_weights = weights[0][similar_users_ids].mean(axis=0)
        mf_weights = weights[1][similar_users_ids].mean(axis=0)

        user_ = np.tile(user_, (num_items, 1))
        return torch.IntTensor(user_[:, 0]), torch.FloatTensor(user_[:, 1:]), [mlp_weights, mf_weights]
    
    @staticmethod
    def preprocess_items(items: pd.DataFrame) -> pd.DataFrame:
        '''
        Preprocesses items data for model input
        '''
        # multi hot encode genres
        items_ = Utils.multi_hot_encode(items, 'genre')
        items_ = Utils.extract_year(items_)
        items_['year'] = items_['year'] / items_['year'].max()
        items_ = items_.drop(['title'], axis=1)
        
        return items_

    @staticmethod
    def remove_missing_values(ratings: pd.DataFrame, items: pd.DataFrame) -> tuple[pd.DataFrame]:
        '''
        Removes rows with missing values from items and ratings dataframes
        '''
        # Get item ids with missing release dates
        nan_item_ids = items[items[['release_date']].isna().any(axis=1)]['item_id']

        # Remove movies with missing release dates
        items.dropna(subset=['release_date'], inplace=True)

        # remove ratings of missing items
        ratings = ratings[~ratings['item_id'].isin(nan_item_ids)]

        return ratings, items

    @staticmethod
    def negative_sampling(ratings: pd.DataFrame, items: pd.DataFrame, num_negatives: int) -> pd.DataFrame:
        '''
        Sample negative items for each user
        '''
        # All Movie ids
        all_items = items['movie_id'].values

        negative_samples = []
        for user_id in ratings['user_id'].unique():
            # Movie ids that the user has interacted with
            pos_items = ratings[ratings['user_id'] == user_id]['movie_id'].values

            # Movie ids that the user has not interacted with
            unrated_items = np.setdiff1d(all_items, pos_items)

            # Sample negative items
            neg_items = np.random.choice(unrated_items, size=num_negatives, replace=False)

            # Create negative samples
            for item_id in neg_items:
                negative_samples.append([user_id, item_id, 0])

        negative_samples = pd.DataFrame(negative_samples, columns=['user_id', 'movie_id', 'rating'])

        ratings['rating'] = [1] * ratings.shape[0]

        return pd.concat([ratings, negative_samples], ignore_index=True)
    
    def ndcg_hit_ratio(y_preds, X_test_users, y_true, k=10) -> tuple[float]:
        '''
        Compute NDCG
        '''
        unique_users = np.unique(X_test_users, axis=0)

        hits = 0
        total_users = len(unique_users)

        y_preds_padded = []
        y_true_padded = []
        for user in unique_users:
            # Get the indices of the user
            user_indices = np.where((X_test_users == user).all(axis=1))[0]
            # Get the predictions for the user
            user_preds = y_preds[user_indices][:k].flatten()
            # Get the true ratings for the user
            user_true = y_true[user_indices][:k].flatten()

            # Calculate the number of hits
            if np.any(user_true == 1):
                hits += 1

            # Pad the sublists to have k elements
            if len(user_preds) < k:
                user_preds = np.pad(user_preds, (0, k - len(user_preds)), mode='constant', constant_values=-1e10)
            if len(user_true) < k:
                user_true = np.pad(user_true, (0, k - len(user_true)), mode='constant', constant_values=0)

            y_preds_padded.append(user_preds)
            y_true_padded.append(user_true)

        # Compute NDCG
        ndcg = ndcg_score(y_true_padded, y_preds_padded, k=k)
        # Compute hit ratio
        hit_ratio = hits / total_users
        return ndcg, hit_ratio
    
    @staticmethod
    def pipeline(request: any, model: nn.Module, weights: list[np.ndarray], users: np.ndarray, movies: pd.DataFrame, movies_og: pd.DataFrame, ratings: pd.DataFrame, mode: str):
        '''
        Pipeline for inference
        '''
        num_items = 200 # Number of items to retrieve
        request = request if isinstance(request, dict) else request.model_dump()

        # preprocess the old user
        user_id, user, weights = Utils.preprocess_user(
                                        user=request,
                                        num_items=num_items,
                                        users=users,
                                        weights=weights,
                                        verbose=True
                                        )
        user_id, user = user_id.to(model.device), user.to(model.device)

        movies = Utils.retrieve(
            movies=movies,
            user=user.detach().cpu().numpy(),
            num_genres=len(request['genres']) if request['genres'] else 3,
            k=num_items,
            random_state=0
        )

        movie_ids, movies = Utils.filter(
            movies=movies,
            ratings=ratings,
            user_id=request['id']
        )
        movie_ids, movies = movie_ids.to(model.device), movies.to(model.device)

        y_pred = model(
            user_id[:len(movies)],
            movie_ids,
            user[:len(movies)],
            movies,
            weights
        ).cpu().detach().numpy()

        movies_retrieved = movies_og[movies_og['movie_id'].isin(movie_ids.cpu().numpy())].sort_values(by='movie_id', key=lambda x: pd.Categorical(x, categories=movie_ids.cpu().numpy(), ordered=True))

        return Utils.order(y_pred, movies_retrieved, mode, top_k=request['top_k']).to_dict(orient='records')
    
    @staticmethod
    def retrieve(movies: pd.DataFrame, user: np.ndarray, k: int, num_genres: int=3, random_state: int=42) -> pd.DataFrame:
        '''
        Retrieve top k movies based on genres based on this equation:
        ```
        num_movies_per_genre = k // (len(genres) + 1) # +1 for the most popular genre
        ```

        Example:
        If k = 100 and genres = ['Action', 'Adventure', 'Animation'], then:
        25 movies will be retrieved for each genre and 25 for the most popular genre.

        movies: DataFrame containing movie information.
        genres: List of genres to retrieve movies for.
        k: Number of movies to retrieve.

        Returns a DataFrame containing the top k movies based on the specified genres.
        '''
        num_movies_per_genre = k // (num_genres + 1)
        most_popular_genres = ['Drama', 'Comedy', 'Action'] # In a real scenario, this would change weekly based on trendy genres

        # Get the 3 most liked genres by the user
        top_n_genres = np.array(genre)[np.argsort(user[0, -18:])[-num_genres:][::-1]]
        
        movies_ = []
        # Retrieve movies for each genre randomly, since we don't have movie ratings to sort by
        for g in top_n_genres:
            m = movies[movies[g] == 1]
            if m.shape[0] < num_movies_per_genre: # Check if there are enough movies for the genre
                movies_.append(m)
                continue
            movies_.append(m.sample(num_movies_per_genre, random_state=random_state))                
        
        # Retrieve movies for the most popular genres
        for g in most_popular_genres:
            m = movies[movies[g] == 1]
            if m.shape[0] < num_movies_per_genre//3: # Check if there are enough movies for the genre
                movies_.append(m)
                continue
            movies_.append(movies[movies[g] == 1].sample(num_movies_per_genre//3, random_state=random_state))

        return pd.concat(movies_, ignore_index=True)
    
    @staticmethod
    def filter(movies: pd.DataFrame, ratings: pd.DataFrame, user_id: int) -> tuple[torch.IntTensor, torch.FloatTensor]:
        '''
        Filter movies that the user has not interacted with, and remove duplicates
        '''
        # Get movie ids that the user has interacted with
        user_movies = ratings[ratings['user_id'] == user_id]['movie_id'].values

        # Filter movies that the user has not interacted with
        movies = movies[~movies['movie_id'].isin(user_movies)]

        # Remove duplicates
        movies = movies.drop_duplicates(subset=['movie_id'])

        return torch.IntTensor(movies['movie_id'].values), torch.FloatTensor(movies.drop(columns=['movie_id']).values)
    
    @staticmethod
    def order(y_pred: np.ndarray, movies: pd.DataFrame, mode: Literal['explicit', 'implicit'], top_k=10) -> list[dict]:
        '''
        Order the predictions
        '''
        col_name= 'predicted_rating' if mode == 'explicit' else 'predicted_score'
        sorted_index = np.argsort(-y_pred, axis=0).reshape(-1).tolist()
        y_pred = y_pred[sorted_index]
        sorted_movies = movies.iloc[sorted_index]
        sorted_movies = sorted_movies.copy()
        sorted_movies[col_name] = y_pred if mode == 'implicit' else y_pred * 5
        sorted_movies.reset_index(drop=True, inplace=True)
        sorted_movies[col_name] = sorted_movies[col_name].apply(lambda x: round(x, 2))

        return sorted_movies.head(top_k)
    
    @staticmethod
    def plot_metrics(history: dict, title: str, figsize: tuple=(12, 4)) -> None:
        '''
        Plot the training and validation losses in one figure and the other metrics in another figure
        '''
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].plot(history['loss'], label='Train Loss')
        ax[0].plot(history['val_loss'], label='Validation Loss')
        ax[0].set_title('Training and Validation Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        # Metrics plot
        for metric, values in history.items():
            if metric not in ['loss', 'val_loss']:
                ax[1].plot(values, label=metric)
            
        ax[1].set_title('Metrics')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Value')
        ax[1].legend()
        plt.suptitle(title)
        plt.show()

class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after a certain number of epochs (patience).
    """
    def __init__(self, patience=3, delta=0, verbose=False, path='checkpoint.pth') -> None:
        """
        Args:
            patience (int): How many epochs to wait after the last time the validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss, model: nn.Module) -> None:
        '''
        Call method
        '''
        score = -val_loss

        if not self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}') if self.verbose else None
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model: nn.Module) -> None:
        '''
        Save the model checkpoint
        '''
        print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...') if self.verbose else None
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss

age = [
    1, 18, 25, 35, 45, 50, 56
]

occupation = [
    'other', 'educator', 'artist', 'clerical', 'grad student',
    'customer service', 'doctor', 'executive', 'farmer', 'homemaker',
    'K-12 student', 'lawyer', 'programmer', 'retired', 'sales', 'scientist',
    'self-employed', 'engineer', 'craftsman', 'unemployed', 'writer'
]

genre = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

cols_dict = {
    'ratings': ['user_id', 'movie_id', 'rating', 'timestamp'],
    'users': ['user_id', 'gender', 'age', 'occupation', 'zip_code'],
    'items': ['movie_id', 'title', 'genre'],
}