import numpy as np
import random as r
import pandas as pd


def generate_low_rank(n,p,k):
    """Cette fonction génère une matrice de notations aléatoire de taille n*p et de rang k"""
    matrix = r.uniform(1, 5, (n, p)) #matrice tirée aléatoirement de façon uniforme à coefficients dans {1,...,5}
    U, S, V = np.linalg.svd(matrix) #décomposition en valeurs singulières
    S[k:] = 0 #On laisse les k plus grandes valeurs singulières et on remet les autres à 0
    matrix = np.dot(U * S, V)
    return matrix

def generate_entries(n,p,k):
    """"Cette fonction tire k entrées aléatoires à conserver dans une matrice n*p (les autres seront cachées)"""
    L=[]
    for _ in range(k):
        i=r.randint(0,n-1)
        j=r.randint(0,p-1)
        L.append((i,j))
    return L

def extract_complete_matrix(df,alpha):
    """Cette fonction est destinée à extraire des sous-matrices complètes à partir
    des données du netflix challenge"""
    user_ratings_count = df.groupby('user_id').size()
    movie_ratings_count = df.groupby('movie_id').size()
    significant_users = user_ratings_count[user_ratings_count >=0].index
    significant_movies = movie_ratings_count[movie_ratings_count >=  (len(df['user_id'].unique()) * alpha)].index
    sub_df = df[(df['user_id'].isin(significant_users)) & (df['movie_id'].isin(significant_movies))]
    user_movie_counts = sub_df.groupby('user_id')['movie_id'].nunique()
    complete_users = user_movie_counts[user_movie_counts == len(significant_movies)].index
    sub_df = df[(df['user_id'].isin(complete_users)) & (df['movie_id'].isin(significant_movies))]
    pivot_df=sub_df.pivot(index='user_id', columns='movie_id', values='rating')
    all_movie_ids = sub_df['movie_id'].unique()
    pivot_df = pivot_df.reindex(columns=all_movie_ids)
    M = pivot_df.values
    return M
