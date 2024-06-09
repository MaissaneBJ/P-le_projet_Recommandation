#importation des bibliothéques nécessaires 
import numpy as np
import random
from tqdm import tqdm
import math as math
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans



#générer une matrice incompléte à partir d'une matrice compléte
def generate_incomplete_matrix(M:np,p:int)-> np:
    n,m=M.shape
    all_tuples = [(i, j) for i in range(n) for j in range(m)]
    omega = random.sample(all_tuples,p)
    new_matrix = np.zeros((n,m))
    for i in tqdm(range (n)):
        for j in tqdm(range (m)):
            if (i,j) in omega:
                new_matrix [i][j] = M[i][j]
    return new_matrix


#donner l'évaluation moyenne des utilisateurs. 
def rating_average (M:np) -> dict :
    """ la fontion retourne un dictionnaire contenant chaque utilisateur et sa notation moyenne"""
    n,p = M.shape
    rating_average={}
    for user in range (n):
        nbr = np.count_nonzero(M[user,:])
        if nbr!=0:
            user_average = np.sum(M[user,:])/nbr
        else:
            user_average = 0 #si l'utilisateur n'a noté aucun film
        rating_average[user] = user_average
    return rating_average



#donner l'évaluation moyenne d'un film
def movie_average(M:np, movie:int) -> int:
    nbr = np.count_nonzero(M[:,movie])
    if nbr != 0:
        movie_average = np.sum(M[:,movie])/nbr
    else:
        movie_average = 0 #si le film n'a été noté par aucun utilisateur
    return movie_average


    
#Calculer le coefficient de similarité entre les utilisateurs qui sera utilisé pour la prédicition de l'évaluation 
def similarity_rating_score (M : np) -> np :
    n,p = M.shape
    dict_rating_average = rating_average(M) #le dictionnaire contenant la moyenne des évaluations de chaque utilisateur
    Sr = np.zeros ((n,n)) #matrice de similarité
    #Calculer la similarité entre deux paires d'utilisateurs
    for user_1 in range (n): #pour chaque utilisateur
        for user_2 in range (user_1+1 , n): 
            sum1 = 0
            sum21 = 0
            sum22 = 0
            for movie in range (p):
                if (M[user_1][movie] != 0 ) and (M[user_2][movie] != 0 ):
                    sum1 += (M[user_1][movie] -  dict_rating_average[user_1]) * (M[user_2][movie] -  dict_rating_average[user_2]) 
                    sum21 += (M[user_1][movie] -  dict_rating_average[user_1]) ** 2
                    sum22 += (M[user_2][movie] -   dict_rating_average[user_2]) ** 2 
                    
            #Le score de similarité est symmetrique
            if math.sqrt(sum21) * math.sqrt(sum22) != 0:
               Sr [user_1][user_2] = sum1/(math.sqrt(sum21) * math.sqrt(sum22) )
               Sr [user_2][user_1] = sum1/(math.sqrt(sum21) * math.sqrt(sum22) )
            else: #si la somme est égale à zéro alors le coefficient de similarité est infini, s'il tend vers +inf alors les deux utilisateurs sont fortement similaires sinon ils sont fortement différents
                #Puisque la valeur maximale que peut prendre le coefficient en valeur absolue est 1, le coefficient sera égal à 1 s'il est +inf et -1 s'il est -inf
                Sr [user_1][user_2] = np.sign(sum1) 
                Sr [user_2][user_1] = np.sign(sum1)
        #la similarité d'un utilisateur avec lui même est égale à 1
        Sr [user_1][user_1] = 1
    return Sr


#La méthode de K_mean
def clustering (M:np , k:int):
    kmeans = KMeans(n_clusters=k) #étant donné un nombre de clusters connu k 
    kmeans.fit(M) 
    centroids = kmeans.cluster_centers_ #un array contenant les positions des centres de chaque cluster
    labels = kmeans.labels_ #un array contenant le groupe auquel appartient chaque utilisateur 
    return centroids,labels


#Recherche de K_optimal
def optimal_k ( M:np ) -> int :
    n,p = M.shape
    k_values = range (2,n)#le nombre de groupe est entre 2 et n-1,
    #au maximum on peut avoir n-1 groupes ou deux utilisateurs uniquement appartiennet au même groupe et les autres occupent des chacun un groupe différent
    #au minimum on peut avoir deux groupes qui regroupent la totalité des utilisateurs
    silhouette_scores = [] #K_optimal choisi est celui qui maximise le coefficient de silhouette 
    for k in k_values:
        centroids , label = clustering (M, k)
        silhouette_avg = silhouette_score(M, label)
        silhouette_scores.append(silhouette_avg)
    if not silhouette_score :
        raise ValueError('silhouette_score is empty')
    return silhouette_scores.index(max(silhouette_scores)) + 2



#déterminer les différents utilisateurs appartenant à un même ensemble
def clusters_matrix(M: np) -> list:
    n, p = M.shape
    k_op = optimal_k(M)
    centroids, labels = clustering(M, k_op)
    clusters = [[] for i in range(k_op)]
    for user in range(n):
        clusters[labels[user]].append(user)
    return clusters, labels


#compléter la matrice incompléte
def complete_matrix (M:np) -> np :
    """ cette fonction prend en paramétre une matrice de recommandation incompléte, les évaluations manquantes sont remplacées par zero
    et retourne une matrice de recommandation compléte en remplaçant les valeurs manquantes par la valeur prédite"""
    n,p = M.shape
    Sr = similarity_rating_score (M)#matrice de similarité
    clusters , labels = clusters_matrix (M) #les différents groupes qui regroupent les utilisateurs
    dict_rating_average = rating_average(M) #un dictionnaire contenant l'évaluation moyenne de chaque utilisateur 
    for user in range (n):
        for movie in range (p):
            if M[user][movie] == 0:
                predicted_rating = 0
                Sum_1 = 0
                Sum_2 = 0
                for user_i in clusters[labels[user]]:
                    if M[user_i][movie] !=0 :
                        Sum_1 += Sr[user][user_i] * ( M[user_i][movie] - dict_rating_average [user_i] )
                        Sum_2 += abs(Sr[user][user_i])
                if Sum_2 != 0: 
                    predicted_rating = dict_rating_average [user] + (Sum_1 / Sum_2)
                else: #si la somme est nulle, on remplace la valeur prédite par la moyenne de l'évaluation du film 
                    predicted_rating = movie_average(M,movie)
                M[user][movie] = predicted_rating
    return M



