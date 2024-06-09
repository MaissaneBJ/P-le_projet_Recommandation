import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import copy

def generate_M(Y,omega):
    """cette fonction retourne un dictionnaire M dont les clés sont les indices (i,j) des entrées observées et les valeurs
    sont les entrées Mij
    Args:
        Y (ndarray): matrice d'évaluations
        omega (set): l'ensemble des indices des entrées observées

    Returns:
        dict: le dictionnaire en question
    """
    M={}
    for i,j in omega:
        M[(i,j)]=Y[i,j]
    return M

########################################### Calculer l'erreur ##################################################

def compute_error(U,V,M,lamda,number_of_samples):
    """ Cette fonction calcule la fonction de perte pour un certain U(matrice des facteurs utilisateurs), 
    V(matrice des facteurs objets), M(matrice des evaluations) et lambda (régularisation)

    Args:
        U (ndarray): matrice de taille n*K
        V (ndarray): matrice de taille m*K
        M (dict): dictionnaire dont les clés sont les indices des entrées observées
        lamda (float): terme de régularisation
        number_of_samples (int): nombre d'entrées observées

    Returns:
        float: la valeur de la fonction de perte
    """
    L = np.dot(U,np.transpose(V))
    sum=0

    for ind in list(M.keys()):
        
        sum +=(1/(2*number_of_samples))*( (M[(ind[0],ind[1])] - L[ind[0],ind[1]])**2  )
    sum+=(lamda/(2*number_of_samples))*(np.linalg.norm(U)**2 +np.linalg.norm(V)**2)
    return sum

######################################### Calculer le gradient ##################################################

def compute_gradient_U(U,V,dict_i, lamda, number_of_smaples):
    """Cette fonction calcule le gradient de la fonction de perte par rapport à U

    Args:
        U (ndarray): matrice des facteurs utilisateurs
        V (ndarray): matrice des facteurs objets
        dict_i (dict): dictionnaire dont les clés sont les lignes i où il ya une entrée observée et dont la valeur et un dictionnaire
        dont les clés sont les colonnes j telles que (i,j) est un indice d'une entrée observée et dont la valeur et l'entrée Mij
        lamda (float): terme de régularisation
        number_of_smaples (int): nombre d'entrées observées

    Returns:
        ndarray: matrice jacobienne de la fonction de perte par rapport à U
    """
    m,r=U.shape
    N=np.dot(U,np.transpose(V))
    dJdU=np.zeros((m,r))
    for i in range(m):
        for j in range(r):
            if i in dict_i:
                for k in dict_i[i]:
                    dJdU[i,j]-=(dict_i[i][k]-N[i,k])*(1/number_of_smaples)*V[k,j]
                dJdU[i,j]+=(1/number_of_smaples)*lamda*U[i,j]
    return dJdU

def compute_gradient_V(U,V,dict_j,lamda,number_of_smaples):
    """Cette fonction calcule le gradient de la fonction de perte par rapport à V

    Args:
        U (ndarray): matrice des facteurs utilisateurs
        V (ndarray): matrice des facteurs objets
        dict_j (dict): dictionnaire dont les clés sont les colonnes j où il ya une entrée observée et dont la valeur et un dictionnaire
        dont les clés sont les lignes i telles que (i,j) est un indice d'une entrée observée et dont la valeur et l'entrée Mij
        lamda (float): terme de régulrisation
        number_of_smaples (int): nombre d'entrées observées

    Returns:
        ndarray: matrice jacobienne de la fonction de perte par rapport à V
    """
    n,r=V.shape
    N=np.dot(U,np.transpose(V))
    dJdV=np.zeros((n,r))
    for i in range(n):
        for j in range(r):
            if i in dict_j:
                for k in dict_j[i]:
                    dJdV[i,j]-=(1/number_of_smaples)*(dict_j[i][k]-N[k,i])*U[k,j]
            dJdV[i,j]+=(1/number_of_smaples)*lamda*V[i,j]
    return dJdV

 ############################### Générer dict_i et dict_j ##########################################

def generate_dict(M,Y):
    n,m=Y.shape
    dict_i={}
    dict_j={}
    for i in range(n):
        dict_i[i]={}
        for j in range(m):
            if (i,j) in M:
                dict_i[i][j]=M[(i,j)]
    for j in range(m):
        dict_j[j]={}
        for i in range(n):
            if (i,j) in M:
                dict_j[j][i]=M[(i,j)]
    return dict_i, dict_j

########################################## Mise à jour des variables ###############################################

def gradient_descent_update(U,V,dJdU,dJdV,learning_rate):
    """Cette fonction met à jour U et V (méthode des descente de gradient)

    Args:
        U (ndarray): matrice des facteurs utilisateurs
        V (ndarray): matrice des facteurs objets
        dJdU (ndarray): matrice jacobienne de la fonction de perte par rapport à U
        dJdV (ndarray): matrice jacobienne de la fonction de perte par rapport à V
        learning_rate (float): le learning rate du descente de grandient

    Returns:
        tuple: les nouveaux U et V
    """
    return U-learning_rate*dJdU,V-learning_rate*dJdV
 ############################### Application de la descente de gradient ##########################################
def gradient_descent(U,V,M,dict_i,dict_j,gradient_U,gradient_V,error,learning_rate,lamda,number_of_smaples,max_iter):
    U1=copy.deepcopy(U)
    V1=copy.deepcopy(V)
    error_history=[error(U,V,M,lamda,number_of_smaples)]
    for i in tqdm(range(1,max_iter+1)):
        dJdU=gradient_U(U1,V1,dict_i,lamda,number_of_smaples)
        dJdV=gradient_V(U1,V1,dict_j,lamda,number_of_smaples)
        U1,V1=gradient_descent_update(U1,V1,dJdU,dJdV,learning_rate)
        err=error(U1,V1,M,lamda,number_of_smaples)  
        error_history.append(err)  #Stocker l'erreur pour voir son évolution
        if i%(max_iter/10)==0:
            print(f"iteration {i}: error = {err}") #Afficher l'erreur de fur et à mesure
    return error_history, U1, V1

####################################### Générer la prédiction #######################################################

def generate_prediction(Y,omega,learning_rate=0.03, lamda=10, max_iter=5000,rank=int(100**(1/5))):
    num_user,num_movies=Y.shape
    M=generate_M(Y,omega)
    dict_i, dict_j=generate_dict(M,Y)
    U=np.random.uniform(0,np.sqrt(5),(num_user,rank))
    V=np.random.uniform(0,np.sqrt(5),(num_movies,rank))
    error_history, U_opt, V_opt= gradient_descent(U,V,M,dict_i,dict_j,compute_gradient_U,compute_gradient_V,compute_error,learning_rate,lamda,len(omega),max_iter)
    return np.dot(U_opt,np.transpose(V_opt))

#################################### Descente du gradient sans afficher l'erreur #############################################

def gradient_descent_noprint(U,V,M,dict_i,dict_j,gradient_U,gradient_V,error,learning_rate,lamda,number_of_smaples,max_iter):
    U1=copy.deepcopy(U)
    V1=copy.deepcopy(V)
    for i in range(1,max_iter+1):
        dJdU=gradient_U(U1,V1,dict_i,lamda,number_of_smaples)
        dJdV=gradient_V(U1,V1,dict_j,lamda,number_of_smaples)
        U1,V1=gradient_descent_update(U1,V1,dJdU,dJdV,learning_rate)

    return  U1, V1
def generate_prediction_noprint(Y,omega,learning_rate=0.03, lamda=10, max_iter=5000,rank=int(100**(1/5))):
    num_user,num_movies=Y.shape
    M=generate_M(Y,omega)
    dict_i, dict_j=generate_dict(M,Y)
    U=np.random.uniform(0,np.sqrt(5),(num_user,rank))
    V=np.random.uniform(0,np.sqrt(5),(num_movies,rank))
    U_opt, V_opt= gradient_descent_noprint(U,V,M,dict_i,dict_j,compute_gradient_U,compute_gradient_V,compute_error,learning_rate,lamda,len(omega),max_iter)
    return np.dot(U_opt,np.transpose(V_opt))
