import numpy as np
import pandas as pd
from numpy.linalg import norm
import random
import matplotlib.pyplot as plt

#calculer le gradient de la fonciton de perte J par rapport aux entrés de U
def compute_gradient_U(U,V,dict_i,n_samples,lamda):
    m,r=U.shape
    N=np.dot(U,np.transpose(V))
    dJdU=np.zeros((m,r), dtype = np.float64)
    for i in range(m):
        for j in range(r):
            if i in dict_i:  
                for k in dict_i[i]:
                        dJdU[i,j]-=(1/n_samples)*((dict_i[i][k]-N[i,k])*2*V[k,j] - 2*lamda*U[i,j])

                     
    return dJdU
#calculer le gradient de la fonciton de perte J par rapport aux entrés de V
def compute_gradient_V(U,V,dict_j,n_samples,lamda):
    n,r=V.shape

    N=np.dot(U,np.transpose(V))
    dJdV=np.zeros((n,r))
    for i in range(n):
        for j in range(r):
            if i in dict_j:
                for k in dict_j[i]:
                    dJdV[i,j]-=(1/n_samples)*((dict_j[i][k]-N[k,i])*2*U[k,j] - 2*lamda*V[i,j])
    return dJdV

#fonction pour minimiser la perte en agissant uniquement sur U
def minimize_U(U,V, dict_i,max_iter,learning_rate,n_samples,lamda):
    while(True):
        deltaU = learning_rate*n_samples*compute_gradient_U(U,V,dict_i,n_samples,lamda)
        if np.linalg.norm(deltaU)/np.linalg.norm(U)< 1e-4:
            break   
        U = U - deltaU
        
    return U

#fonction pour minimiser la perte en agissant uniquement sur V
def minimize_V(U,V,dict_j,max_iter,learning_rate,n_samples,lamda):  
    while(True):
        deltaV=learning_rate*n_samples*compute_gradient_V(U,V,dict_j,n_samples,lamda)
        if np.linalg.norm(deltaV)/np.linalg.norm(V)< 0.5e-4:
            break
        V = V - deltaV
    return V

#minimsation alternée , T nombre d'itérations 
def AltMin(U0,V0,T,dict_i,dict_j, n_samples,lamda):
    learning_rate = 1e-3
    max_iter = 500
    
    U = U0
    V=V0
    for t in range(T):
        V = minimize_V(U,V, dict_j,max_iter,learning_rate, n_samples,lamda)
        U = minimize_U(U,V, dict_i,max_iter,learning_rate , n_samples,lamda)        
    return (U,V)

#initialiser U et V et retourner le produit final U*tranpose(V)
def AltMinComplete(omega,Y,m,n,k,lamda,T):
    
    P_omega_Y = np.zeros((m,n))
    for x in omega: # projection de Y sur omega
        P_omega_Y[x]=Y[x]
    #initialisation:
    U0,V0 = np.linalg.svd(P_omega_Y)[0][:,:k] , np.transpose(np.linalg.svd(P_omega_Y)[2][:k,:])

   
    dict_i , dict_j = generate_dict(Y,m,n)
    n_samples = len(omega)
    U,V = AltMin(U0,V0, T, dict_i, dict_j,n_samples,lamda) 
    return np.dot(U,np.transpose(V)) 

#dict_j (dict): dictionnaire dont les clés sont les colonnes j où il ya une entrée observée et dont la valeur et un dictionnaire
#dont les clés sont les lignes i telles que (i,j) est un indice d'une entrée observée et dont la valeur est l'entrée Yij
def generate_dict(Y,m,n):
    
    dict_i={}
    dict_j={}
    for i in range(m):
        dict_i[i]={}
        for j in range(n):
            if (i,j) in Y:
                dict_i[i][j]=Y[(i,j)]
    for j in range(n):
        dict_j[j]={}
        for i in range(m):
            if (i,j) in Y:
                dict_j[j][i]=Y[(i,j)]
    return dict_i, dict_j



#fonction finale : transforme les donées pour q'uils soient entre -1 et 1 avant de faire la minimisation alternée . 
def predict_AltMin(Y,m,n,lamda=0.05,T=10,k=5):

    omega = list(Y.keys())
    max_value = max(np.abs(list(Y.values())))
    Y_new = {cle: valeur / max_value for cle, valeur in Y.items()}
    return max_value * AltMinComplete(omega,Y_new,m,n,k,lamda)




    
    
    

 
