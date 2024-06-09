import numpy as np
import numpy.linalg as alg
import cvxpy as cp
import numpy.random as r
from utilities import generate_low_rank, generate_entries


def optimisation_convexe(A,L):
    """Cette fonction prend une matrice complète A (à prédire), une liste des entrées à cacher et retourne la matrice prédite par la méthode de relaxation convexe""" 
    n,p=A.shape
    L1=tuple(zip(*L))
    known=A[L1] #entrées révélées
    M=cp.Variable(A.shape) #variable à optimiser: matrice de même taille que A
    objectives=cp.Minimize(cp.norm(M,'nuc')) #Objectif: minimiser la norme nucléaire
    constraints=[M[L1]==known] #Sous la contrainte: M coincide avec A sur known
    prob=cp.Problem(objectives,constraints)    
    N=prob.solve()
    return M.value

def testbench(A,L):
    """Cette fonction est destinée à faire des tests et des comparaisons.
    Elle prend une matrice complète A (à prédire) et une liste des entrées à cacher.
    Elle retourne la matrice prédite par la méthode de relaxation convexe en affichant plusieurs métriques d'évaluation:
    norme nucléaire et rang de la matrice A d'origine,nombre de valeurs singulières contenant 90% de l'énergie du résultat,
    RMSE, erreur maximale et nombre d'erreurs supérieures à 1."""
    n,p=A.shape
    L1=tuple(zip(*L))
    known=A[L1]
    M=cp.Variable(A.shape)
    objectives=cp.Minimize(cp.norm(M,'nuc'))
    constraints=[M[L1]==known]
    prob=cp.Problem(objectives,constraints)    
    N=prob.solve()
    print("Norme nucléaire de la matrice d'origine:",alg.norm(A,ord='nuc'))
    print("Norme nucléaire du résultat:",N) 
    print("Rang de la matrice d'origine:", alg.matrix_rank(A))
    L=sorted(alg.svd(M.value,compute_uv=False),reverse=True)
    S=0
    i=0
    while S<0.9*N:
        S+=L[i]
        i+=1
    print("90% de l'énergie est contenue dans",i,"valeurs singulières")
    E=A-M.value
    print("RMSE:", alg.norm(E)/A.size**0.5)
    print("Erreur maximale:" ,np.max(np.abs(E)))
    print("Nombre d'erreurs supérieures à 1:",np.sum(np.abs(E)>1))
    return M.value


def test(n,p,k,m):
    """Cette fonction prend la taille (n,p) d'une matrice, un rang k et un nombre d'entrées à révéler m.
    Elle tire une matrice aléatoire de taille (n,p) et de rang k, m entrées aléatoires à révéler et affiche les métriques d'évaluation de testbench.""" 
    A=generate_low_rank(n,p,k) #générer une matrice aléatoire de taille (n,p) et de rang k
    L=generate_entries(n,p,m) #générer m entrées aléatoires à révéler
    testbench(A,L)
def test_convex(A,L):
    """Cette fonction prend une matrice complète A (à prédire), une liste des entrées à cacher
    et retourne la RMSE de la prédiction de la méthode de relaxation convexe"""
    E=A-optimisation_convexe(A,L)
    return alg.norm(E)/A.size**0.5
    
