import numpy as np
import numpy.random as r
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
