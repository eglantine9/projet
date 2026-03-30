import numpy as np

def fusion (matrice,pond): #tableau de vecteurs lignes et vecteur de pondération dans le même ordre que les images
    """
    merge different weighted vectors together

    Parameters
    ----------
    matrice : numpy.array
        matrice of all the vectors, number of row correspond to the number of vectors to fuse
    pond : numpy.array or list
        vector containing to the weights of each vector in the same order as the matrice

    Returns
    -------
    new_image : numpy.array
        New vector computed by the weighted merge

    Examples
    --------
    >>> fusion(np.array([[0.5,0.8,0.2,0.6,0.7],[0.4,0.7,0.1,0.5,0.8]]), (1,2))
    [0.43333333 0.73333333 0.13333333 0.53333333 0.76666667]
    """

    new_image=[]
    div = sum(pond)
    for col in range(0,np.shape(matrice)[1]): #Parcour toutes les colonnes du tableau
        for vec in range(0,np.shape(matrice)[0]):
            matrice[vec,col] = matrice[vec,col]*pond[vec]/div

    new_image=np.sum(matrice,axis=0) #fais la somme des colonnes de la matrice (déjà divisées par div)
    return new_image
#On fait la moyene de chaque colonnes de la matrice pour ne former qu'un seul vecteur



def mutation(vecteur, nb_variants, taux_mutation, niveau_mutation):
    """
    produce multuple variations from an initial vector

    Parameters
    ----------
    vecteur : numpy.array
        Initial vector to produce variations from

    nb_variants : int
        Number of variations to produce

    taux_mutation : int or float
        Amount of mutations to apply

    niveau_mutation : int or float
        Intensity of mutations to apply

    Returns
    -------
    np.array(vecteurs_mutes)
        vectors produced by the random mutations

    Examples
    --------
    >>> mutation(np.array([0.4,0.3,0.9,0.8,0.2,0.1,0.4]), nb_variants=3, taux_mutation=0.4, niveau_mutation=0.1)
    [[0.42754856 0.26299389 0.9        0.8        0.2        0.1       0.3276892 ]
    [0.41722165 0.3460928  0.85578323 0.8        0.19935116 0.19189771     0.4       ]
    [0.4        0.3        0.9507284  0.8        0.2        0.15455378     0.27444405]]
    """

    vecteurs_mutes=[]
    for i in range (nb_variants):
        vect_mutation=vecteur.copy()
        mute=np.random.rand(len(vecteur)) #variable aléatoire entre 0 et 1, elles representes les valeurs qui seront mutés si < taux de mutation
        bruit=np.random.normal(0,niveau_mutation, len(vecteur))#petite variable representant la mutation entre 0 et niveau de mutation
        for j in range(len(vecteur)):
            if mute[j]<taux_mutation:
                vect_mutation[j]+=bruit[j] #on ajoute la mutation a la valeur de l'indice où survient une mutation
            else:
                j=j+1
        vecteurs_mutes.append(vect_mutation)#On crée une liste de vecteur mutés
    return np.array(vecteurs_mutes)
#Mute est une liste de meme taille que le vecteur, les valeurs de chacunes de ses cases seront générés aléatoirement et comprises entre 0 et 1
#bruit est un vecteur ou toute les valeurs sont des valeurs de mutations( proche de 0 car suit une loi normale)
#Pour les valeurs de Mute qui seront inférieur au taux de mutation, on garde l'indice, cet indice sera celui qui dans le vect_mutation permettra de muté une valeur précise avec l'ajout du bruit[i]


#test
matrice=np.array([[0.5,0.8,0.2,0.6,0.7],
                 [0.4,0.7,0.1,0.5,0.8]])
print("La matrice est de type :", type(matrice))

vecteur=np.array([0.4,0.3,0.9,0.8,0.2,0.1,0.4])
print("Le vecteur est de type :", type(vecteur))

merge=fusion(matrice,(1,2))
resultat = mutation(vecteur, nb_variants=3, taux_mutation=0.4, niveau_mutation=0.1)

print("fusion:", merge)
print(" Vecteur:",vecteur)
print("\n resultat:",resultat)
