import numpy as np

def fusion (matrice): #tableau de vecteurs lignes
    new_image=[]
    new_image=np.mean(matrice,axis=1) #fais la moyenne des colonnes de la matrice 
    return new_image
#On fait la moyene de chaque colonnes de la matrice pour ne former qu'un seul vecteur

def mutation(vecteur, nb_variants, taux_mutation, niveau_mutation):
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

vecteur=np.array([0.4,0.3,0.9,0.8,0.2,0.1,0.4])
merge=fusion(matrice)
resultat = mutation(vecteur, nb_variants=3, taux_mutation=0.4, niveau_mutation=0.1)

print("fusion:", merge)
print(" Vecteur:",vecteur)
print("\n resultat:",resultat)
