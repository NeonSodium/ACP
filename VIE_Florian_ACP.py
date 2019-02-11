# VIE Florian

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)


print('-------')
print('Donnees')
print('-------')

data = pd.read_csv("chart_dota2_twitch.csv", sep = ";", index_col = 0)
print(data,'\n')

M = data.values


# ----------------

# Nombre de lignes
nb_ligne = np.size(M, axis=0)
print("NB Individus : ", nb_ligne)
# Nombre de colonnes
nb_col = np.size(M, axis=1)
print("NB Variables : ", nb_col)
print('\n')

# ----------------
print('-------------')
print('Vecteur moyen')
print('-------------')

g = np.average(M,axis=0)
g_frame = pd.DataFrame({'moyenne':g}).set_index(data.columns)
print(g_frame)

# ----------------

#print("\n4) Données centrées :")

Y = np.zeros((nb_ligne, nb_col))
for i in range(0, nb_ligne):
    Y[i] = M[i] - g
#print(Y)

# ----------------

#print("\n5) Matrice variance-covariance :")

Yt = np.transpose(Y)
D = np.identity(nb_ligne)/nb_ligne
V = np.dot(Yt, D)
V = np.dot(V, Y)
#print(V)

# ----------------

#print("\n6) Données centrées réduites :")

et = np.std(M,axis=0)
Z = np.zeros((nb_ligne, nb_col))
for i in range(0, nb_ligne):
    Z[i] = Y[i] / et
#print(Z)

# ----------------
print('\n------------------------')
print('Matrice des corrélations')
print('------------------------')

Ds = np.identity(nb_col)
for i in range(0, nb_col):
    Ds[i] = Ds[i] / et

R = np.dot(Ds,V)
R = np.dot(R,Ds)

R_frame = pd.DataFrame(R).set_index(data.columns)
R_frame.columns = data.columns
print(R_frame)


# ----------------

# Inertie

print('\n-------')
print('Inertie')
print('-------')

inertie = np.linalg.eigvals(R)

print('Valeurs propres : ',inertie, '\n')

incum = np.zeros(np.size(inertie))

for i in range(np.size(inertie)):
    if i == 0 :
        incum[i] = inertie[i]
    else:
        incum[i] = inertie[i] + incum[i-1]

ratio = np.zeros(np.size(inertie))
for i in range(np.size(inertie)):
    ratio[i] = incum[i] / np.size(inertie)

incum_frame = pd.DataFrame({'cum':incum})
ratio_frame = pd.DataFrame({'ratio':ratio})
inertie_frame = pd.DataFrame({'inertie':inertie})

iner = pd.concat([inertie_frame, incum_frame],axis = 1, join = 'inner')
iner = pd.concat([iner, ratio_frame],axis = 1, join = 'inner')

print(iner)


print('\n----')
print('Axes')
print('----')

l = np.zeros(2)
l[0] = sorted(inertie, reverse = True)[0]
l[1] = sorted(inertie, reverse = True)[1]

lAxe = np.zeros(2, dtype=int)
for i in range(np.size(inertie)):
    if inertie[i] == l[0]:
        lAxe[0] = i
    if inertie[i] == l[1]:
        lAxe[1] = i

print('Axes : ',lAxe[0],' et ',lAxe[1])


# ----------------

print('\n-------------------------')
print('Coordonnées des individus')
print('-------------------------')

vecPropre = np.linalg.eig(R)[1]

coordIndiv = np.dot(Z, vecPropre)
print(pd.DataFrame(coordIndiv).set_index(data.index))



print('\n---------------------------------')
print('Correlation Variables Composantes')
print('---------------------------------')

Zt = np.transpose(Z)
cvc = np.dot(Zt / nb_ligne, coordIndiv)
cvc = np.divide(cvc, np.sqrt(inertie))

print(pd.DataFrame(cvc).set_index(data.columns))


print('\n-----------------------------------')
print('Contribution des individus aux axes')
print('-----------------------------------')

cont = np.square(coordIndiv)
for i in range(nb_ligne):
    cont[i] = np.divide(cont[i], nb_ligne * inertie)
    
print(pd.DataFrame(cont).set_index(data.index))

print('')

# Sur-representation
for i in range(nb_ligne):
    if cont[i][lAxe[0]] > 0.25 :
        print(cont[i][lAxe[0]],' semble sur-represente sur axe ',lAxe[0])
    if cont[i][lAxe[1]] > 0.25 :
        print(cont[i][lAxe[1]],' semble sur-represente sur axe ',lAxe[1])

print('\n-------------------------------')
print('Carre des distances a l\'origine')
print('-------------------------------')

distance = np.sum(Z**2,axis=1)
print(pd.DataFrame(distance).set_index(data.index))

print('\n------------------------------------------------')
print('Qualite de la representation d\'un individu COS2')
print('------------------------------------------------')

cos2 = coordIndiv**2
for j in range(nb_col):
    cos2[:,j] = cos2[:,j]/distance

print(pd.DataFrame(cos2).set_index(data.index))



print('----------------------------------------------------------------')

# Premier plan principal

fig, axes = plt.subplots(figsize = (10,10))
axes.set_xlim(-6, 6)
axes.set_ylim(-6, 6)
for i in range(nb_ligne):
    plt.annotate(data.index[i],(coordIndiv[i,0],coordIndiv[i,1]))
    
plt.plot([-6,6],[0,0])
plt.plot([0,0],[-6,6])
plt.xlabel(lAxe[0])
plt.ylabel(lAxe[1])

plt.title('Premier Plan Principal')
plt.savefig('premier_plan.png')
plt.show()


# Cercle des correlations

fig, axes = plt.subplots(figsize = (10,10))
axes.set_xlim(-1, 1)
axes.set_ylim(-1, 1)

plt.plot([-1,1],[0,0])
plt.plot([0,0],[-1,1])
plt.xlabel(lAxe[0])
plt.ylabel(lAxe[1])

cercle = plt.Circle((0,0), 1, fill = False)
axes.add_artist(cercle)

for i in range(np.size(cvc[0])):
    plt.annotate(data.columns[i], (cvc[i][lAxe[0]], cvc[i][lAxe[1]]))
plt.title('Cercle des Correlations')
fig.savefig('cercle_corr.png')
plt.show()






