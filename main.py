import numpy as np
from sklearn.linear_model import LogisticRegression

# Données d'entraînement fictives, simplifiées à 2 variables : [forme_physique, victoires_5_derniers]
X_train = np.array([
    [8, 4],
    [6, 3],
    [5, 2],
    [9, 5],
    [4, 1],
    [7, 4],
    [3, 0],
    [10, 5],
    [6, 2],
    [2, 0],
    [9, 4],
    [1, 0],
    [8, 5],
    [3, 1],
    [7, 3]
])

# 1 = victoire, 0 = défaite (cible simulée)
y_train = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])

# Entraînement du modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# Fonction pour demander les données d'un joueur
def demander_joueur(nom):
    print(f"\n🔷 {nom}")
    forme = float(input("  Forme physique (0 à 10) : "))
    victoires = int(input("  Victoires récentes (sur 5 matchs) : "))
    return np.array([[forme, victoires]])

# Interaction utilisateur
try:
    joueur_A = demander_joueur("Joueur A")
    joueur_B = demander_joueur("Joueur B")

    # Probabilité brute selon IA
    pA = model.predict_proba(joueur_A)[0][1]
    pB = model.predict_proba(joueur_B)[0][1]

    # Normalisation : chance relative de A contre B
    proba_A = (pA / (pA + pB)) * 100
    proba_B = 100 - proba_A

    print(f"\n📊 Résultat IA :")
    print(f"  Joueur A : {proba_A:.2f}% de chances de gagner")
    print(f"  Joueur B : {proba_B:.2f}% de chances de gagner")

except ValueError:
    print("❌ Erreur : merci de saisir des nombres valides.")