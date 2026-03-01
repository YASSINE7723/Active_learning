#  Eye Tracking & Active Learning 

 **Auteur :** Zerkani Yassine (ENSIM, Le Mans Université)

##  Description du Projet
L'estimation du regard par apprentissage profond (Deep Learning) nécessite généralement d'énormes quantités de données annotées. Ce projet vise à développer un modèle de prédiction du regard (coordonnées X, Y sur un écran) via une simple webcam, et à étudier sa capacité de généralisation. 

L'étude s'articule autour de deux environnements :
* **Domaine A (DA) :** Environnement strictement contrôlé (sujet unique, lumière fixe). Il sert à entraîner notre modèle de référence (**GazeNet**).
* **Domaine B (DB) :** Environnement fortement hétérogène (nouveaux sujets, lunettes, changements de caméra et de luminosité). Il sert à tester la chute de performance liée au **Domain Shift**.

Pour adapter le modèle au Domaine B avec un minimum d'annotations manuelles, nous implémentons et comparons plusieurs stratégies d'**Apprentissage Actif (Active Learning)** : Aléatoire, Incertitude (MC Dropout, Contraste), Diversité (K-Means, GMM, Cosinus) et une méthode Mixte.

---

##  Structure du Répertoire et Fichiers

Voici l'explication des scripts et notebooks présents dans ce dépôt :

### 🛠️ 1. Préparation et Entraînement Initial (Domaine A)
* **`datacollection.py`** : Script permettant de collecter les images via la webcam et de générer le fichier CSV contenant les vérités terrain (coordonnées X, Y).
* **`training1.py`** : Script d'entraînement du modèle de base (GazeNet) sur les données du Domaine A. Génère le checkpoint de référence (ex: `gaze_model.pth`).
* **`n.py`** : Script de test crucial. Il charge le modèle entraîné sur le Domaine A et l'évalue **directement sur le Domaine B sans réentraînement**. Il permet de quantifier la perte de performance brute liée au *Domain Shift*.

###  2. Stratégies d'Apprentissage Actif (Active Learning sur le Domaine B)
* **`Random_vs_Baseline.ipynb`** : Notebook établissant la baseline de l'Active Learning en évaluant une stratégie de sélection purement aléatoire (Moyenne sur plusieurs itérations avec intervalle de confiance).
* **`Incertitude.py`** : Script implémentant l'échantillonnage basé sur l'incertitude du modèle (*MC Dropout* et *Perturbation de Contraste*).
* **`diversite.py`** : Script implémentant l'échantillonnage basé sur la géométrie de l'espace latent pour maximiser la couverture des données (*K-Means Euclidien*, *Similarité Cosinus*, *Gaussian Mixture Models - GMM*).
* **`mixte.py`** : Script implémentant la **stratégie combinée (Pré-filtrage)**. Il présélectionne les images les plus incertaines (MC Dropout) puis force la diversité (K-Means) sur cet échantillon pour obtenir les meilleures performances.

###  3. Exploration et Notebooks
* **`Lets_start.ipynb`** : Notebook d'exploration initiale, de visualisation des données ou de test d'architecture.
* **`Active_Learning_Domaine_B.ipynb`** : Notebook principal synthétisant les boucles d'Active Learning et permettant de visualiser les graphiques comparatifs finaux de l'Erreur Absolue Moyenne (MAE).

###  4. Configuration
* **`requirements.txt`** : Liste des dépendances Python nécessaires (PyTorch, OpenCV, Scikit-learn, Pandas, etc.).
* **`.gitignore`** : Fichiers et dossiers à ignorer par Git (datasets locaux, checkpoints lourds, etc.).

---

##  Installation et Exécution

**1. Cloner le dépôt et installer les dépendances :**
```bash
git clone [https://github.com/VOTRE_NOM/VOTRE_DEPOT.git](https://github.com/VOTRE_NOM/VOTRE_DEPOT.git)
cd VOTRE_DEPOT
pip install -r requirements.txt
