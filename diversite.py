import os
import shutil
import cv2
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
PATH_IMG_STEV = r"C:\Users\Yassin\Desktop\eye tracking\data\DomaineB\Stev22\Stev2"
PATH_CSV_STEV = r"C:\Users\Yassin\Desktop\eye tracking\data\DomaineB\Stev22\SC2_.csv"

PATH_IMG_YASS = r"C:\Users\Yassin\Desktop\eye tracking\data\DomaineB\yass\yassine"
PATH_CSV_YASS = r"C:\Users\Yassin\Desktop\eye tracking\data\DomaineB\yass\SCtes_t_.csv"

TARGET_ROOT_B = r"C:\Users\Yassin\Desktop\eye tracking\data\DomaineB\Data_domaineB"
CHECKPOINT_A = 'gaze_model.pth' 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)

# ==========================================
# 2. CHARGEMENT INTELLIGENT
# ==========================================
def load_data_smart(csv_path, img_folder):
    if not os.path.exists(csv_path) or not os.path.exists(img_folder): return pd.DataFrame()
    files_map = {}
    for f in os.listdir(img_folder):
        if f.lower().endswith(('.png', '.jpg')):
            try: files_map[int(f.split('_')[-1].split('.')[0])] = f
            except ValueError: continue
    
    try: df = pd.read_csv(csv_path)
    except: df = pd.read_excel(csv_path)

    valid_rows = []
    for idx, row in df.iterrows():
        fid = int(row['frame_id'])
        if fid in files_map:
            row['src_path'] = os.path.join(img_folder, files_map[fid])
            row['filename'] = files_map[fid]
            valid_rows.append(row)
    return pd.DataFrame(valid_rows)

print("--- 1. Analyse des fichiers sources ---")
df_full = pd.concat([load_data_smart(PATH_CSV_STEV, PATH_IMG_STEV), load_data_smart(PATH_CSV_YASS, PATH_IMG_YASS)], ignore_index=True)
print(f"Total images détectées : {len(df_full)}")

# ==========================================
# 3. CRÉATION PHYSIQUE TRAIN / TEST
# ==========================================
print("\n--- 2. Création des dossiers physiques ---")
df_train_src, df_test_src = train_test_split(df_full, test_size=0.20, random_state=42)

def create_dataset_folder(df, folder_name):
    target_dir = os.path.join(TARGET_ROOT_B, folder_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        new_paths = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copie {folder_name}"):
            dst = os.path.join(target_dir, row['filename'])
            shutil.copy(row['src_path'], dst)
            row['final_path'] = dst
            new_paths.append(row)
        return pd.DataFrame(new_paths)
    else:
        df['final_path'] = df.apply(lambda r: os.path.join(target_dir, r['filename']), axis=1)
        return df

df_train_final = create_dataset_folder(df_train_src, "train")
df_test_final = create_dataset_folder(df_test_src, "test")

# ==========================================
# 4. DATASET & MODÈLE
# ==========================================
class EyeTrackingDatasetB(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['final_path'])
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), IMG_SIZE)
        label = torch.tensor([row['x']/1272.0, abs(row['y'])/712.0], dtype=torch.float32)
        img_t = transforms.Normalize([0.5]*3, [0.5]*3)(transforms.ToTensor()(img))
        return img_t, label

class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4, 4),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4, 4)
        )
        self.regressor = nn.Sequential(
            nn.Linear(32 * 14 * 14, 256), nn.ReLU(),
            nn.Linear(256, 2), nn.Sigmoid()
        )
        
    def forward(self, x): 
        x = self.features(x).view(x.size(0), -1)
        x = F.dropout(x, p=0.2, training=self.training)
        return self.regressor(x)

def extract_features(model, pool_df):
    """Extrait l'espace latent (features) du modèle pour la diversité"""
    pool_loader = DataLoader(EyeTrackingDatasetB(pool_df), batch_size=32, shuffle=False)
    model.eval()
    features_list = []
    with torch.no_grad():
        for imgs, _ in tqdm(pool_loader, desc="Extraction Features", leave=False):
            imgs = imgs.to(DEVICE)
            feats = model.features(imgs).view(imgs.size(0), -1)
            features_list.append(feats.cpu().numpy())
    
    features_matrix = np.vstack(features_list)
    pca = PCA(n_components=min(50, len(features_matrix)), random_state=42)
    return pca.fit_transform(features_matrix)

# ==========================================
# 5. FONCTIONS DE DIVERSITÉ (LES 3 MÉTHODES)
# ==========================================

# Méthode 1: K-Means (Distance Euclidienne)
def get_kmeans_euclidean_indices(features, n_to_add):
    kmeans = KMeans(n_clusters=n_to_add, random_state=42, n_init=10)
    kmeans.fit(features)
    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features)
    return closest_indices

# Méthode 2: Similarité Cosinus (Greedy)
def get_cosine_diversity_indices(features, n_to_add):
    selected_indices = []
    first_idx = np.random.RandomState(42).randint(0, len(features))
    selected_indices.append(first_idx)
    min_distances = cosine_distances(features, features[selected_indices]).flatten()
    
    for _ in range(1, n_to_add):
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)
        new_dist = cosine_distances(features, features[next_idx].reshape(1, -1)).flatten()
        min_distances = np.minimum(min_distances, new_dist)
    return selected_indices

# Méthode 3: Gaussian Mixture Models (Probabiliste)
def get_gmm_diversity_indices(features, n_to_add):
    gmm = GaussianMixture(n_components=n_to_add, random_state=42)
    gmm.fit(features)
    # On trouve l'image la plus proche de la moyenne de chaque distribution gaussienne
    closest_indices, _ = pairwise_distances_argmin_min(gmm.means_, features)
    return closest_indices

def evaluate_model(model, loader, criterion):
    model.eval()
    t_mae = 0.0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out = model(imgs)
            t_mae += torch.mean(torch.abs(out - lbls)).item() * imgs.size(0)
    return t_mae / len(loader.dataset)

def train_model(model, loader, criterion, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()

# ==========================================
# 6. SIMULATION ACTIVE LEARNING
# ==========================================
print("\n--- 3. Lancement Simulation Active Learning (Diversité) ---")
steps = [0, 250, 500, 1000, 2000, 4000]
test_loader = DataLoader(EyeTrackingDatasetB(df_test_final), batch_size=32, shuffle=False)
criterion = nn.MSELoss()

# ------------------------------------------
# A. STRATÉGIE RANDOM (3 Runs)
# ------------------------------------------
NB_RUNS = 3
random_mae_history = {s: [] for s in steps}

print("\n>>> Évaluation RANDOM (3 répétitions)")
for run in range(NB_RUNS):
    print(f"  --> Run {run+1}/{NB_RUNS}")
    df_pool = df_train_final.copy()
    df_labeled = pd.DataFrame()
    
    for i, step in enumerate(steps):
        model = GazeNet().to(DEVICE)
        model.load_state_dict(torch.load(CHECKPOINT_A, weights_only=True))
        
        if step > 0:
            n_to_add = step - steps[i-1]
            if n_to_add > len(df_pool): n_to_add = len(df_pool)
            new_samples = df_pool.sample(n=n_to_add, random_state=42+run+i)
            df_pool = df_pool.drop(new_samples.index)
            df_labeled = pd.concat([df_labeled, new_samples])
            
            train_loader = DataLoader(EyeTrackingDatasetB(df_labeled), batch_size=32, shuffle=True)
            optimizer = optim.Adam(model.parameters(), lr=0.0005)
            train_model(model, train_loader, criterion, optimizer, epochs=3)
            
        mae = evaluate_model(model, test_loader, criterion)
        random_mae_history[step].append(mae)

random_mae_mean = [np.mean(random_mae_history[s]) for s in steps]
random_mae_std = [np.std(random_mae_history[s]) for s in steps]

# Fonction générique pour éviter de répéter le code des boucles
def run_diversity_strategy(name, get_indices_func):
    print(f"\n>>> Évaluation DIVERSITÉ : {name}")
    mae_history = []
    df_pool = df_train_final.copy()
    df_labeled = pd.DataFrame()

    for i, step in enumerate(steps):
        model = GazeNet().to(DEVICE)
        model.load_state_dict(torch.load(CHECKPOINT_A, weights_only=True))
        
        if step > 0:
            n_to_add = step - steps[i-1]
            features = extract_features(model, df_pool)
            selected_indices = get_indices_func(features, n_to_add)
            new_samples = df_pool.iloc[selected_indices]
            
            df_pool = df_pool.drop(new_samples.index)
            df_labeled = pd.concat([df_labeled, new_samples])
            
            train_loader = DataLoader(EyeTrackingDatasetB(df_labeled), batch_size=32, shuffle=True)
            optimizer = optim.Adam(model.parameters(), lr=0.0005)
            train_model(model, train_loader, criterion, optimizer, epochs=3)
            
        mae = evaluate_model(model, test_loader, criterion)
        mae_history.append(mae)
        print(f"    Palier {step} -> MAE: {mae:.4f}")
    return mae_history

# Lancement des 3 stratégies de diversité
kmeans_mae_history = run_diversity_strategy("K-Means (Euclidien)", get_kmeans_euclidean_indices)
cosine_mae_history = run_diversity_strategy("Similarité Cosinus", get_cosine_diversity_indices)
gmm_mae_history = run_diversity_strategy("Gaussian Mixture Models (GMM)", get_gmm_diversity_indices)

# ==========================================
# 7. GRAPHIQUE FINAL COMPARATIF
# ==========================================

plt.figure(figsize=(12, 7))

plt.axhline(y=random_mae_mean[0], color='r', linestyle='--', label='Baseline (Domaine A - 0 image B)')

# Tracé Random avec zone d'ombre
plt.plot(steps, random_mae_mean, marker='o', color='b', label='Random (Moyenne sur 3 runs)')
plt.fill_between(steps, 
                 np.array(random_mae_mean) - np.array(random_mae_std), 
                 np.array(random_mae_mean) + np.array(random_mae_std), 
                 color='b', alpha=0.2, label='Intervalle Confiance (Random)')

# Tracés des 3 Méthodes de Diversité
plt.plot(steps, kmeans_mae_history, marker='D', color='purple', label='Diversité (K-Means Euclidien)', linewidth=2)
plt.plot(steps, cosine_mae_history, marker='*', color='teal', label='Diversité (Similarité Cosinus)', linewidth=2)
plt.plot(steps, gmm_mae_history, marker='X', color='darkorange', label='Diversité (GMM Probabiliste)', linewidth=2)

plt.title("Active Learning : Stratégies de Diversité (Euclidien, Cosinus, GMM) vs Aléatoire", fontsize=14)
plt.xlabel("Budget d'annotation (Nombre d'images ajoutées au Train B)", fontsize=12)
plt.ylabel("Erreur Absolue Moyenne (MAE) sur le Test B", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.4)

plt.tight_layout()
plt.show()

print("\nSimulation terminée avec succès ! Le graphique avec les 3 méthodes de diversité a été généré.")