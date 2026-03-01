import os
import shutil
import cv2
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Sources (Où sont tes images actuelles)
PATH_IMG_STEV = r"C:\Users\Yassin\Desktop\eye tracking\data\DomaineB\Stev22\Stev2"
PATH_CSV_STEV = r"C:\Users\Yassin\Desktop\eye tracking\data\DomaineB\Stev22\SC2_.csv"

PATH_IMG_YASS = r"C:\Users\Yassin\Desktop\eye tracking\data\DomaineB\yass\yassine"
PATH_CSV_YASS = r"C:\Users\Yassin\Desktop\eye tracking\data\DomaineB\yass\SCtes_t_.csv"

# Destination (Où on va créer les dossiers propres)
TARGET_ROOT_B = r"C:\Users\Yassin\Desktop\eye tracking\data\DomaineB\Data_domaineB"

# Modèle Checkpoint Domaine A
CHECKPOINT_A = 'gaze_model.pth' 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)

# ==========================================
# 2. CHARGEMENT INTELLIGENT (SCAN)
# ==========================================
def load_data_smart(csv_path, img_folder):
    if not os.path.exists(csv_path) or not os.path.exists(img_folder):
        return pd.DataFrame()

    files_map = {}
    # Scan dossier
    for f in os.listdir(img_folder):
        if f.lower().endswith(('.png', '.jpg')):
            try:
                # Logique SC1_0000 -> 0 et SCtes_t_0000 -> 0
                last_part = f.split('_')[-1]
                digits = last_part.split('.')[0]
                fid = int(digits)
                files_map[fid] = f
            except ValueError: continue
    
    # Lecture CSV
    try: df = pd.read_csv(csv_path)
    except: df = pd.read_excel(csv_path)

    valid_rows = []
    for idx, row in df.iterrows():
        fid = int(row['frame_id'])
        if fid in files_map:
            # On stocke le chemin source actuel
            row['src_path'] = os.path.join(img_folder, files_map[fid])
            row['filename'] = files_map[fid]
            valid_rows.append(row)
            
    return pd.DataFrame(valid_rows)

# Chargement
print("--- 1. Analyse des fichiers sources ---")
df_stev = load_data_smart(PATH_CSV_STEV, PATH_IMG_STEV)
df_yass = load_data_smart(PATH_CSV_YASS, PATH_IMG_YASS)

if df_stev.empty and df_yass.empty:
    print("ERREUR : Aucune donnée trouvée.")
    exit()

# Fusion
df_full = pd.concat([df_stev, df_yass], ignore_index=True)
print(f"Total images détectées : {len(df_full)}")

# ==========================================
# 3. CRÉATION PHYSIQUE TRAIN / TEST
# ==========================================
print("\n--- 2. Création des dossiers physiques (Train/Test) ---")

# Split logique (20% Test, 80% Train)
df_train_src, df_test_src = train_test_split(df_full, test_size=0.20, random_state=42)

# Fonction pour copier
def create_dataset_folder(df, folder_name):
    target_dir = os.path.join(TARGET_ROOT_B, folder_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Copie des images vers {target_dir} ...")
        new_paths = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            src = row['src_path']
            dst = os.path.join(target_dir, row['filename'])
            shutil.copy(src, dst)
            row['final_path'] = dst # On met à jour le chemin vers le nouveau dossier
            new_paths.append(row)
        return pd.DataFrame(new_paths)
    else:
        print(f"Le dossier {folder_name} existe déjà. On utilise les fichiers existants.")
        # On met juste à jour les chemins
        df['final_path'] = df.apply(lambda r: os.path.join(target_dir, r['filename']), axis=1)
        return df

# Exécution de la copie
df_train_final = create_dataset_folder(df_train_src, "train")
df_test_final = create_dataset_folder(df_test_src, "test")

print(f"Train B : {len(df_train_final)} images | Test B : {len(df_test_final)} images")

# ==========================================
# 4. DATASET & MODÈLE
# ==========================================
class EyeTrackingDatasetB(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # On charge depuis le NOUVEAU dossier final
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
    def forward(self, x): return self.regressor(self.features(x).view(x.size(0), -1))

# ==========================================
# 5. SIMULATION (MAE & MSE)
# ==========================================
print("\n--- 3. Lancement Simulation (MAE & MSE) ---")

steps = [0, 250, 500, 1000, 2000, 4000] 
mae_results = []
mse_results = []

test_loader = DataLoader(EyeTrackingDatasetB(df_test_final), batch_size=32, shuffle=False)
criterion = nn.MSELoss()

for n in steps:
    # Reset Modèle A
    model = GazeNet().to(DEVICE)
    if os.path.exists(CHECKPOINT_A): model.load_state_dict(torch.load(CHECKPOINT_A))
    else: 
        print(f"Erreur: {CHECKPOINT_A} manquant."); break

    # Entraînement
    if n > 0:
        if n > len(df_train_final): n = len(df_train_final)
        subset = df_train_final.sample(n=n, random_state=42)
        train_loader = DataLoader(EyeTrackingDatasetB(subset), batch_size=32, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        
        model.train()
        for _ in range(3): # 3 époques
            for imgs, lbls in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(imgs.to(DEVICE)), lbls.to(DEVICE))
                loss.backward(); optimizer.step()

    # Test
    model.eval()
    t_mae, t_mse = 0.0, 0.0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            out = model(imgs.to(DEVICE))
            # MAE
            t_mae += torch.mean(torch.abs(out - lbls.to(DEVICE))).item() * imgs.size(0)
            # MSE
            t_mse += criterion(out, lbls.to(DEVICE)).item() * imgs.size(0)
    
    final_mae = t_mae / len(df_test_final)
    final_mse = t_mse / len(df_test_final)
    
    mae_results.append(final_mae)
    mse_results.append(final_mse)
    
    tag = "BASELINE" if n == 0 else f"Random ({n})"
    print(f"[{tag}] -> MAE: {final_mae:.4f} | MSE: {final_mse:.4f}")

# ==========================================
# 6. GRAPHIQUE FINAL
# ==========================================
plt.figure(figsize=(14, 6))

# Graphe MAE
plt.subplot(1, 2, 1)
plt.axhline(y=mae_results[0], color='r', linestyle='--', label=f'Baseline MAE')
plt.plot(steps, mae_results, marker='o', color='b', label='Active Learning (MAE)')
plt.title("Précision (MAE)")
plt.xlabel("Images ajoutées"); plt.ylabel("MAE"); plt.legend(); plt.grid(True)

# Graphe MSE
plt.subplot(1, 2, 2)
plt.axhline(y=mse_results[0], color='orange', linestyle='--', label=f'Baseline MSE')
plt.plot(steps, mse_results, marker='s', color='green', label='Active Learning (MSE)')
plt.title("Convergence (MSE)")
plt.xlabel("Images ajoutées"); plt.ylabel("MSE"); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()
print("Terminé. Dossiers créés dans :", TARGET_ROOT_B)