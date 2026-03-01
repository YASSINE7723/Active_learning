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
from tqdm import tqdm
from sklearn.utils import shuffle

# ==========================================
# 1. CONFIGURATION ET QUOTAS
# ==========================================
CSV_PATH = r"C:\Users\Yassin\Desktop\eye tracking\data\DomaineA\coords_20260121_082431.csv"
DATASET_ROOT = r"C:\Users\Yassin\Desktop\eye tracking\data\DomaineA\Data_domaineA"

# Quotas demandés par le projet
N_TRAIN = 1248
N_DEV = 268
N_TEST = 268

# ==========================================
# 2. CRÉATION PHYSIQUE DES 3 DOSSIERS
# ==========================================
def create_physical_split():
    print("--- Étape 1 : Création des dossiers et tri des images ---")
    df = pd.read_csv(CSV_PATH)
    
    # Identification des images présentes à la racine
    valid_data = []
    for idx, row in df.iterrows():
        img_name = f"frame_{int(row['frame_id']):04d}.png"
        img_path = os.path.join(DATASET_ROOT, img_name)
        if os.path.exists(img_path):
            valid_data.append(row)
    
    df_exists = pd.DataFrame(valid_data)
    df_exists = shuffle(df_exists, random_state=42)
    
    splits = {
        'train': df_exists.iloc[:N_TRAIN],
        'dev': df_exists.iloc[N_TRAIN : N_TRAIN + N_DEV],
        'test': df_exists.iloc[N_TRAIN + N_DEV : N_TRAIN + N_DEV + N_TEST]
    }

    for name, data in splits.items():
        target_dir = os.path.join(DATASET_ROOT, name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print(f"Copie des images vers le dossier : {name}")
            for _, row in tqdm(data.iterrows(), total=len(data), desc=name):
                img_name = f"frame_{int(row['frame_id']):04d}.png"
                shutil.copy(os.path.join(DATASET_ROOT, img_name), os.path.join(target_dir, img_name))
        else:
            print(f"Le dossier '{name}' est déjà prêt.")

create_physical_split()

# ==========================================
# 3. DATASET ET ARCHITECTURE GAZENET
# ==========================================
class EyeTrackingDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.df_all = pd.read_csv(csv_file)
        self.img_dir = img_dir
        files = os.listdir(img_dir)
        present_ids = [int("".join(filter(str.isdigit, f))) for f in files if f.endswith('.png')]
        self.df = self.df_all[self.df_all['frame_id'].isin(present_ids)].reset_index(drop=True)
        self.coords = self.df[['x','y']].values.astype(np.float32)
        # Normalisation 1272x712
        self.coords[:, 0] /= 1272.0
        self.coords[:, 1] /= 712.0

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        fid = int(self.df.iloc[idx]['frame_id'])
        img = cv2.imread(os.path.join(self.img_dir, f"frame_{fid:04d}.png"))
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))
        label = torch.tensor([self.coords[idx][0], abs(self.coords[idx][1])], dtype=torch.float32)
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
# 4. ENTRAÎNEMENT, VALIDATION ET TEST
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(EyeTrackingDataset(CSV_PATH, os.path.join(DATASET_ROOT, "train")), batch_size=32, shuffle=True)
dev_loader = DataLoader(EyeTrackingDataset(CSV_PATH, os.path.join(DATASET_ROOT, "dev")), batch_size=32)
test_loader = DataLoader(EyeTrackingDataset(CSV_PATH, os.path.join(DATASET_ROOT, "test")), batch_size=32)

model = GazeNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Suivi des métriques pour les 3 courbes
history = {'train_loss': [], 'val_loss': [], 'test_loss': [],
           'train_mae': [], 'val_mae': [], 'test_mae': []}

best_val_loss = float('inf')
patience, trigger = 3, 0

print(f"\n--- Étape 2 : Entraînement sur {device} ---")

for epoch in range(25):
    # Phase d'entraînement
    model.train()
    t_loss, t_mae = 0, 0
    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward(); optimizer.step()
        t_loss += loss.item() * imgs.size(0)
        t_mae += torch.mean(torch.abs(outputs - lbls)).item() * imgs.size(0)
    
    # Phase d'évaluation (Val & Test)
    model.eval()
    v_loss, v_mae, te_loss, te_mae = 0, 0, 0, 0
    with torch.no_grad():
        for imgs, lbls in dev_loader:
            out = model(imgs.to(device))
            v_loss += criterion(out, lbls.to(device)).item() * imgs.size(0)
            v_mae += torch.mean(torch.abs(out - lbls.to(device))).item() * imgs.size(0)
        for imgs, lbls in test_loader:
            out = model(imgs.to(device))
            te_loss += criterion(out, lbls.to(device)).item() * imgs.size(0)
            te_mae += torch.mean(torch.abs(out - lbls.to(device))).item() * imgs.size(0)

    # Stockage des moyennes
    history['train_loss'].append(t_loss/len(train_loader.dataset))
    history['val_loss'].append(v_loss/len(dev_loader.dataset))
    history['test_loss'].append(te_loss/len(test_loader.dataset))
    history['train_mae'].append(t_mae/len(train_loader.dataset))
    history['val_mae'].append(v_mae/len(dev_loader.dataset))
    history['test_mae'].append(te_mae/len(test_loader.dataset))

    print(f"Loss -> Train: {history['train_loss'][-1]:.4f} | Val: {history['val_loss'][-1]:.4f} | Test: {history['test_loss'][-1]:.4f}")

    # Logique d'Early Stopping sur Val Loss
    if history['val_loss'][-1] < best_val_loss:
        best_val_loss = history['val_loss'][-1]
        torch.save(model.state_dict(), 'gaze_model.pth')
        trigger = 0
    else:
        trigger += 1
        if trigger >= patience:
            print("\n[STOP] Divergence détectée.")
            break

# ==========================================
# 5. AFFICHAGE DES 3 GRAPHIQUES
# ==========================================
plt.figure(figsize=(18, 5))

# Graphique 1 : MSE Loss (Train, Val, Test)
plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.plot(history['test_loss'], label='Test Loss', linestyle='--')
plt.title("Évolution de la Loss (MSE)")
plt.xlabel("Époques"); plt.legend()

# Graphique 2 : MAE (Train, Val, Test)
plt.subplot(1, 3, 2)
plt.plot(history['train_mae'], label='Train MAE')
plt.plot(history['val_mae'], label='Val MAE')
plt.plot(history['test_mae'], label='Test MAE', linestyle='--')
plt.title("Évolution de l'Erreur (MAE)")
plt.xlabel("Époques"); plt.legend()

# Graphique 3 : Scatter Plot Final sur le Test Set
model.load_state_dict(torch.load('gaze_model.pth'))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        out = model(imgs.to(device))
        all_preds.append(out.cpu().numpy())
        all_labels.append(lbls.numpy())
all_preds, all_labels = np.vstack(all_preds), np.vstack(all_labels)

plt.subplot(1, 3, 3)
plt.scatter(all_labels[:100, 0], all_labels[:100, 1], c='blue', label='Vrai', alpha=0.5)
plt.scatter(all_preds[:100, 0], all_preds[:100, 1], c='red', label='Prédit', alpha=0.5)
plt.title("Test Set : Vrai vs Prédit (100 pts)")
plt.legend()

plt.tight_layout()
plt.show()

print(f"\nFINI. Meilleur MAE Test : {min(history['test_mae']):.4f}")
print("Checkpoint sauvegardé : gaze_model.pth")