import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import pickle

# Importuri modele proprii
from autoencoder import Autoencoder
from patchcore import PatchCoreDetector
from padim import PaDiMDetector 

constante_obiecte = {
    "tranzistor": {},
    "cablu": {},
    "surub": {},
    "piulita": {},
    "pcb1": {},
    "pcb2": {},
    "pcb3": {},
    "pcb4": {}
}

img_size = 256
batch_size = 32
num_epochs = 200

def prelucrare_imagini(nume_obiect, tip_algoritm="ae"):
    if nume_obiect not in constante_obiecte:
        raise ValueError(f"Eroare: Obiectul '{nume_obiect}' nu este in dictionar!")

    if tip_algoritm.lower() in ["patchcore", "padim"]:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

class SetDate(Dataset):
    def __init__(self, cale_fisier, transform=None):
        self.cale_fisier = cale_fisier
        self.transform = transform
        extensii_valide = ('.png', '.jpg', '.jpeg')
        if not os.path.exists(cale_fisier):
            self.cai_imagini = []
        else:
            self.cai_imagini = [
                os.path.join(cale_fisier, f)
                for f in os.listdir(cale_fisier)
                if f.lower().endswith(extensii_valide)
            ]

    def __len__(self):
        return len(self.cai_imagini)

    def __getitem__(self, index):
        cale_poza = self.cai_imagini[index]
        imagine = Image.open(cale_poza).convert('RGB')
        if self.transform:
            imagine = self.transform(imagine)
        return imagine

def antreneaza_padim(obiect, device):
    print(f"\n--- Începe antrenarea PaDiM pentru: {obiect} ---")
    cale_train = rf"E:\Facultate\code\Licenta\Aplicatie\dataset\{obiect}\train\good"
    
    if not os.path.exists(cale_train):
        print(f"EROARE: Folderul nu exista: {cale_train}")
        return

    transformari = prelucrare_imagini(obiect, tip_algoritm="padim")
    dataset = SetDate(cale_train, transform=transformari)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    detector = PaDiMDetector(device=device)
    detector.train(loader)  

    cale_salvare = f"model_{obiect}_padim.pkl"
    detector.save(cale_salvare)
    print(f"✓ Model PaDiM salvat: {cale_salvare}")

def antreneaza_patchcore(obiect, device):
    print(f"\n--- Începe antrenarea PatchCore pentru: {obiect} ---")
    cale_train = rf"E:\Facultate\code\Licenta\Aplicatie\dataset\{obiect}\train\good"

    if not os.path.exists(cale_train):
        print(f"EROARE: Folderul nu exista: {cale_train}")
        return

    transformari = prelucrare_imagini(obiect, tip_algoritm="patchcore")
    dataset = SetDate(cale_train, transform=transformari)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    detector = PatchCoreDetector(device=device)
    detector.train_model(loader)  

    cale_salvare = f"model_{obiect}_patchcore.pkl"
    detector.save(cale_salvare)
    print(f"✓ Model PatchCore salvat: {cale_salvare}")

def ruleaza_antrenare(tip_algoritm="autoencoder"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lista_obiecte = ["tranzistor","cablu", "surub", "piulita", "pcb1", "pcb2", "pcb3", "pcb4"]

    for obiect in lista_obiecte:
        if tip_algoritm.lower() == "padim":
            antreneaza_padim(obiect, device)
            continue

        if tip_algoritm.lower() == "patchcore":
            antreneaza_patchcore(obiect, device)
            continue

        # Logica pentru Autoencoder
        print(f"\nIncepe antrenarea pentru: {obiect} ({tip_algoritm})")
        cale_train = rf"E:\Facultate\code\Licenta\Aplicatie\dataset\{obiect}\train\good"
        if not os.path.exists(cale_train):
            print(f"Folderul nu exista: {cale_train}")
            continue

        transformari = prelucrare_imagini(obiect, tip_algoritm=tip_algoritm)
        dataset = SetDate(cale_train, transform=transformari)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = Autoencoder(width=img_size, height=img_size, depth=3, filtre=(32, 64, 128, 256), DimLatenta=1024).to(device)

        nume_model_pth = f"model_{obiect}_{tip_algoritm.lower()}.pth"
        if os.path.exists(nume_model_pth):
            print(f"Se incarca modelul existent: {nume_model_pth}")
            model.load_state_dict(torch.load(nume_model_pth, map_location=device))

        optimizare = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizare, 'min', patience=7, factor=0.1)
        model.train()

        for epoch in range(num_epochs):
            loss_total = 0
            for batch in loader:
                batch = batch.to(device)
                optimizare.zero_grad()

                reconstructie = model(batch)
                loss = nn.functional.mse_loss(reconstructie, batch)

                loss.backward()
                optimizare.step()
                loss_total += loss.item()

            loss_mediu = loss_total / len(loader)
            scheduler.step(loss_mediu)

            if (epoch + 1) % 10 == 0:
                print(f"Epoca [{epoch+1}/{num_epochs}] | Loss: {loss_mediu:.6f}")

        torch.save(model.state_dict(), nume_model_pth)
        print(f"✓ Model Autoencoder salvat: {nume_model_pth}")

if __name__ == "__main__":
    algoritm_tinta = "padim" 
    ruleaza_antrenare(tip_algoritm=algoritm_tinta)