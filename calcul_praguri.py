import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import glob
import customtkinter as ctk
from tkinter import filedialog

from autoencoder import Autoencoder
from patchcore import PatchCoreDetector
from padim import PaDiMDetector

class UtilitarPraguri(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Generator Raport Scoruri AI - Licenta")
        self.geometry("600x600")
        ctk.set_appearance_mode("dark")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.label = ctk.CTkLabel(self, text="Calculare Statistici si Praguri", font=("Roboto", 22, "bold"))
        self.label.pack(pady=25)

        self.lbl_alg = ctk.CTkLabel(self, text="1. Selecteaza Algoritmul:")
        self.lbl_alg.pack()
        self.combo_alg = ctk.CTkComboBox(self, values=["Autoencoder", "PaDiM", "PatchCore"], width=200)
        self.combo_alg.pack(pady=10)

        self.lbl_obj = ctk.CTkLabel(self, text="2. Selecteaza Obiectul:")
        self.lbl_obj.pack()
        self.combo_obj = ctk.CTkComboBox(self, values=["cablu", "tranzistor", "surub", "piulita", "pcb1", "pcb2", "pcb3", "pcb4"], width=200)
        self.combo_obj.pack(pady=10)

        self.btn_folder = ctk.CTkButton(self, text="3. Alege Folder Imagini", command=self.alege_folder, fg_color="#1a73e8")
        self.btn_folder.pack(pady=20)

        self.path_label = ctk.CTkLabel(self, text="Niciun folder selectat", font=("Arial", 11), text_color="gray")
        self.path_label.pack()

        self.btn_start = ctk.CTkButton(self, text="GENEREAZA RAPORT .TXT", fg_color="#28a745", hover_color="#218838",
                                       font=("Arial", 14, "bold"), height=40, command=self.proceseaza)
        self.btn_start.pack(pady=30)

        self.status = ctk.CTkLabel(self, text="Status: Pregatit", font=("Arial", 12))
        self.status.pack(side="bottom", pady=20)

        self.folder_selectat = ""

    def alege_folder(self):
        self.folder_selectat = filedialog.askdirectory()
        if self.folder_selectat:
            nume_scurt = "..." + self.folder_selectat[-40:] if len(self.folder_selectat) > 40 else self.folder_selectat
            self.path_label.configure(text=f"Folder: {nume_scurt}", text_color="white")

    def incarca_stare_model(self, model, cale_fisier):
        if not os.path.exists(cale_fisier):
            raise FileNotFoundError(f"Nu exista fisierul: {cale_fisier}")

        checkpoint = torch.load(cale_fisier, map_location=self.device, weights_only=True)
        nume_noi = {}
        for cheie, valoare in checkpoint.items():
            noua_cheie = cheie
            if cheie.startswith("codificator_conv"):    noua_cheie = cheie.replace("codificator_conv", "encoder.0")
            elif cheie.startswith("aplatizare"):         noua_cheie = cheie.replace("aplatizare", "encoder.1")
            elif cheie.startswith("codificator_liniar"): noua_cheie = cheie.replace("codificator_liniar", "encoder.2")
            elif cheie.startswith("decodificator_liniar"): noua_cheie = cheie.replace("decodificator_liniar", "decoder_linear")
            elif cheie.startswith("decodificator_conv"):   noua_cheie = cheie.replace("decodificator_conv", "decoder_conv")
            nume_noi[noua_cheie] = valoare

        model.load_state_dict(nume_noi, strict=False)
        return model

    def proceseaza(self):
        if not self.folder_selectat:
            self.status.configure(text="EROARE: Selecteaza un folder!", text_color="#ff4d4d")
            return

        alg = self.combo_alg.get()
        obj = self.combo_obj.get()
        folder_basename = os.path.basename(self.folder_selectat)

        try:
            self.status.configure(text=f"Se incarca modelul {alg}...", text_color="yellow")
            self.update()

            # 1. Initializare model
            if alg == "Autoencoder":
                model = Autoencoder(256, 256, 3, (32, 64, 128, 256), 1024)
                model = self.incarca_stare_model(model, f"model_{obj}_autoencoder.pth")
                model.to(self.device).eval()

            elif alg == "PatchCore":
                detector = PatchCoreDetector(device=self.device)
                detector.load(f"model_{obj}_patchcore.pkl")

            elif alg == "PaDiM":
                detector = PaDiMDetector(device=self.device)
                detector.load(f"model_{obj}_padim.pkl")

            # 2. Colectare imagini
            imagini = glob.glob(os.path.join(self.folder_selectat, "*.*"))
            imagini = [f for f in imagini if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not imagini:
                self.status.configure(text="EROARE: Nu sunt imagini in folder!", text_color="red")
                return

            scouri = []

            if alg in ["PatchCore", "PaDiM"]:
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor()
                ])

            self.status.configure(text=f"Se proceseaza {len(imagini)} imagini...", text_color="cyan")
            self.update()

            for path in imagini:
                img_pil = Image.open(path).convert("RGB")
                img_tensor = transform(img_pil).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    if alg == "Autoencoder":
                        output = model(img_tensor)
                        scor = torch.mean((img_tensor - output) ** 2).item()
                    elif alg == "PatchCore":
                        scor, _ = detector.predict(img_tensor)
                    elif alg == "PaDiM":
                        scor, _ = detector.predict(img_tensor)

                scouri.append((os.path.basename(path), scor))

            # 3. Calcul statistici si salvare raport complet (Statistici + Lista)
            if not os.path.exists("praguri"):
                os.makedirs("praguri")

            nume_fisier = f"praguri/Raport_{obj}_{alg}_{folder_basename}.txt"
            doar_scoruri = [s[1] for s in scouri]

            with open(nume_fisier, "w") as f:
                f.write(f"RAPORT STATISTICI SI SCORURI\n")
                f.write(f"Obiect: {obj.upper()} | Algoritm: {alg}\n")
                f.write(f"Folder procesat: {self.folder_selectat}\n")
                f.write("-" * 60 + "\n")
                f.write(f"NR. IMAGINI: {len(doar_scoruri)}\n")
                f.write(f"MIN SCOR:    {np.min(doar_scoruri):.10f}\n")
                f.write(f"MAX SCOR:    {np.max(doar_scoruri):.10f}\n")
                f.write(f"MEDIE SCOR:  {np.mean(doar_scoruri):.10f}\n")
                f.write("-" * 60 + "\n")
                f.write("LISTA SCORURI INDIVIDUALE:\n")
                for nume_img, scor in scouri:
                    f.write(f"{nume_img} -> {scor:.10f}\n")
                f.write("-" * 60 + "\n")

            self.status.configure(text=f"Raport salvat in 'praguri'", text_color="#48bb78")

        except Exception as e:
            self.status.configure(text=f"EROARE: Verificati consola", text_color="#ff4d4d")
            print(f"Detalii eroare: {e}")

if __name__ == "__main__":
    app = UtilitarPraguri()
    app.mainloop()