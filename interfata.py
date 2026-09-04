import customtkinter as ctk
from PIL import Image, ImageTk, ImageOps
import torch
from torchvision import transforms
import os
import numpy as np
import shutil
from autoencoder import Autoencoder
from padim import PaDiMDetector 
from patchcore import PatchCoreDetector

config_obiecte = {
    "cablu": {
        "algoritm_optim": "PaDiM",
        "praguri": {"Autoencoder": 0.009, "PaDiM": 38.5, "PatchCore": 2.0}
    },
    "tranzistor": {
        "algoritm_optim": "PaDiM",
        "praguri": {"Autoencoder": 0.0020, "PaDiM": 46.0, "PatchCore": 0.5}
    },
    "surub": {
        "algoritm_optim": "PaDiM",
        "praguri": {"Autoencoder": 0.001, "PaDiM": 32.0, "PatchCore": 2.0}
    },
    "piulita": {
        "algoritm_optim": "PaDiM",
        "praguri": {"Autoencoder": 0.003, "PaDiM": 33.5, "PatchCore": 1.7}
    },
    "pcb1": {
        "algoritm_optim": "PatchCore",
        "praguri": {"Autoencoder": 0.0028, "PaDiM": 25.0, "PatchCore": 1.0}
    },
    "pcb2": {
        "algoritm_optim": "Autoencoder",
        "praguri": {"Autoencoder": 0.0025, "PaDiM": 35.0, "PatchCore": 1.0}
    },
    "pcb3": {
        "algoritm_optim": "PatchCore",
        "praguri": {"Autoencoder": 0.0025, "PaDiM": 30.0, "PatchCore": 1.0}
    },
    "pcb4": {
        "algoritm_optim": "PaDiM",
        "praguri": {"Autoencoder": 0.0026, "PaDiM": 26.0, "PatchCore": 2.5}
    }
}

CULORI_ALGORITM = {
    "Autoencoder": "#e67e22",
    "PaDiM":       "#8e44ad",
    "PatchCore":   "#1a73e8",
}

class SistemInspectieAI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Software Inspecție Calitate - Analiză AI")
        self.geometry("1200x780")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cale_imagine = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.lbl_titlu = ctk.CTkLabel(self.sidebar, text="PANOU DE CONTROL",
                                      font=("Roboto", 20, "bold"))
        self.lbl_titlu.pack(pady=30, padx=20)

        self.lbl_obj = ctk.CTkLabel(self.sidebar, text="Tip Componentă:")
        self.lbl_obj.pack(pady=(10, 0))
        self.combo_obiect = ctk.CTkComboBox(
            self.sidebar,
            values=list(config_obiecte.keys()),
            command=self._on_obiect_schimbat
        )
        self.combo_obiect.pack(pady=10, padx=20)

        self.lbl_alg_titlu = ctk.CTkLabel(self.sidebar, text="Algoritm selectat automat:",
                                            font=("Arial", 11))
        self.lbl_alg_titlu.pack(pady=(18, 0))

        self.lbl_algoritm_badge = ctk.CTkLabel(
            self.sidebar, text="—",
            font=("Arial", 14, "bold"),
            fg_color="#2c3e50", corner_radius=8,
            padx=14, pady=6
        )
        self.lbl_algoritm_badge.pack(pady=6, padx=20)

        self.lbl_motiv = ctk.CTkLabel(
            self.sidebar, text="",
            font=("Arial", 10), wraplength=220,
            text_color="gray70", justify="center"
        )
        self.lbl_motiv.pack(pady=(0, 10), padx=10)

        ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30").pack(fill="x", padx=20, pady=10)

        self.btn_incarca = ctk.CTkButton(self.sidebar, text="ÎNCARCĂ IMAGINE",
                                         fg_color="#1a73e8", font=("Arial", 14, "bold"),
                                         command=self.incarca_imagine)
        self.btn_incarca.pack(pady=10, padx=20)

        self.btn_start = ctk.CTkButton(self.sidebar, text="EXECUTĂ ANALIZĂ",
                                       fg_color="#1a73e8", font=("Arial", 14, "bold"),
                                       command=self.executa_analiza)
        self.btn_start.pack(pady=10, padx=20)

        self.btn_batch = ctk.CTkButton(self.sidebar, text="TESTARE ÎN GRUP",
                                       fg_color="#1a73e8", font=("Arial", 14, "bold"),
                                       command=self.executa_batch_testing)
        self.btn_batch.pack(pady=10, padx=20)

        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure((0, 1), weight=1)

        self.setup_display_areas()

        self.status_bar = ctk.CTkLabel(self, text="Sistem pregătit.", fg_color="gray10",
                                       anchor="w", padx=20)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

        self._on_obiect_schimbat(self.combo_obiect.get())

    def _on_obiect_schimbat(self, obiect):
        cfg = config_obiecte.get(obiect, {})
        alg = cfg.get("algoritm_optim", "—")
        culoare = CULORI_ALGORITM.get(alg, "#2c3e50")
        self.lbl_algoritm_badge.configure(text=alg, fg_color=culoare)
        self.lbl_motiv.configure(
            text=f"Algoritm ales pe baza celui mai bun scor obținut în testare pentru componenta „{obiect}“."
        )

    def get_algoritm_optim(self, obj_nume: str) -> str:
        return config_obiecte[obj_nume]["algoritm_optim"]

    def setup_display_areas(self):
        self.frame_in = ctk.CTkFrame(self.main_frame)
        self.frame_in.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.frame_in, text="IMAGINE DE TESTARE",
                     font=("Arial", 12, "bold")).pack(pady=5)
        self.canvas_orig = ctk.CTkLabel(self.frame_in, text="Fără Imagine",
                                        width=450, height=450,
                                        fg_color="black", corner_radius=10)
        self.canvas_orig.pack(padx=10, pady=10)

        self.frame_out = ctk.CTkFrame(self.main_frame)
        self.frame_out.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.frame_out, text="REZULTAT PROCESARE AI",
                     font=("Arial", 12, "bold")).pack(pady=5)
        self.canvas_res = ctk.CTkLabel(self.frame_out, text="Așteptare Analiză",
                                        width=450, height=450,
                                        fg_color="black", corner_radius=10)
        self.canvas_res.pack(padx=10, pady=10)

    def incarca_stare_model(self, model, cale_fisier):
        checkpoint = torch.load(cale_fisier, map_location=self.device, weights_only=True)
        nume_noi = {}
        for cheie, valoare in checkpoint.items():
            noua_cheie = cheie
            if cheie.startswith("codificator_conv"):      noua_cheie = cheie.replace("codificator_conv", "encoder.0")
            elif cheie.startswith("aplatizare"):           noua_cheie = cheie.replace("aplatizare", "encoder.1")
            elif cheie.startswith("codificator_liniar"):   noua_cheie = cheie.replace("codificator_liniar", "encoder.2")
            elif cheie.startswith("decodificator_liniar"): noua_cheie = cheie.replace("decodificator_liniar", "decoder_linear")
            elif cheie.startswith("decodificator_conv"):   noua_cheie = cheie.replace("decodificator_conv", "decoder_conv")
            nume_noi[noua_cheie] = valoare
        model.load_state_dict(nume_noi, strict=False)
        return model

    def incarca_imagine(self):
        from tkinter import filedialog
        cale_baza = r"E:\Facultate\code\Licenta\Aplicatie\dataset"
        obj = self.combo_obiect.get()
        init_dir = os.path.join(cale_baza, obj) if os.path.exists(os.path.join(cale_baza, obj)) else cale_baza
        self.cale_imagine = filedialog.askopenfilename(
            initialdir=init_dir, filetypes=[("Imagini", "*.png *.jpg *.jpeg")])
        if self.cale_imagine:
            img_pil = Image.open(self.cale_imagine).convert("RGB")
            self.img_ctk_orig = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(450, 450))
            self.canvas_orig.configure(image=self.img_ctk_orig, text="")

    def get_model_ae(self, obiect):
        nume_fisier = f"model_{obiect}_autoencoder.pth"
        if not os.path.exists(nume_fisier):
            nume_fisier = f"model_{obiect}_autoencoder_50.pth"
        model = Autoencoder(width=256, height=256, depth=3, filtre=(32, 64, 128, 256), DimLatenta=1024)
        model = self.incarca_stare_model(model, nume_fisier)
        return model.to(self.device).eval()

    def genereaza_heatmap(self, anomaly_map):
        if isinstance(anomaly_map, list):
            if len(anomaly_map) > 0 and isinstance(anomaly_map[0], torch.Tensor):
                harta = anomaly_map[0].detach().cpu().numpy()
            else:
                harta = np.array(anomaly_map, dtype=np.float32)
        elif isinstance(anomaly_map, torch.Tensor):
            harta = anomaly_map.detach().cpu().numpy()
        else:
            harta = np.array(anomaly_map, dtype=np.float32)

        if harta.ndim > 2:
            harta = np.squeeze(harta)

        numitor = harta.max() - harta.min() + 1e-8
        harta_norm = (harta - harta.min()) / numitor
        
        img_bw = Image.fromarray((harta_norm * 255).astype(np.uint8)).resize((450, 450))
        img_color = ImageOps.colorize(img_bw, black="blue", white="red")
        return ctk.CTkImage(light_image=img_color, dark_image=img_color, size=(450, 450))

    def prezice_imagine(self, img_pil, alg_nume, obj_nume, m_ae, d_padim, d_pc):
        t_simple = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        t_norm = transforms.Compose([
            transforms.Resize((256, 256)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_t      = t_simple(img_pil).unsqueeze(0).to(self.device)
        img_t_norm = t_norm(img_pil).unsqueeze(0).to(self.device)
        praguri    = config_obiecte[obj_nume]["praguri"]

        with torch.no_grad():
            if alg_nume == "Autoencoder":
                output = m_ae(img_t)
                err = torch.mean((img_t - output) ** 2).item()
                return err > praguri["Autoencoder"], output

            elif alg_nume == "PaDiM":
                score_padim, a_map = d_padim.predict(img_t_norm)
                return score_padim > praguri["PaDiM"], a_map

            elif alg_nume == "PatchCore":
                err_pc, a_map = d_pc.predict(img_t_norm)
                return err_pc > praguri["PatchCore"], a_map

    def executa_analiza(self):
        if not self.cale_imagine:
            return
        try:
            obj_nume  = self.combo_obiect.get()
            alg_nume  = self.get_algoritm_optim(obj_nume)
            img_pil   = Image.open(self.cale_imagine).convert("RGB")

            self.status_bar.configure(
                text=f"Se rulează {alg_nume} pentru {obj_nume}…", text_color="gray80")
            self.update_idletasks()

            m_ae   = self.get_model_ae(obj_nume) if alg_nume == "Autoencoder" else None
            d_padim = None
            if alg_nume == "PaDiM":
                d_padim = PaDiMDetector(device=self.device)
                d_padim.load(f"model_{obj_nume}_padim.pkl")
            d_pc = None
            if alg_nume == "PatchCore":
                d_pc = PatchCoreDetector(device=self.device)
                d_pc.load(f"model_{obj_nume}_patchcore.pkl")

            is_defect, extra = self.prezice_imagine(img_pil, alg_nume, obj_nume, m_ae, d_padim, d_pc)

            if alg_nume in ["PatchCore", "PaDiM"]:
                self.img_ctk_res = self.genereaza_heatmap(extra)
            elif alg_nume == "Autoencoder":
                res_np = (torch.clamp(extra.squeeze().detach().cpu(), 0, 1)
                          .permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                self.img_ctk_res = ctk.CTkImage(light_image=Image.fromarray(res_np), size=(450, 450))
            else:
                self.img_ctk_res = None

            if self.img_ctk_res:
                self.canvas_res.configure(image=self.img_ctk_res, text="")

            verdict = "DEFECT" if is_defect else "CORECT"
            self.status_bar.configure(
                text=f"Verdict: {verdict}  |  Algoritm folosit: {alg_nume}  |  Componentă: {obj_nume}",
                text_color="white"
            )
            self.frame_out.configure(fg_color="#ff4d4d" if is_defect else "#48bb78")

        except Exception as e:
            self.status_bar.configure(text=f"EROARE: {str(e)}", text_color="red")
            print(f"DEBUG EROARE: {e}")

    def executa_batch_testing(self):
        from tkinter import filedialog
        cale_folder = filedialog.askdirectory(title="Selectează folderul pentru testare în grup")
        if not cale_folder:
            return

        obj_nume = self.combo_obiect.get()
        alg_nume = self.get_algoritm_optim(obj_nume)

        cale_corect = os.path.join(cale_folder, "corect")
        cale_defect  = os.path.join(cale_folder, "defect")
        os.makedirs(cale_corect, exist_ok=True)
        os.makedirs(cale_defect,  exist_ok=True)

        m_ae = self.get_model_ae(obj_nume) if alg_nume == "Autoencoder" else None
        d_padim = None
        if alg_nume == "PaDiM":
            d_padim = PaDiMDetector(device=self.device)
            d_padim.load(f"model_{obj_nume}_padim.pkl")
        d_pc = None
        if alg_nume == "PatchCore":
            d_pc = PatchCoreDetector(device=self.device)
            d_pc.load(f"model_{obj_nume}_patchcore.pkl")

        extensii = ('.png', '.jpg', '.jpeg')
        fisiere  = [f for f in os.listdir(cale_folder) if f.lower().endswith(extensii)]

        c_ok, c_def = 0, 0
        for idx, f in enumerate(fisiere, 1):
            self.status_bar.configure(
                text=f"[{alg_nume}] Procesez {idx}/{len(fisiere)}: {f}", text_color="gray80")
            self.update_idletasks()

            cale_f = os.path.join(cale_folder, f)
            img    = Image.open(cale_f).convert("RGB")
            is_def, _ = self.prezice_imagine(img, alg_nume, obj_nume, m_ae, d_padim, d_pc)

            dest = cale_defect if is_def else cale_corect
            shutil.copy(cale_f, os.path.join(dest, f))
            if is_def:
                c_def += 1
            else:
                c_ok += 1

        self.status_bar.configure(
            text=f"Grup finalizat cu {alg_nume} | {c_ok} CORECTE, {c_def} DEFECTE.",
            text_color="#48bb78"
        )

if __name__ == "__main__":
    app = SistemInspectieAI()
    app.mainloop()