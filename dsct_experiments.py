# dsct_experiments.py
# ============================================================

import os, io, sys, csv, json, math, time, random, argparse, itertools, yaml
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms.functional import to_pil_image

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import open_clip
from diffusers import AutoencoderKL, StableDiffusionPipeline, StableDiffusionInpaintPipeline

# -------------------------- Utils --------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    
    if not path: return
    target = path
    # If path looks like file, create parent
    if os.path.splitext(path)[1] or os.path.basename(path):
        target = os.path.dirname(path) or path
    if target:
        os.makedirs(target, exist_ok=True)

class CSVLogger:
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        ensure_dir(csv_path)
        self._f = open(csv_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=["step","name","value"])
        self._w.writeheader()
        self._f.flush()

    def add_scalar(self, name: str, value: float, step: int):
        self._w.writerow({"step": step, "name": name, "value": float(value)})
        if step % 25 == 0:
            self._f.flush()

    def close(self):
        self._f.flush(); self._f.close()

def save_json(obj, path):
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# -------------------------- Datasets --------------------------

class ClassFolderDataset(Dataset):
   
    def __init__(self, root: str, classes: Optional[List[str]]=None, size: int=224):
        self.items = []
        # If classes not supplied, discover subfolders as classes (sorted for stability)
        if classes is None:
            classes = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
        self.classes = list(classes)
        for y, cname in enumerate(self.classes):
            cdir = os.path.join(root, cname)
            if not os.path.isdir(cdir): 
                continue
            for f in os.listdir(cdir):
                if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                    self.items.append((os.path.join(cdir, f), y))
        if len(self.items) == 0:
            raise RuntimeError(f"No images found in class-structured root: {root}")
        self.t = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        img = Image.open(p).convert("RGB")
        return {"image": self.t(img), "label": y, "pil": img}

class ImageFolderFlat(Dataset):
    ###Flat image folder (no labels). Returns (tensor, path).
    def __init__(self, root: str, size=224):
        self.paths = []
        for r,_,fs in os.walk(root):
            for f in fs:
                if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                    self.paths.append(os.path.join(r,f))
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in flat folder: {root}")
        self.t = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.t(img), self.paths[idx]

# -------------------------- Masker (BPI) --------------------------

class ForegroundMasker(nn.Module):
    #Simple foreground estimation via DeepLabv3.
    def __init__(self, device="cuda"):
        super().__init__()
        self.net = deeplabv3_resnet50(weights="DEFAULT").to(device).eval()
        self.pre = T.Compose([
            T.Resize((512, 512)),  # force H=W
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])
        self.device = device

    @torch.no_grad()
    def forward(self, pil_imgs: List[Image.Image]) -> List[Image.Image]:
        b = torch.stack([self.pre(im) for im in pil_imgs]).to(self.device)
        lab = self.net(b)["out"].softmax(1).argmax(1)  # [B,H,W]
        masks = []
        for i in range(lab.size(0)):
            fg = (lab[i] != 0).float()   # non-background heuristic
            fg = T.Resize(pil_imgs[i].size[::-1], interpolation=T.InterpolationMode.NEAREST)(fg.unsqueeze(0))
            masks.append(to_pil_image(fg.clamp(0,1)))
        return masks  # white=FG

# -------------------------- DSCT Trainer --------------------------

# -------------------------- DSCT Trainer --------------------------

class DSCTTrainer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.device = torch.device(cfg["device"])

        # Logging
        train_csv = os.path.join(cfg["log_dir"], "train_log.csv")
        ensure_dir(train_csv)
        self.logger = CSVLogger(train_csv)

        # CLIP
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            cfg["clip_arch"], pretrained=cfg["clip_pretrained"], device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(cfg["clip_arch"])

        # prompts for classes
        class_names = cfg["class_names"]
        prompts = [f"a photo of a {c}" for c in class_names]
        with torch.no_grad():
            tok = self.tokenizer(prompts).to(self.device)
            self.text_emb = F.normalize(self.clip_model.encode_text(tok).float(), dim=-1)

        # tune visual encoder only
        self.clip_model.train()
        params = itertools.chain(self.clip_model.visual.parameters())
        self.opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        # VAE
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(cfg["sd_vae_repo"], subfolder="vae").to(self.device).eval()

        # Diffusion for DSG
        self.sd = None
        if cfg["use_dsg"]:
            torch_dtype = torch.float16 if "cuda" in cfg["device"] else torch.float32
            self.sd = StableDiffusionPipeline.from_pretrained(cfg["sd_repo"], torch_dtype=torch_dtype).to(self.device)
            self.sd.safety_checker = None

        # Inpainting for BPI
        self.inpaint, self.masker = None, None
        if cfg["use_bpi"]:
            torch_dtype = torch.float16 if "cuda" in cfg["device"] else torch.float32
            self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
                cfg["sd_inpaint_repo"], torch_dtype=torch_dtype
            ).to(self.device)
            self.inpaint.safety_checker = None
            self.masker = ForegroundMasker(cfg["device"])

        # ✅ Custom collate_fn to handle PIL.Image in batch
        def collate_keep_pil(batch):
            imgs = torch.stack([b["image"] for b in batch])
            labels = torch.tensor([b["label"] for b in batch])
            pils = [b["pil"] for b in batch]  # keep list of PILs, don’t collate
            return {"image": imgs, "label": labels, "pil": pils}

        # DataLoader with safe collate
        self.train_set = ClassFolderDataset(cfg["id_root"], cfg.get("class_names"), size=cfg["image_size"])
        self.loader = DataLoader(
            self.train_set,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_keep_pil,  # 👈 FIX HERE
        )

        # CLIP norm constants
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1,3,1,1)
        self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1,3,1,1)

    # ---------- Losses ----------
    def _class_loss(self, img_emb, y):
        scale = 100.0
        logits = scale * (img_emb @ self.text_emb.T)
        return F.cross_entropy(logits, y)

    def _ood_loss(self, ood_emb):
        scale = 50.0
        sims = scale * (ood_emb @ self.text_emb.T)
        max_sim = sims.max(1).values
        return -torch.log(torch.clamp(1 - torch.sigmoid(max_sim), 1e-6, 1.)).mean()

    # ---------- Helpers ----------
    def _clip_image_embed(self, x):
        emb = self.clip_model.encode_image(x).float()
        return F.normalize(emb, dim=-1)


    def _to_clip_tensor(self, pil_list: List[Image.Image]):
        t = T.Compose([
            T.Resize((self.cfg["image_size"], self.cfg["image_size"]), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.mean.flatten().tolist(), std=self.std.flatten().tolist()),
        ])
        return torch.stack([t(im) for im in pil_list]).to(self.device)

    @torch.no_grad()
    def _reconstruct_l2(self, x):
        px = (x * self.std + self.mean).clamp(0,1)
        lat = self.vae.encode(px * 2 - 1).latent_dist.sample()
        rx  = self.vae.decode(lat).sample
        rx  = (rx + 1) / 2
        e = (px - rx).pow(2).mean(dim=[1,2,3])
        return e, rx

    def _dsg_generate(self, pil_list: List[Image.Image]) -> List[Image.Image]:
        neg_pool = self.cfg["negative_pool"]
        neg_prompts = [random.choice(neg_pool) for _ in pil_list]
        out = self.sd(prompt=neg_prompts, num_inference_steps=self.cfg.get("sd_steps", 25),
                      height=self.cfg.get("sd_h", 512), width=self.cfg.get("sd_w", 512))
        return [img.convert("RGB") for img in out.images]

    def _bpi_purify(self, pil_list: List[Image.Image]) -> List[Image.Image]:
        fg_masks = self.masker(pil_list)
        res = []
        for img, m in zip(pil_list, fg_masks):
            bg_mask = Image.eval(m, lambda v: 255 - v)
            out = self.inpaint(prompt="", image=img, mask_image=bg_mask,
                               num_inference_steps=self.cfg.get("sd_steps", 25))
            res.append(out.images[0].convert("RGB"))
        return res

    # ---------- Train ----------
        # ---------- Train ----------
    def train(self):
        global_step = 0
        epochs = self.cfg["epochs"]
        lambda_ood = float(self.cfg.get("lambda_ood", 0.5))
        use_dsg = bool(self.cfg["use_dsg"])
        use_bpi = bool(self.cfg["use_bpi"])
        use_ruc = bool(self.cfg["use_ruc"])
        alpha_ruc = float(self.cfg.get("alpha_ruc", 1.0))
        beta_align = float(self.cfg.get("beta_align", 0.0))

        for epoch in range(1, epochs + 1):
            self.clip_model.train()
            losses = {"cls": 0, "ood": 0, "align": 0, "total": 0}
            e_rec_stats = []

            for it, batch in enumerate(self.loader):
                pil_batch = batch["pil"]
                x = batch["image"].to(self.device)
                y = batch["label"].to(self.device)

                # BPI purification
                if use_bpi:
                    with torch.no_grad():
                        purified = self._bpi_purify(pil_batch)
                    x_clean = self._to_clip_tensor(purified)
                else:
                    x_clean = x

                # Forward through CLIP (requires grad)
                img_emb = self._clip_image_embed(x_clean)
                L_cls = self._class_loss(img_emb, y)

                # DSG: diffusion-generated OOD samples
                L_ood = torch.tensor(0.0, device=self.device, requires_grad=True)
                if use_dsg:
                    with torch.no_grad():
                        x_ood_pils = self._dsg_generate(pil_batch)
                    x_ood = self._to_clip_tensor(x_ood_pils)
                    ood_emb = self._clip_image_embed(x_ood)
                    L_ood = self._ood_loss(ood_emb)

                # RUC weighting (non-differentiable reconstruction)
                L_ood_scaled = torch.tensor(0.0, device=self.device)
                E_rec_mean = 0.0
                L_align = torch.tensor(0.0, device=self.device)

                if use_ruc:
                    with torch.no_grad():
                        E_rec, recon = self._reconstruct_l2(x_clean)
                    E_rec_mean = float(E_rec.mean().item())
                    ruc_w = (1 + alpha_ruc * E_rec).mean().detach()
                    L_ood_scaled = lambda_ood * ruc_w * L_ood

                    # optional CLIP manifold alignment (keeps gradient)
                    if beta_align > 0:
                        recon_norm = (recon - self.mean) / self.std
                        f_rec = self._clip_image_embed(recon_norm)
                        L_align = F.mse_loss(img_emb, f_rec) * beta_align
                else:
                    L_ood_scaled = lambda_ood * L_ood

                # total differentiable loss
                L_total = L_cls + L_ood_scaled + L_align

                # safeguard: ensure requires_grad
                if not L_total.requires_grad:
                    L_total = L_cls  # fallback to supervised loss

                self.opt.zero_grad(set_to_none=True)
                L_total.backward()
                self.opt.step()

                # tracking
                losses["cls"] += L_cls.item()
                losses["ood"] += L_ood_scaled.item()
                losses["align"] += L_align.item()
                losses["total"] += L_total.item()
                e_rec_stats.append(E_rec_mean)

                if it % 10 == 0:
                    step = epoch * 1000 + it
                    self.logger.add_scalar("train/cls", L_cls.item(), step)
                    self.logger.add_scalar("train/ood_scaled", L_ood_scaled.item(), step)
                    self.logger.add_scalar("train/align", L_align.item(), step)
                    self.logger.add_scalar("train/total", L_total.item(), step)
                    self.logger.add_scalar("train/Erec_mean", E_rec_mean, step)
                global_step += 1

            n = max(1, len(self.loader))
            print(f"[Epoch {epoch:02d}] cls={losses['cls']/n:.4f}  "
                  f"ood={losses['ood']/n:.4f}  align={losses['align']/n:.4f}  "
                  f"total={losses['total']/n:.4f}  Erec={np.mean(e_rec_stats):.4f}")

        ensure_dir(self.cfg["ckpt_path"])
        torch.save(self.clip_model.state_dict(), self.cfg["ckpt_path"])
        self.logger.close()
        print(f"[Checkpoint] Saved to {self.cfg['ckpt_path']}")
        print(f"[Logs] Train CSV -> {os.path.join(self.cfg['log_dir'], 'train_log.csv')}")



# -------------------------- Evaluation --------------------------

def robust_minmax(x, lo=5, hi=95, eps=1e-8):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / (b - a + eps), 0.0, 1.0)

@torch.no_grad()
def extract_clip_stats(model, text_emb, loader, device, sim_scale=100.0):
    sims, ents, confs = [], [], []
    for x, _ in tqdm(loader, desc="CLIP pass"):
        x = x.to(device)
        img_emb = F.normalize(model.encode_image(x).float(), dim=-1)
        logits = sim_scale * (img_emb @ text_emb.T)
        probs = logits.softmax(-1)
        H = -(probs * (probs.clamp_min(1e-12)).log()).sum(-1)  # entropy
        Hn = H / np.log(probs.size(-1))
        max_sim = (img_emb @ text_emb.T).max(1).values
        max_prob = probs.max(-1).values
        sims.append(max_sim.detach().cpu())
        ents.append(Hn.detach().cpu())
        confs.append(max_prob.detach().cpu())
    sims = torch.cat(sims).numpy()
    ents = torch.cat(ents).numpy()
    confs = torch.cat(confs).numpy()
    return sims, ents, confs

@torch.no_grad()
def vae_recon_energy(vae, loader, device, mean, std):
    E_all = []
    for x, _ in tqdm(loader, desc="VAE recon"):
        x = x.to(device)
        px = (x * std + mean).clamp(0,1)
        lat = vae.encode(px * 2 - 1).latent_dist.sample()
        rx  = vae.decode(lat).sample
        rx  = (rx + 1) / 2
        e = (px - rx).pow(2).mean(dim=[1,2,3])
        E_all.append(e.detach().cpu())
    return torch.cat(E_all).numpy()

def fpr_at_95_tpr_id(id_scores, ood_scores):
    thr = np.sort(np.concatenate([id_scores, ood_scores]))
    # TPR by threshold; find closest to 0.95
    tpr = [(id_scores >= t).mean() for t in thr]
    fpr = [(ood_scores >= t).mean() for t in thr]
    tpr = np.array(tpr); fpr = np.array(fpr)
    idx = np.argmin(np.abs(tpr - 0.95))
    return float(fpr[idx])

def expected_calibration_error(confs: np.ndarray, labels: np.ndarray, bins: int = 15):
    """ECE for binary ground truth (ID=1, OOD=0) using confidence for positive (ID)."""
    # Here we treat 'confidence' as ID confidence. labels: 1 for ID, 0 for OOD.
    ece = 0.0
    bin_bounds = np.linspace(0, 1, bins+1)
    for i in range(bins):
        lo, hi = bin_bounds[i], bin_bounds[i+1]
        sel = (confs >= lo) & (confs < hi)
        if not np.any(sel): 
            continue
        acc = labels[sel].mean()
        conf = confs[sel].mean()
        ece += (sel.mean()) * abs(acc - conf)
    return float(ece)

def evaluate_ruc(cfg: Dict, save_path: Optional[str] = None) -> Dict[str, float]:
    device = torch.device(cfg["device"])
    # CLIP
    model, _, _ = open_clip.create_model_and_transforms(cfg["clip_arch"], pretrained=cfg["clip_pretrained"], device=device)
    if os.path.isfile(cfg["ckpt_path"]):
        print(f"[Eval] Loading checkpoint: {cfg['ckpt_path']}")
        model.load_state_dict(torch.load(cfg["ckpt_path"], map_location=device), strict=False)
    model.eval()

    # Text emb
    tokenizer = open_clip.get_tokenizer(cfg["clip_arch"])
    prompts = [f"a photo of a {c}" for c in cfg["class_names"]]
    with torch.no_grad():
        tok = tokenizer(prompts).to(device)
        text_emb = F.normalize(model.encode_text(tok).float(), dim=-1)

    # Data
    id_ds  = ImageFolderFlat(cfg["id_root"], size=cfg["image_size"])
    ood_ds = ImageFolderFlat(cfg["ood_root"], size=cfg["image_size"])
    id_ld  = DataLoader(id_ds,  batch_size=cfg["batch_size"], num_workers=cfg.get("num_workers", 4))
    ood_ld = DataLoader(ood_ds, batch_size=cfg["batch_size"], num_workers=cfg.get("num_workers", 4))

    # Stats
    sims_id,  ent_id,  conf_id  = extract_clip_stats(model, text_emb, id_ld,  device, cfg["sim_scale"])
    sims_ood, ent_ood, conf_ood = extract_clip_stats(model, text_emb, ood_ld, device, cfg["sim_scale"])

    # VAE recon energy
    vae = AutoencoderKL.from_pretrained(cfg["sd_vae_repo"], subfolder="vae").to(device).eval()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)
    E_id  = vae_recon_energy(vae, id_ld,  device, mean, std)
    E_ood = vae_recon_energy(vae, ood_ld, device, mean, std)

    # Normalize (robust)
    ent_id_n  = robust_minmax(ent_id)
    ent_ood_n = robust_minmax(ent_ood)
    E_id_n    = robust_minmax(E_id)
    E_ood_n   = robust_minmax(E_ood)
    conf_id_n   = robust_minmax(sims_id)   # use similarity as ID confidence proxy
    conf_ood_n  = robust_minmax(sims_ood)

    # RUC uncertainty (higher=more OOD)
    gamma = float(cfg.get("gamma_ruc", 0.5))
    u_id  = gamma*ent_id_n  + (1-gamma)*E_id_n
    u_ood = gamma*ent_ood_n + (1-gamma)*E_ood_n

    # OOD-positive metrics
    y_ood = np.concatenate([np.zeros_like(u_id), np.ones_like(u_ood)])
    u_all = np.concatenate([u_id, u_ood])
    auroc = roc_auc_score(y_ood, u_all)
    prec, rec, _ = precision_recall_curve(y_ood, u_all)
    auprc = auc(rec, prec)

    # FPR@95 with ID positive using similarity-based confidence
    fpr95 = fpr_at_95_tpr_id(conf_id_n, conf_ood_n)

    # ECE as ID calibration using softmax max-prob from CLIP logits (already computed as conf_id/conf_ood)
    id_labels = np.concatenate([np.ones_like(conf_id), np.zeros_like(conf_ood)])
    id_conf   = np.concatenate([conf_id, conf_ood])
    ece = expected_calibration_error(id_conf, id_labels, bins=int(cfg.get("ece_bins", 15)))

    metrics = {
        "auroc_oodpos": float(auroc),
        "auprc_oodpos": float(auprc),
        "fpr95_idpos":  float(fpr95),
        "ece":          float(ece),
    }

    if save_path:
        ensure_dir(save_path)
        with open(save_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            w.writeheader(); w.writerow(metrics)
        print(f"[Eval] Saved metrics -> {save_path}")

    print("\n=== RUC-based Evaluation ===")
    print(f"AUROC (OOD+): {auroc*100:.2f}")
    print(f"AUPRC (OOD+): {auprc*100:.2f}")
    print(f"FPR@95 (ID+): {fpr95*100:.2f}")
    print(f"ECE (ID conf): {ece*100:.2f}")
    return metrics

# -------------------------- Ablations & Multi-seed --------------------------

def run_train(cfg):
    set_seed(cfg.get("seed", 42))
    ensure_dir(cfg.get("log_dir","logs"))
    ensure_dir(cfg.get("ckpt_path","checkpoints/model.pt"))
    trainer = DSCTTrainer(cfg)
    trainer.train()

def run_eval(cfg):
    set_seed(cfg.get("seed", 42))
    ensure_dir(cfg.get("log_dir","logs"))
    out_csv = os.path.join(cfg["log_dir"], "eval_log.csv")
    evaluate_ruc(cfg, out_csv)

def run_ablation(cfg, seeds: List[int], combos: List[Tuple[bool,bool,bool]], out_dir: str):
    """
    combos list items are (use_dsg, use_bpi, use_ruc)
    Runs training+eval per seed and combo; writes summary CSV + latex table to add to Dsct paper(cpts_528) dirctly.
    """
    rows = []
    for (use_dsg, use_bpi, use_ruc) in combos:
        tag = f"DSG{int(use_dsg)}_BPI{int(use_bpi)}_RUC{int(use_ruc)}"
        for sd in seeds:
            cfg_run = dict(cfg)
            cfg_run.update({
                "use_dsg": use_dsg,
                "use_bpi": use_bpi,
                "use_ruc": use_ruc,
                "seed": sd,
                "log_dir": os.path.join(out_dir, f"{tag}_seed{sd}"),
                "ckpt_path": os.path.join(out_dir, f"{tag}_seed{sd}", "ckpt.pt"),
            })
            print(f"\n[Run] {tag}  seed={sd}")
            run_train(cfg_run)
            metrics = evaluate_ruc(cfg_run, save_path=os.path.join(cfg_run["log_dir"], "eval_log.csv"))
            rows.append({"tag": tag, "seed": sd, **metrics})
    # Save summary CSV
    summary_csv = os.path.join(out_dir, "ablation_summary.csv")
    ensure_dir(summary_csv)
    with open(summary_csv, "w", newline="") as f:
        flds = ["tag", "seed", "auroc_oodpos", "auprc_oodpos", "fpr95_idpos", "ece"]
        w = csv.DictWriter(f, fieldnames=flds); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[Ablation] Summary -> {summary_csv}")

    # Aggregate mean±std per tag & write LaTeX table
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        tag = r["tag"]
        for k in ["auroc_oodpos","auprc_oodpos","fpr95_idpos","ece"]:
            agg[tag][k].append(float(r[k]))
    table_lines = []
    header = "\\begin{tabular}{lcccc}\\toprule\n" \
             "Config & AUROC$\\uparrow$ & AUPRC$\\uparrow$ & FPR@95$\\downarrow$ & ECE$\\downarrow$ \\\\\n\\midrule"
    table_lines.append(header)
    for tag, d in agg.items():
        def fmt(k, scale=100.0):
            arr = np.array(d[k]); mu = arr.mean()*scale; sd = arr.std()*scale
            return f"{mu:.2f}$\\pm${sd:.2f}"
        line = f"{tag} & {fmt('auroc_oodpos')} & {fmt('auprc_oodpos')} & {fmt('fpr95_idpos')} & {fmt('ece')} \\\\"
        table_lines.append(line)
    table_lines.append("\\bottomrule\n\\end{tabular}")
    tex_path = os.path.join(out_dir, "ablation_table.tex")
    ensure_dir(tex_path)
    with open(tex_path, "w") as f:
        f.write("\n".join(table_lines))
    print(f"[Ablation] LaTeX table -> {tex_path}")

# -------------------------- Default Config --------------------------

def default_cfg() -> Dict:
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "clip_arch": "ViT-B-16",
        "clip_pretrained": "openai",
        "image_size": 224,
        "batch_size": 32,
        "num_workers": 4,
        "epochs": 20,
        "lr": 1e-5,
        "weight_decay": 1e-4,
        # DSCT toggles
        "use_dsg": True,
        "use_bpi": True,
        "use_ruc": True,
        "lambda_ood": 0.5,
        "alpha_ruc": 1.0,
        "beta_align": 0.0,
        "gamma_ruc": 0.5,
        # Diffusion repos
        "sd_repo": "runwayml/stable-diffusion-v1-5",
        "sd_inpaint_repo": "runwayml/stable-diffusion-inpainting",
        "sd_vae_repo": "runwayml/stable-diffusion-v1-5",  # uses subfolder="vae"
        "sd_steps": 25,
        "sd_h": 512, "sd_w": 512,
        # Prompts
        "class_names": ["cat","dog"],  # override via YAML/CLI
        "negative_pool": ["a photo of a traffic sign", "a watercolor landscape", "a close-up of a keyboard"],
        # Paths
        "id_root":  "data/ID",          # class-structured
        "ood_root": "data/OOD_flat",    # flat folder
        "log_dir":  "runs/default",
        "ckpt_path":"runs/default/ckpt.pt",
        # Eval
        "sim_scale": 100.0,
        "ece_bins": 15,
        # Repro
        "seed": 42,
    }

# -------------------------- CLI --------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DSCT Experiments")
    p.add_argument("--config", type=str, default=None, help="YAML config path (optional)")
    p.add_argument("--mode", type=str, required=True, choices=["train","eval","ablate"],
                   help="Run mode")
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="Seeds for ablation/multi-seed")
    p.add_argument("--out_dir", type=str, default="runs/ablation", help="Output dir for ablation")
    # quick overrides
    p.add_argument("--id_root", type=str, default=None)
    p.add_argument("--ood_root", type=str, default=None)
    p.add_argument("--class_names", type=str, nargs="*", default=None)
    p.add_argument("--negative_pool", type=str, nargs="*", default=None)
    p.add_argument("--use_dsg", type=int, default=None)
    p.add_argument("--use_bpi", type=int, default=None)
    p.add_argument("--use_ruc", type=int, default=None)
    return p.parse_args()

def load_cfg(args) -> Dict:
    cfg = default_cfg()
    if args.config is not None:
        with open(args.config, "r") as f:
            user_cfg = yaml.safe_load(f)
        cfg.update(user_cfg or {})

    # CLI overrides
    if args.id_root: cfg["id_root"] = args.id_root
    if args.ood_root: cfg["ood_root"] = args.ood_root
    if args.class_names is not None and len(args.class_names) > 0:
        cfg["class_names"] = args.class_names
    if args.negative_pool is not None and len(args.negative_pool) > 0:
        cfg["negative_pool"] = args.negative_pool
    if args.use_dsg is not None: cfg["use_dsg"] = bool(args.use_dsg)
    if args.use_bpi is not None: cfg["use_bpi"] = bool(args.use_bpi)
    if args.use_ruc is not None: cfg["use_ruc"] = bool(args.use_ruc)

    return cfg

# -------------------------- Main --------------------------

def main():
    args = parse_args()
    cfg = load_cfg(args)

    if args.mode == "train":
        run_train(cfg)

    elif args.mode == "eval":
        run_eval(cfg)

    elif args.mode == "ablate":
        seeds = args.seeds or [42, 43, 44]
        combos = [
            (0,0,0),  # CLIP baseline (no DSG/BPI/RUC)
            (1,0,0),  # DSG only
            (0,1,0),  # BPI only
            (0,0,1),  # RUC only
            (1,1,0),  # DSG + BPI
            (1,0,1),  # DSG + RUC
            (0,1,1),  # BPI + RUC
            (1,1,1),  # Full DSCT
        ]
        combos = [(bool(a), bool(b), bool(c)) for (a,b,c) in combos]
        run_ablation(cfg, seeds, combos, args.out_dir)

if __name__ == "__main__":
    main()
