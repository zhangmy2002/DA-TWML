# -*- coding: utf-8 -*-
"""
DA-TWML few-shot experiments for petrographic thin-section images.

This script is designed for the following experiments:
1) DA-TWML score-component ablation:
   ProtoNet, Random Top-k, Loss-only, Loss+U, Loss+D, Loss+C,
   Loss+U+D, Loss+U+C, Full DA-TWML.
2) Multiple seeds and more test episodes.
3) 5-way 1-shot and 5-way 5-shot.
4) PPL/XPL cross-domain evaluation when polarization metadata can be inferred.

Dependencies:
    pip install torch torchvision pandas openpyxl pillow numpy tqdm scikit-learn

Example:
    python rock_fewshot_datwml_experiments.py ^
        --data_root "D:\\南京大学岩石教学薄片显微图像数据集" ^
        --meta_dir "." ^
        --out_dir "runs_acml" ^
        --experiment all ^
        --ways 5 --shots 1 5 --query 5 ^
        --seeds 0 1 2 3 4 ^
        --test_episodes 1000

Notes:
- The script first tries to read Excel metadata, then falls back to path-based labels.
- Class labels are split disjointly into train/val/test.
- If PPL/XPL cannot be inferred, cross-domain experiments are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence, Any

try:
    import numpy as np
    np_version = np.__version__
    if int(np_version.split('.')[0]) >= 2:
        print("[WARN] NumPy 2.x detected. Attempting compatibility fix...")
        os.environ['NUMPY_DISABLE_LAPACK'] = '1'
except ImportError:
    print("[ERROR] NumPy not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2"])
    import numpy as np

try:
    import pandas as pd
except ImportError:
    print("[ERROR] pandas not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

try:
    from PIL import Image
except ImportError:
    print("[ERROR] Pillow not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
    from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    print("[ERROR] tqdm not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    from torchvision import transforms
except ImportError:
    print("[ERROR] PyTorch not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    from torchvision import transforms


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

LABEL_COL_CANDIDATES = [
    "岩石定名", "岩石名称", "类别", "岩类名称", "名称", "类别名称", "薄片名称", "样品名称",
    "rock_name", "rock", "class", "label", "category", "lithology",
]
FILENAME_COL_CANDIDATES = [
    "文件名", "图片名", "图像名", "照片名", "filename", "file", "image", "image_name", "path",
]
POLAR_COL_CANDIDATES = [
    "偏光", "偏振", "成像条件", "光性", "polarization", "polar", "mode", "ppl_xpl",
]
SOURCE_COL_CANDIDATES = [
    "来源", "学校", "数据来源", "source", "institution", "domain", "采集来源",
]
MAG_COL_CANDIDATES = [
    "倍率", "放大倍数", "magnification", "mag", "objective",
]


@dataclass
class Record:
    path: str
    label: str
    broad_group: str
    polarization: str
    source: str
    magnification: str
    domain: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def normalize_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "unknown"
    return str(x).strip()


def pick_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    # fuzzy contains
    for c in columns:
        cl = str(c).lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def infer_broad_group(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["变质", "metamorphic", "gneiss", "schist", "marble"]):
        return "metamorphic"
    if any(k in t for k in ["沉积", "sedimentary", "sandstone", "shale", "mudstone", "limestone", "dolomite"]):
        return "sedimentary"
    if any(k in t for k in ["火成", "igneous", "granite", "basalt", "andesite", "dacite", "gabbro"]):
        return "igneous"
    return "unknown"


def infer_polarization(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["xpl", "cross", "cross-polar", "cross_polar", "正交", "正偏", " crossed"]):
        return "XPL"
    if any(k in t for k in ["ppl", "plain", "plain-polar", "plain_polar", "单偏", "单偏光"]):
        return "PPL"
    return "unknown"


def infer_source(text: str) -> str:
    if "南京" in text or "nanjing" in text.lower() or "nju" in text.lower():
        return "NJU"
    if "中国地质" in text or "cug" in text.lower():
        return "CUG"
    if "北京" in text or "peking" in text.lower() or "pku" in text.lower():
        return "PKU"
    return "unknown"


def infer_magnification(text: str) -> str:
    t = text.lower()
    m = re.search(r"(\d{2,4})\s*[x×]", t)
    if m:
        return m.group(1) + "x"
    for val in ["40", "100", "200", "400"]:
        if f"{val}倍" in text or f"{val}x" in t:
            return val + "x"
    return "unknown"


def read_excel_metadata(meta_dir: Path) -> Dict[str, Dict[str, str]]:
    """Return a mapping from file name/stem to metadata row."""
    meta: Dict[str, Dict[str, str]] = {}
    if not meta_dir.exists():
        return meta

    excel_configs = {
        "南京大学火成岩教学薄片照片信息表.xlsx": {
            "label_col": "Unnamed: 3",
            "fname_col": "Unnamed: 4",
            "start_row": 2,
            "broad_group": "igneous",
        },
        "南京大学沉积岩教学薄片照片信息表.xlsx": {
            "label_col": "Unnamed: 3",
            "fname_col": "Unnamed: 4",
            "start_row": 3,
            "broad_group": "sedimentary",
        },
        "南京大学变质岩教学薄片照片信息表.xlsx": {
            "label_col": "Unnamed: 3",
            "fname_col": "Unnamed: 4",
            "start_row": 2,
            "broad_group": "metamorphic",
        },
    }

    for xlsx in list(meta_dir.glob("*.xlsx")) + list(meta_dir.glob("*.xls")):
        try:
            df = pd.read_excel(xlsx)
        except Exception as e:
            print(f"[WARN] Cannot read metadata file {xlsx}: {e}")
            continue
        if df.empty:
            continue

        config = excel_configs.get(xlsx.name)
        if config:
            label_col = config["label_col"]
            fname_col = config["fname_col"]
            start_row = config["start_row"]
            broad_group = config["broad_group"]
        else:
            label_col = pick_column(df.columns, LABEL_COL_CANDIDATES)
            fname_col = pick_column(df.columns, FILENAME_COL_CANDIDATES)
            start_row = 0
            broad_group = None

        for idx, row in df.iterrows():
            if idx < start_row:
                continue

            row_dict: Dict[str, str] = {}
            if label_col is not None and label_col in df.columns:
                label_val = row.get(label_col)
                row_dict["label"] = normalize_text(label_val)
            else:
                row_dict["label"] = "unknown"

            if row_dict["label"] == "unknown" or row_dict["label"] == "" or row_dict["label"].lower() == "nan":
                continue

            if broad_group:
                row_dict["broad_group"] = broad_group
            else:
                row_text = " ".join([normalize_text(v) for v in row.values]) + " " + xlsx.name
                row_dict["broad_group"] = infer_broad_group(row_text)

            row_dict["polarization"] = "unknown"
            row_dict["source"] = "NJU"
            row_dict["magnification"] = "unknown"

            if fname_col is not None and fname_col in df.columns:
                fname_val = row.get(fname_col)
                if pd.notna(fname_val):
                    fname_str = str(fname_val)
                    if "\n" in fname_str:
                        for part in fname_str.split("\n"):
                            part = part.strip()
                            if part:
                                meta[part] = row_dict
                                meta[Path(part).stem] = row_dict
                    else:
                        meta[fname_str] = row_dict
                        meta[Path(fname_str).stem] = row_dict

            if "label" in row_dict and row_dict["label"] != "unknown":
                meta[row_dict["label"]] = row_dict

    print(f"[INFO] Loaded metadata keys: {len(meta)} from {meta_dir}")
    return meta


KNOWN_DATA_DIRS = {
    "南京大学变质岩教学薄片照片数据集",
    "南京大学沉积岩教学薄片照片数据集",
    "南京大学火成岩教学薄片照片数据集",
}

def scan_dataset(data_root: Path, meta_dir: Optional[Path] = None) -> List[Record]:
    meta = read_excel_metadata(meta_dir) if meta_dir is not None else {}
    
    image_paths = []
    for data_dir in KNOWN_DATA_DIRS:
        dir_path = data_root / data_dir
        if dir_path.exists():
            image_paths.extend([p for p in dir_path.rglob("*") if p.suffix.lower() in IMG_EXTS])
    
    if not image_paths:
        raise RuntimeError(f"No images found in data directories under {data_root}")

    all_meta_labels = {v["label"] for v in meta.values() if "label" in v and v["label"] != "unknown"}

    def find_best_label_match(filename_label: str) -> str:
        if filename_label in all_meta_labels:
            return filename_label
        for meta_label in sorted(all_meta_labels, key=lambda x: -len(x)):
            if meta_label in filename_label or filename_label in meta_label:
                return meta_label
        return filename_label

    records: List[Record] = []
    for p in image_paths:
        rel_text = str(p.relative_to(data_root))
        parent_label = p.parent.name.strip()
        md = meta.get(p.name) or meta.get(p.stem) or meta.get(parent_label) or {}

        label = normalize_text(md.get("label")) if md else "unknown"
        if label == "unknown" or label == "" or label.lower() == "nan":
            stem = p.stem
            match = re.match(r'([变沉火])(\d+)([\u4e00-\u9fff（）()]+?)(\d+-\d+)?$', stem)
            if match:
                label = match.group(3)
            elif parent_label not in KNOWN_DATA_DIRS:
                label = parent_label
            else:
                label = stem
        
        if label in {"images", "image", "imgs", "照片", "图片", "原图"}:
            label = p.stem

        if all_meta_labels:
            label = find_best_label_match(label)

        broad = normalize_text(md.get("broad_group")) if md else "unknown"
        if broad == "unknown":
            broad = infer_broad_group(rel_text + " " + label)

        pol = normalize_text(md.get("polarization")) if md else "unknown"
        if pol == "unknown":
            pol = infer_polarization(rel_text)

        source = normalize_text(md.get("source")) if md else "unknown"
        if source == "unknown":
            source = infer_source(rel_text)

        mag = normalize_text(md.get("magnification")) if md else "unknown"
        if mag == "unknown":
            mag = infer_magnification(rel_text)

        domain_parts = [source, mag, pol]
        domain = "|".join([x for x in domain_parts if x and x != "unknown"]) or "unknown"
        records.append(Record(str(p), label, broad, pol, source, mag, domain))

    # Drop classes with too few images? Keep for now; sampler will filter.
    print(f"[INFO] Found {len(records)} images, {len(set(r.label for r in records))} labels")
    return records


def stratified_class_split(records: List[Record], seed: int, train_frac=0.70, val_frac=0.15) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    group_to_classes: Dict[str, List[str]] = {}
    label_to_group: Dict[str, str] = {}
    for r in records:
        label_to_group.setdefault(r.label, r.broad_group)
    for lab, g in label_to_group.items():
        group_to_classes.setdefault(g, []).append(lab)
    splits = {"train": [], "val": [], "test": []}
    for g, labs in group_to_classes.items():
        labs = sorted(set(labs))
        rng.shuffle(labs)
        n = len(labs)
        n_train = max(1, int(round(n * train_frac))) if n >= 3 else max(1, n - 1)
        n_val = max(1, int(round(n * val_frac))) if n >= 5 else 0
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1 if n >= 3 else 0
        splits["train"].extend(labs[:n_train])
        splits["val"].extend(labs[n_train:n_train+n_val])
        splits["test"].extend(labs[n_train+n_val:])
    for k in splits:
        splits[k] = sorted(splits[k])
    return splits


def load_or_create_splits(records: List[Record], out_dir: Path, seed: int, split_file: Optional[Path]) -> Dict[str, List[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if split_file is not None and split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        print(f"[INFO] Loaded split file: {split_file}")
        return splits
    path = out_dir / f"class_splits_seed{seed}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    splits = stratified_class_split(records, seed=seed)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved split file: {path}")
    return splits


class ImageRecordDataset(Dataset):
    def __init__(self, records: List[Record], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img = Image.open(r.path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, r.label, r.domain, r.polarization, r.path


class EpisodicSampler:
    def __init__(
        self,
        records: List[Record],
        ways: int,
        shots: int,
        query: int,
        transform,
        seed: int,
        support_polarization: Optional[str] = None,
        query_polarization: Optional[str] = None,
        hard_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        self.records = records
        self.ways = ways
        self.shots = shots
        self.query = query
        self.transform = transform
        self.rng = random.Random(seed)
        self.support_polarization = support_polarization
        self.query_polarization = query_polarization
        self.hard_pairs = hard_pairs or []
        self.class_to_records: Dict[str, List[Record]] = {}
        for r in records:
            self.class_to_records.setdefault(r.label, []).append(r)
        self.labels = sorted(self.class_to_records.keys())

    def _filter_by_pol(self, recs: List[Record], pol: Optional[str]) -> List[Record]:
        if pol is None or pol.lower() in {"all", "mixed", "none"}:
            return recs
        return [r for r in recs if r.polarization.upper() == pol.upper()]

    def _eligible_labels(self) -> List[str]:
        labels = []
        for lab, recs in self.class_to_records.items():
            sup = self._filter_by_pol(recs, self.support_polarization)
            qry = self._filter_by_pol(recs, self.query_polarization)
            if self.support_polarization and self.query_polarization and self.support_polarization != self.query_polarization:
                if len(sup) >= self.shots and len(qry) >= self.query:
                    labels.append(lab)
            else:
                if len(recs) >= self.shots + self.query:
                    labels.append(lab)
        return labels

    def _choose_classes(self) -> List[str]:
        eligible = self._eligible_labels()
        if len(eligible) < self.ways:
            raise RuntimeError(
                f"Not enough eligible classes for {self.ways}-way episode. "
                f"Eligible={len(eligible)}, support_pol={self.support_polarization}, query_pol={self.query_polarization}"
            )
        # If hard pairs are provided and present, force one pair into the episode.
        present_pairs = [(a, b) for a, b in self.hard_pairs if a in eligible and b in eligible]
        chosen: List[str] = []
        if present_pairs and self.rng.random() < 0.7:
            a, b = self.rng.choice(present_pairs)
            chosen.extend([a, b])
        remaining = [x for x in eligible if x not in chosen]
        chosen.extend(self.rng.sample(remaining, self.ways - len(chosen)))
        self.rng.shuffle(chosen)
        return chosen

    def sample_episode(self) -> Dict[str, Any]:
        classes = self._choose_classes()
        support_imgs, query_imgs = [], []
        support_y, query_y = [], []
        support_domains, query_domains = [], []
        support_paths, query_paths = [], []
        for ci, lab in enumerate(classes):
            recs = list(self.class_to_records[lab])
            sup_pool = self._filter_by_pol(recs, self.support_polarization)
            qry_pool = self._filter_by_pol(recs, self.query_polarization)
            if self.support_polarization and self.query_polarization and self.support_polarization != self.query_polarization:
                sup_chosen = self.rng.sample(sup_pool, self.shots)
                qry_chosen = self.rng.sample(qry_pool, self.query)
            else:
                chosen = self.rng.sample(recs, self.shots + self.query)
                sup_chosen = chosen[:self.shots]
                qry_chosen = chosen[self.shots:]
            for r in sup_chosen:
                img = self.transform(Image.open(r.path).convert("RGB"))
                support_imgs.append(img); support_y.append(ci); support_domains.append(r.domain); support_paths.append(r.path)
            for r in qry_chosen:
                img = self.transform(Image.open(r.path).convert("RGB"))
                query_imgs.append(img); query_y.append(ci); query_domains.append(r.domain); query_paths.append(r.path)
        return {
            "support_x": torch.stack(support_imgs),
            "support_y": torch.tensor(support_y, dtype=torch.long),
            "query_x": torch.stack(query_imgs),
            "query_y": torch.tensor(query_y, dtype=torch.long),
            "classes": classes,
            "support_domains": support_domains,
            "query_domains": query_domains,
            "support_paths": support_paths,
            "query_paths": query_paths,
        }


class Conv4Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden=64, out_dim=64, dropout=0.0):
        super().__init__()
        blocks = []
        c = in_channels
        for _ in range(4):
            blocks += [
                nn.Conv2d(c, hidden, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            if dropout > 0:
                blocks.append(nn.Dropout2d(dropout))
            c = hidden
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        z = self.net(x)
        z = self.pool(z).flatten(1)
        z = self.proj(z)
        return F.normalize(z, dim=1)


def prototypes_from_support(z_support: torch.Tensor, y_support: torch.Tensor, ways: int) -> torch.Tensor:
    protos = []
    for c in range(ways):
        protos.append(z_support[y_support == c].mean(dim=0))
    return torch.stack(protos, dim=0)


def proto_logits(z_query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    # negative squared Euclidean distance
    dist = torch.cdist(z_query, prototypes, p=2).pow(2)
    return -dist


def episode_forward(encoder: nn.Module, episode: Dict[str, Any], device: torch.device, ways: int):
    sx = episode["support_x"].to(device)
    sy = episode["support_y"].to(device)
    qx = episode["query_x"].to(device)
    qy = episode["query_y"].to(device)
    z_support = encoder(sx)
    z_query = encoder(qx)
    protos = prototypes_from_support(z_support, sy, ways)
    logits = proto_logits(z_query, protos)
    loss = F.cross_entropy(logits, qy)
    probs = F.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)
    acc = (pred == qy).float().mean()
    return loss, acc, probs, z_support, z_query, protos


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()


def prototype_confusion(protos: torch.Tensor, tau_c: float) -> torch.Tensor:
    # high when at least one other prototype is close
    ways = protos.size(0)
    d2 = torch.cdist(protos, protos, p=2).pow(2)
    mask = torch.eye(ways, device=protos.device).bool()
    sim = torch.exp(-d2 / max(tau_c, 1e-8)).masked_fill(mask, -1.0)
    return sim.max(dim=1).values.mean()


def zscore(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1:
        return torch.zeros_like(x)
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


def compute_reference_center(encoder, records: List[Record], transform, device, max_images=512) -> torch.Tensor:
    encoder.eval()
    sample = random.sample(records, min(max_images, len(records)))
    feats = []
    with torch.no_grad():
        for r in sample:
            img = transform(Image.open(r.path).convert("RGB")).unsqueeze(0).to(device)
            feats.append(encoder(img).squeeze(0).cpu())
    encoder.train()
    return torch.stack(feats, dim=0).mean(dim=0).to(device)


def make_transforms(image_size: int, train: bool, aug: str = "conservative"):
    if train and aug == "conservative":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.02, hue=0.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def train_one_method(
    args,
    method: str,
    train_sampler: EpisodicSampler,
    val_sampler: EpisodicSampler,
    test_sampler: EpisodicSampler,
    train_records: List[Record],
    device: torch.device,
    result_dir: Path,
) -> Dict[str, Any]:
    encoder = Conv4Encoder(out_dim=args.embedding_dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    best_state = None
    best_epoch = -1
    history = []

    variants = {
        "protonet": dict(use_score=False, random=False, L=False, U=False, G=False, C=False, topk=False, weighted=False),
        "random_topk": dict(use_score=False, random=True, L=False, U=False, G=False, C=False, topk=True, weighted=False),
        "loss_only": dict(use_score=True, random=False, L=True, U=False, G=False, C=False, topk=True, weighted=True),
        "loss_u": dict(use_score=True, random=False, L=True, U=True, G=False, C=False, topk=True, weighted=True),
        "loss_d": dict(use_score=True, random=False, L=True, U=False, G=True, C=False, topk=True, weighted=True),
        "loss_c": dict(use_score=True, random=False, L=True, U=False, G=False, C=True, topk=True, weighted=True),
        "loss_u_d": dict(use_score=True, random=False, L=True, U=True, G=True, C=False, topk=True, weighted=True),
        "loss_u_c": dict(use_score=True, random=False, L=True, U=True, G=False, C=True, topk=True, weighted=True),
        "full": dict(use_score=True, random=False, L=True, U=True, G=True, C=True, topk=True, weighted=True),
    }
    if method not in variants:
        raise ValueError(f"Unknown method: {method}")
    cfg = variants[method]

    steps_per_epoch = max(1, args.episodes_per_epoch // args.meta_batch)
    k = args.topk if args.topk > 0 else max(1, int(math.ceil(args.topk_ratio * args.meta_batch)))
    k = min(k, args.meta_batch)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        ref_center = None
        if cfg["G"]:
            ref_center = compute_reference_center(encoder, train_records, train_sampler.transform, device, max_images=args.ref_center_images)
        epoch_losses = []
        for _ in range(steps_per_epoch):
            episodes = [train_sampler.sample_episode() for _ in range(args.meta_batch)]
            losses, accs, Us, Gs, Cs = [], [], [], [], []
            for ep in episodes:
                loss, acc, probs, z_s, z_q, protos = episode_forward(encoder, ep, device, args.ways)
                losses.append(loss)
                accs.append(acc.detach())
                Us.append(entropy_from_probs(probs).detach())
                center = torch.cat([z_s, z_q], dim=0).mean(dim=0)
                if ref_center is None:
                    Gs.append(torch.tensor(0.0, device=device))
                else:
                    Gs.append(torch.norm(center.detach() - ref_center.detach(), p=2))
                Cs.append(prototype_confusion(protos, args.tau_c).detach())
            loss_vec = torch.stack(losses)
            # Scheduler scores are detached intentionally; gradients pass through selected query losses.
            score = torch.zeros(args.meta_batch, device=device)
            if cfg["L"]:
                score = score + args.lambda_l * zscore(loss_vec.detach())
            if cfg["U"]:
                score = score + args.lambda_u * zscore(torch.stack(Us))
            if cfg["G"]:
                score = score + args.lambda_g * zscore(torch.stack(Gs))
            if cfg["C"]:
                score = score + args.lambda_c * zscore(torch.stack(Cs))

            if method == "protonet":
                meta_loss = loss_vec.mean()
                selected_idx = torch.arange(args.meta_batch, device=device)
            elif cfg["random"]:
                perm = torch.randperm(args.meta_batch, device=device)
                selected_idx = perm[:k]
                meta_loss = loss_vec[selected_idx].mean()
            else:
                selected_idx = torch.topk(score, k=k, largest=True).indices
                if cfg["weighted"]:
                    w = F.softmax(score[selected_idx] / args.tau_s, dim=0)
                    meta_loss = (w * loss_vec[selected_idx]).sum()
                else:
                    meta_loss = loss_vec[selected_idx].mean()

            opt.zero_grad(set_to_none=True)
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
            opt.step()
            epoch_losses.append(float(meta_loss.detach().cpu()))

        val_acc, val_ci = evaluate(encoder, val_sampler, device, args.ways, args.val_episodes)
        history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "val_acc": val_acc, "val_ci": val_ci})
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k_: v.detach().cpu().clone() for k_, v in encoder.state_dict().items()}
        print(f"[{method}] epoch {epoch:03d} loss={np.mean(epoch_losses):.4f} val={val_acc:.2f}±{val_ci:.2f}")

    if best_state is not None:
        encoder.load_state_dict(best_state)
    test_acc, test_ci = evaluate(encoder, test_sampler, device, args.ways, args.test_episodes)

    result = {
        "method": method,
        "seed": args.seed,
        "ways": args.ways,
        "shots": args.shot_current,
        "query": args.query,
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "test_ci95": test_ci,
        "best_epoch": best_epoch,
        "support_pol": test_sampler.support_polarization or "mixed",
        "query_pol": test_sampler.query_polarization or "mixed",
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(result_dir / f"history_{method}_seed{args.seed}_shot{args.shot_current}.csv", index=False)
    torch.save(best_state, result_dir / f"model_{method}_seed{args.seed}_shot{args.shot_current}.pt")
    with open(result_dir / f"result_{method}_seed{args.seed}_shot{args.shot_current}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


@torch.no_grad()
def evaluate(encoder, sampler: EpisodicSampler, device: torch.device, ways: int, n_episodes: int) -> Tuple[float, float]:
    encoder.eval()
    accs = []
    for _ in range(n_episodes):
        ep = sampler.sample_episode()
        _, acc, _, _, _, _ = episode_forward(encoder, ep, device, ways)
        accs.append(float(acc.cpu()) * 100.0)
    encoder.train()
    mean = float(np.mean(accs))
    ci95 = float(1.96 * np.std(accs, ddof=1) / math.sqrt(len(accs))) if len(accs) > 1 else 0.0
    return mean, ci95


def records_for_split(records: List[Record], labels: Sequence[str], polarization: Optional[str] = None) -> List[Record]:
    s = set(labels)
    out = [r for r in records if r.label in s]
    if polarization is not None and polarization.lower() not in {"all", "mixed", "none"}:
        out = [r for r in out if r.polarization.upper() == polarization.upper()]
    return out


def load_hard_pairs(path: Optional[Path]) -> List[Tuple[str, str]]:
    if path is None or not path.exists():
        return []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [tuple(x) for x in data]
    df = pd.read_csv(path)
    cols = list(df.columns)
    if len(cols) < 2:
        return []
    return [(str(r[cols[0]]), str(r[cols[1]])) for _, r in df.iterrows()]


def summarize_results(results: List[Dict[str, Any]], out_path: Path) -> None:
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    group_cols = [c for c in ["method", "ways", "shots", "support_pol", "query_pol"] if c in df.columns]
    summary = df.groupby(group_cols).agg(
        mean_test_acc=("test_acc", "mean"),
        std_test_acc=("test_acc", "std"),
        n=("test_acc", "count"),
        mean_ci95_episode=("test_ci95", "mean"),
        mean_best_val=("best_val_acc", "mean"),
        mean_best_epoch=("best_epoch", "mean"),
    ).reset_index()
    summary["ci95_over_seeds"] = 1.96 * summary["std_test_acc"].fillna(0) / np.sqrt(summary["n"].clip(lower=1))
    summary.to_csv(out_path.with_name(out_path.stem + "_summary.csv"), index=False, encoding="utf-8-sig")
    print("\n[SUMMARY]")
    print(summary.to_string(index=False))


def make_samplers(args, records, splits, train_tf, eval_tf, shot, support_pol=None, query_pol=None, train_pol=None, hard_pairs=None):
    train_records = records_for_split(records, splits["train"], polarization=train_pol)
    val_records = records_for_split(records, splits["val"])
    test_records = records_for_split(records, splits["test"])
    if len(train_records) == 0 or len(val_records) == 0 or len(test_records) == 0:
        raise RuntimeError("Empty train/val/test records. Please check splits and polarization filters.")
    train_sampler = EpisodicSampler(train_records, args.ways, shot, args.query, train_tf, args.seed, hard_pairs=hard_pairs)
    val_sampler = EpisodicSampler(val_records, args.ways, shot, args.query, eval_tf, args.seed + 123, hard_pairs=hard_pairs)
    test_sampler = EpisodicSampler(
        test_records, args.ways, shot, args.query, eval_tf, args.seed + 456,
        support_polarization=support_pol, query_polarization=query_pol, hard_pairs=hard_pairs,
    )
    return train_sampler, val_sampler, test_sampler, train_records


def check_if_completed(result_dir: Path, method: str, seed: int, shot: int) -> bool:
    result_file = result_dir / f"result_{method}_seed{seed}_shot{shot}.json"
    return result_file.exists()

def run(args):
    print("=" * 60)
    print("DA-TWML Experiment Started")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Meta dir: {args.meta_dir}")
    print(f"Out dir: {args.out_dir}")
    print(f"Experiment: {args.experiment}")
    print(f"Seeds: {args.seeds}")
    print(f"Shots: {args.shots}")
    print(f"Ways: {args.ways}")
    print(f"Query: {args.query}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    data_root = Path(args.data_root)
    meta_dir = Path(args.meta_dir) if args.meta_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = scan_dataset(data_root, meta_dir)
    pd.DataFrame([asdict(r) for r in records]).to_csv(out_dir / "scanned_records.csv", index=False, encoding="utf-8-sig")

    hard_pairs = load_hard_pairs(Path(args.hard_pairs) if args.hard_pairs else None)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    all_results: List[Dict[str, Any]] = []
    train_tf = make_transforms(args.image_size, train=True, aug=args.aug)
    eval_tf = make_transforms(args.image_size, train=False)

    if args.methods is None or len(args.methods) == 0:
        if args.experiment == "ablation":
            methods = ["protonet", "random_topk", "loss_only", "loss_u", "loss_d", "loss_c", "loss_u_d", "loss_u_c", "full"]
        else:
            methods = ["protonet", "loss_only", "full"]
    else:
        methods = args.methods

    total_tasks = 0
    completed_tasks = 0
    skipped_tasks = 0

    for seed in args.seeds:
        args.seed = seed
        set_seed(seed)
        splits = load_or_create_splits(records, out_dir, seed=seed, split_file=Path(args.split_file) if args.split_file else None)
        print(f"[INFO] Split sizes seed={seed}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

        for shot in args.shots:
            args.shot_current = shot
            if args.experiment in {"ablation", "all"}:
                ablation_methods = methods if args.experiment != "ablation" else ["protonet", "random_topk", "loss_only", "loss_u", "loss_d", "loss_c", "loss_u_d", "loss_u_c", "full"]
                total_tasks += len(ablation_methods)
                
                for method in ablation_methods:
                    result_dir = out_dir / f"seed{seed}" / f"shot{shot}" / method
                    if check_if_completed(result_dir, method, seed, shot):
                        print(f"[SKIP] seed={seed}, shot={shot}, method={method} - already completed")
                        skipped_tasks += 1
                        with open(result_dir / f"result_{method}_seed{seed}_shot{shot}.json", "r", encoding="utf-8") as f:
                            all_results.append(json.load(f))
                        continue
                
                print(f"\n[RUN] Ablation / main experiments: seed={seed}, shot={shot}")
                train_sampler, val_sampler, test_sampler, train_records = make_samplers(
                    args, records, splits, train_tf, eval_tf, shot, hard_pairs=hard_pairs
                )
                
                for method in ablation_methods:
                    result_dir = out_dir / f"seed{seed}" / f"shot{shot}" / method
                    if check_if_completed(result_dir, method, seed, shot):
                        continue
                    res = train_one_method(args, method, train_sampler, val_sampler, test_sampler, train_records, device, result_dir)
                    res["experiment"] = "main_ablation"
                    all_results.append(res)
                    completed_tasks += 1

            if args.experiment in {"cross_domain", "all"}:
                cross_methods = [m for m in methods if m in {"protonet", "loss_only", "full"}]
                total_tasks += len(cross_methods) * 2
                
                for train_pol, sup_pol, qry_pol, tag in [("PPL", "PPL", "XPL", "PPL_to_XPL"), ("XPL", "XPL", "PPL", "XPL_to_PPL")]:
                    all_skipped = True
                    for method in cross_methods:
                        result_dir = out_dir / f"seed{seed}" / f"shot{shot}" / tag / method
                        if not check_if_completed(result_dir, method, seed, shot):
                            all_skipped = False
                            break
                    
                    if all_skipped:
                        print(f"[SKIP] Cross-domain {tag}: seed={seed}, shot={shot} - all methods completed")
                        skipped_tasks += len(cross_methods)
                        for method in cross_methods:
                            result_dir = out_dir / f"seed{seed}" / f"shot{shot}" / tag / method
                            with open(result_dir / f"result_{method}_seed{seed}_shot{shot}.json", "r", encoding="utf-8") as f:
                                res = json.load(f)
                                res["experiment"] = tag
                                all_results.append(res)
                        continue
                    
                    print(f"\n[RUN] Cross-domain {tag}: seed={seed}, shot={shot}")
                    try:
                        train_sampler, val_sampler, test_sampler, train_records = make_samplers(
                            args, records, splits, train_tf, eval_tf, shot,
                            support_pol=sup_pol, query_pol=qry_pol, train_pol=train_pol,
                            hard_pairs=hard_pairs,
                        )
                    except Exception as e:
                        print(f"[WARN] Skip {tag}: {e}")
                        continue
                    for method in cross_methods:
                        result_dir = out_dir / f"seed{seed}" / f"shot{shot}" / tag / method
                        if check_if_completed(result_dir, method, seed, shot):
                            continue
                        res = train_one_method(args, method, train_sampler, val_sampler, test_sampler, train_records, device, result_dir)
                        res["experiment"] = tag
                        all_results.append(res)
                        completed_tasks += 1

    print(f"\n[COMPLETION] Total: {total_tasks}, Skipped: {skipped_tasks}, Completed: {completed_tasks}")
    summarize_results(all_results, out_dir / "all_results.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=r"D:\南京大学岩石教学薄片显微图像数据集", help="Dataset root, e.g. D:\\南京大学岩石教学薄片显微图像数据集")
    p.add_argument("--meta_dir", type=str, default=r"D:\南京大学岩石教学薄片显微图像数据集", help="Directory containing xlsx metadata files")
    p.add_argument("--out_dir", type=str, default="runs_acml")
    p.add_argument("--split_file", type=str, default="", help="Optional JSON split file with train/val/test label lists")
    p.add_argument("--hard_pairs", type=str, default="", help="Optional CSV/JSON file of hard class pairs")
    p.add_argument("--experiment", type=str, default="ablation", choices=["ablation", "cross_domain", "all"])
    p.add_argument("--methods", nargs="*", default=None, help="Subset of methods: protonet random_topk loss_only loss_u loss_d loss_c loss_u_d loss_u_c full")
    p.add_argument("--ways", type=int, default=5)
    p.add_argument("--shots", type=int, nargs="+", default=[1,5])
    p.add_argument("--query", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--episodes_per_epoch", type=int, default=80)
    p.add_argument("--val_episodes", type=int, default=200)
    p.add_argument("--test_episodes", type=int, default=1000)
    p.add_argument("--meta_batch", type=int, default=4)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--topk_ratio", type=float, default=0.5)
    p.add_argument("--image_size", type=int, default=84)
    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--aug", type=str, default="conservative", choices=["none", "conservative"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--lambda_l", type=float, default=1.0)
    p.add_argument("--lambda_u", type=float, default=1.0)
    p.add_argument("--lambda_g", type=float, default=1.0)
    p.add_argument("--lambda_c", type=float, default=1.0)
    p.add_argument("--tau_s", type=float, default=2.0)
    p.add_argument("--tau_c", type=float, default=1.0)
    p.add_argument("--ref_center_images", type=int, default=512)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
