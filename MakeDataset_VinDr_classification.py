import os, glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class MakeDataset_VinDr_classification(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_csv: str,
        transform=None,
        mode: str = 'train',
        split_size: float = 0.2,
        target_size: int = 384
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size

        # 1. Load and filter CSV
        df = pd.read_csv(label_csv)
        df = df.dropna(subset=['img_path', 'laterality', 'view_position', 'breast_birads'])

        # 2. Clean BI-RADS labels: strip text prefix, parse to numeric
        df['breast_birads'] = (
            df['breast_birads']
            .astype(str)
            .str.replace(r'^\s*BI-?RADS\s*', '', regex=True)
        )
        df['breast_birads'] = pd.to_numeric(df['breast_birads'], errors='coerce')
        df = df.dropna(subset=['breast_birads'])
        df['breast_birads'] = df['breast_birads'].astype(int)

        # 3. Binary mappings for classification (choose one by uncommenting)
        # Option A: BI-RADS >= 4 → suspicious (1), else (0)
        # df['label'] = df['breast_birads'].apply(lambda x: 1 if x >= 4 else 0)

        # Option B: Normal (0) vs Suspicious (1)
        # df['label'] = df['breast_birads'].apply(lambda x: 0 if x <= 2 else 1)

        # Option C: BI-RADS 1 → 0 (negative), BI-RADS 2–5 → 1 (positive)
        #df['label'] = df['breast_birads'].apply(lambda x: 0 if x == 1 else 1)

        # Option D: Remove BI-RADS 1 cases completely
        df = df[df['breast_birads'] != 1]

        # Option E: Benign (<=3) → 0, Malignant (>=4) → 1
        df['label'] = df['breast_birads'].apply(lambda x: 0 if x <= 2 else 1)

        # Debugging distributions
        print("\nView distribution:")
        print(df['view_position'].value_counts())
        print("\nLaterality distribution:")
        print(df['laterality'].value_counts())

        # 4. Build image lookup mapping
        self.image_map = {}
        for full_path in glob.glob(os.path.join(image_dir, '**', '*.*'), recursive=True):
            rel = os.path.relpath(full_path, image_dir)
            self.image_map[rel] = full_path
            base = os.path.basename(rel)
            if base not in self.image_map:
                self.image_map[base] = full_path

        # 5. Pair up CC and MLO images per breast
        samples = []
        grouped = df.groupby(['study_id', 'laterality'])
        for (study_id, laterality), group in grouped:
            cc = group[group['view_position'] == 'CC']
            mlo = group[group['view_position'] == 'MLO']
            if cc.empty or mlo.empty:
                continue

            cc_rel = cc.iloc[0]['img_path']
            mlo_rel = mlo.iloc[0]['img_path']

            try:
                cc_full = self.find_image_path(cc_rel)
                mlo_full = self.find_image_path(mlo_rel)
            except FileNotFoundError as e:
                print(f"[Warning] Missing file for {study_id} {laterality}: {e}")
                continue

            label = int(cc.iloc[0]['label'])
            samples.append((cc_full, mlo_full, label))

        # 6. Shuffle and split
        np.random.seed(42)
        idx = np.random.permutation(len(samples))
        if mode == 'train':
            split_at = int(len(samples) * (1 - split_size))
            sel = idx[:split_at]
        elif mode == 'val':
            split_at = int(len(samples) * (1 - split_size))
            sel = idx[split_at:]
        else:  # 'all' or 'test'
            sel = idx

        self.samples = [samples[i] for i in sel]

        # Debugging counts
        print(f"Initial count: {len(df)}")
        print(f"After dropping NA: {len(df)}")
        print(f"Unique study_ids: {df['study_id'].nunique()}")
        print(f"Final paired samples: {len(samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cc_path, mlo_path, label = self.samples[idx]
        cc = Image.open(cc_path).convert('RGB')
        mlo = Image.open(mlo_path).convert('RGB')

        if self.transform:
            cc = self.transform(cc)
            mlo = self.transform(mlo)

        return cc, mlo, label

    def find_image_path(self, img_relpath: str) -> str:
        if img_relpath in self.image_map:
            return self.image_map[img_relpath]
        base = os.path.basename(img_relpath)
        if base in self.image_map:
            return self.image_map[base]
        raise FileNotFoundError(f"No match for {img_relpath}")
