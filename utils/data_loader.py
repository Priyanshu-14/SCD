import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SceneChangeDataset(Dataset):
    def __init__(self, processed_dir="data/processed", annotation_dir="data/annotations", split="train"):
        """
        Args:
            processed_dir (str): path to preprocessed .npy feature folders
            annotation_dir (str): path to labels
            split (str): "train", "val", "test"
        """
        self.processed_dir = processed_dir
        self.annotation_dir = annotation_dir
        self.split = split

        # List of all video folders
        self.video_names = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        vid_dir = os.path.join(self.processed_dir, video_name)

        # Load preprocessed features
        frames = np.load(os.path.join(vid_dir, "frames.npy"))
        edges = np.load(os.path.join(vid_dir, "edges.npy"))
        hists = np.load(os.path.join(vid_dir, "hist.npy"))
        flows = np.load(os.path.join(vid_dir, "flow.npy"))
        motion = np.load(os.path.join(vid_dir, "motion.npy"))
        coding = np.load(os.path.join(vid_dir, "coding.npy"))

        # Load labels (same length as frames)
        ann_path = os.path.join(self.annotation_dir, f"{video_name}_labels.npy")
        if os.path.exists(ann_path):
            labels = np.load(ann_path)
        else:
            # fallback dummy labels (all zeros)
            labels = np.zeros(frames.shape[0], dtype=np.int64)

        # Convert to torch tensors
        return {
            "frames": torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2),   # (T, C, H, W)
            "edges": torch.tensor(edges, dtype=torch.float32).unsqueeze(1),           # (T, 1, H, W)
            "hists": torch.tensor(hists, dtype=torch.float32),                        # (T, bins*3)
            "flows": torch.tensor(flows, dtype=torch.float32).permute(0, 3, 1, 2),    # (T-1, 2, H, W)
            "motion": torch.tensor(motion, dtype=torch.float32).permute(0, 3, 1, 2),  # (T-1, 2, H//16, W//16)
            "coding": torch.tensor(coding, dtype=torch.float32),                      # (T, num_coding_features)
            "labels": torch.tensor(labels, dtype=torch.long)                          # (T,)
        }


def get_dataloader(batch_size=1, split="train", shuffle=True):
    dataset = SceneChangeDataset(split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


if __name__ == "__main__":
    # Quick test
    dl = get_dataloader(batch_size=1, split="train")
    batch = next(iter(dl))
    for k, v in batch.items():
        print(k, v.shape)
