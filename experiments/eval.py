import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.scd_model import SceneChangeDetectionModel
import os


# -------------------------------
# 1. Dummy Test Dataset
# -------------------------------
class DummySceneDataset(Dataset):
    def __init__(self, num_samples=20, seq_len=16, num_classes=3):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        batch = {
            "visual": torch.randn(self.seq_len, 3, 64, 64),
            "motion": torch.randn(self.seq_len, 1, 64, 64),
            "flow": torch.randn(self.seq_len, 2, 64, 64),
            "edge": torch.randn(self.seq_len, 1, 64, 64),
            "hist": torch.randn(self.seq_len, 48),
            "coding": torch.randn(self.seq_len, 10),
            "lengths": torch.tensor(self.seq_len),
        }
        labels = torch.randint(0, self.num_classes, (self.seq_len,))
        return batch, labels


# -------------------------------
# 2. Collate Function
# -------------------------------
def collate_fn(batch):
    batch_dict = {}
    lengths = []
    labels_list = []

    for features, labels in batch:
        for k, v in features.items():
            if k == "lengths":
                lengths.append(v)
            else:
                batch_dict.setdefault(k, []).append(v)
        labels_list.append(labels)

    for k in batch_dict:
        batch_dict[k] = torch.stack(batch_dict[k], dim=0)
    batch_dict["lengths"] = torch.stack(lengths, dim=0)
    labels = torch.stack(labels_list, dim=0)
    return batch_dict, labels


# -------------------------------
# 3. Evaluation Function
# -------------------------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_frames = 0, 0, 0

    with torch.no_grad():
        for batch, labels in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            labels = labels.to(device)

            logits = model(batch)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            labels = labels.view(B * T)

            loss = criterion(logits, labels)

            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_frames += labels.numel()
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_frames
    return avg_loss, accuracy


# -------------------------------
# 4. Main
# -------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DummySceneDataset(num_samples=20, seq_len=16, num_classes=3)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    model = SceneChangeDetectionModel(num_classes=3).to(device)

    # Load checkpoint
    checkpoint_path = "outputs/checkpoints/scd_latest.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint not found at {checkpoint_path}. Train the model first.")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"✅ Loaded checkpoint: {checkpoint_path}")

    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, dataloader, criterion, device)
    print(f"Evaluation: Loss={loss:.4f}, Accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
