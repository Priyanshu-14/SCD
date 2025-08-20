import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.data_loader import get_dataloader
from models.scd_model import SceneChangeDetectionModel
from utils.metrics import f1_score_scene


# -------------------------------
# Training Loop
# -------------------------------
def train(num_epochs=5, batch_size=1, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"🚀 Training on device: {device}")

    # Load data
    train_loader = get_dataloader(batch_size=batch_size, split="train", shuffle=True)

    # Initialize model
    model = SceneChangeDetectionModel()
    model.to(device)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()   # binary classification per frame (scene cut or not)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        all_preds, all_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move data to device
            frames = batch["frames"].to(device)       # (B, T, C, H, W)
            edges = batch["edges"].to(device)         # (B, T, 1, H, W)
            hists = batch["hists"].to(device)         # (B, T, 48)
            flows = batch["flows"].to(device)         # (B, T-1, 2, H, W)
            motion = batch["motion"].to(device)       # (B, T-1, 2, H//16, W//16)
            coding = batch["coding"].to(device)       # (B, T, num_coding_features)
            labels = batch["labels"].to(device)       # (B, T)

            # Forward pass
            outputs = model(frames, edges, hists, flows, motion, coding)  # (B, T, 2)

            # Reshape for loss: combine batch + time
            B, T, C = outputs.shape
            loss = criterion(outputs.view(B*T, C), labels.view(B*T))

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Collect metrics
            preds = torch.argmax(outputs, dim=-1).detach().cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())

        # Compute F1
        f1 = f1_score_scene(all_preds, all_labels)
        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, F1={f1:.4f}")

    print("🎉 Training complete!")


if __name__ == "__main__":
    train(num_epochs=3, batch_size=1, lr=1e-3)
