import os
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
    criterion = nn.CrossEntropyLoss()   # binary classification per frame
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # === Resume from latest checkpoint if available ===
    ckpt_dir = "outputs/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = os.path.join(ckpt_dir, "scd_latest.pth")

    start_epoch = 0
    if os.path.exists(latest_path):
        print(f"♻️ Resuming from checkpoint: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"✅ Resumed training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        all_preds, all_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move data to device
            frames = batch["frames"].to(device)
            edges = batch["edges"].to(device)
            hists = batch["hists"].to(device)
            flows = batch["flows"].to(device)
            motion = batch["motion"].to(device)
            coding = batch["coding"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(frames, edges, hists, flows, motion, coding)

            # Reshape for loss
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

        # === Save checkpoint ===
        ckpt_path = os.path.join(ckpt_dir, f"scd_epoch{epoch+1}.pth")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, ckpt_path)

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, latest_path)

        print(f"✅ Saved checkpoint: {ckpt_path} and {latest_path}")

    print("🎉 Training complete!")


if __name__ == "__main__":
    train(num_epochs=3, batch_size=1, lr=1e-3)
