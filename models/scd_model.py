import torch
import torch.nn as nn

# Import encoders
from models.encoders.visual_encoder import VisualEncoder
from models.encoders.motion_encoder import MotionEncoder
from models.encoders.flow_encoder import FlowEncoder
from models.encoders.edge_encoder import EdgeEncoder
from models.encoders.hist_encoder import HistEncoder
from models.encoders.coding_encoder import CodingEncoder

# Import fusion, temporal, classifier
from models.fusion import FusionModule
from models.temporal import TemporalBackbone
from models.classifier import SceneChangeClassifier


import torch
import torch.nn as nn

class SceneChangeDetectionModel(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=64, num_classes=2):
        super(SceneChangeDetectionModel, self).__init__()

        # ---------------------------
        # Simple encoders
        # ---------------------------
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.hist_encoder = nn.Linear(48, 32)
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(2, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(2, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.coding_encoder = nn.Linear(10, 16)

        # ---------------------------
        # Infer feature dimensions (dummy forward)
        # ---------------------------
        with torch.no_grad():
            dummy_img = torch.zeros(1, 3, 64, 64)   # matches preprocessing size
            dummy_edge = torch.zeros(1, 1, 64, 64)
            dummy_flow = torch.zeros(1, 2, 64, 64)
            dummy_motion = torch.zeros(1, 2, 4, 4)

            self.frame_dim = self.frame_encoder(dummy_img).shape[1]
            self.edge_dim = self.edge_encoder(dummy_edge).shape[1]
            self.flow_dim = self.flow_encoder(dummy_flow).shape[1]
            self.motion_dim = self.motion_encoder(dummy_motion).shape[1]

        # ---------------------------
        # Fusion + classifier
        # ---------------------------
        fusion_input_dim = self.frame_dim + self.edge_dim + 32 + self.flow_dim + self.motion_dim + 16
        self.fusion = nn.Linear(fusion_input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, frames, edges, hists, flows, motion, coding):
        """
        Args:
            frames: (B, T, 3, H, W)
            edges:  (B, T, 1, H, W)
            hists:  (B, T, 48)
            flows:  (B, T-1, 2, H, W)
            motion: (B, T-1, 2, H//16, W//16)
            coding: (B, T, 10)
        """
        B, T, _, _, _ = frames.shape
        outputs = []

        for t in range(T):
            f_feat = self.frame_encoder(frames[:, t])
            e_feat = self.edge_encoder(edges[:, t])
            h_feat = self.hist_encoder(hists[:, t])
            c_feat = self.coding_encoder(coding[:, t])

            if t > 0:
                fl_feat = self.flow_encoder(flows[:, t-1])
                m_feat = self.motion_encoder(motion[:, t-1])
            else:
                # first frame has no flow/motion → zeros of correct size
                fl_feat = torch.zeros(B, self.flow_dim, device=frames.device)
                m_feat = torch.zeros(B, self.motion_dim, device=frames.device)

            combined = torch.cat([f_feat, e_feat, h_feat, c_feat, fl_feat, m_feat], dim=-1)
            fused = self.fusion(combined)
            out = self.classifier(fused)
            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (B, T, num_classes)

