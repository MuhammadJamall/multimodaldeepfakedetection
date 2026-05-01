"""
detector.py
-----------
DeepfakeDetector — top-level model that assembles all components.

  VisualEncoder   : (B, T, 6, 224, 224) → (B, T, 512)
  AudioEncoder    : (B, T, 80, F)        → (B, T, 512)
  CrossAttnFusion : (B, T, 512) × 2      → (B, 1024)
  ClassifierHead  : (B, 1024)            → (B, 1)   [sigmoid probability]

Also exposes:
  - set_warmup_mode()   : freeze backbones, set fusion+head trainable
  - set_finetune_mode() : unfreeze everything
  - get_param_groups()  : differential learning rates for AdamW
"""

import torch
import torch.nn as nn

from models.visual_encoder import VisualEncoder
from models.audio_encoder  import AudioEncoder
from models.cross_attention import CrossAttentionFusion


class ClassifierHead(nn.Module):
    """
    Simple FFN classifier.

    Input  : (B, 1024)
    Hidden : 256  + ReLU
    Output : (B, 1)  Sigmoid probability
    Dropout: 0.3 before final linear
    """

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 1)


class DeepfakeDetector(nn.Module):
    """
    Full multimodal deepfake detection model.

    Forward pass signature:
        frames : (B, T, 6, 224, 224)   — 6-channel visual frames
        mel    : (B, T, 80, F)         — Mel spectrogram windows
        → prob : (B, 1)                — deepfake probability in [0, 1]

    Additionally returns intermediate embeddings as a dict when
    return_embeddings=True, which is needed for LSE-D loss computation.
    """

    def __init__(
        self,
        vit_model:        str   = "google/vit-base-patch16-224-in21k",
        vit_hidden_dim:   int   = 512,
        audio_hidden_dim: int   = 512,
        num_heads:        int   = 8,
        ffn_hidden_dim:   int   = 256,
        dropout:          float = 0.3,
    ):
        super().__init__()

        # ── Sub-modules ───────────────────────────────────────────────────────
        self.visual_encoder = VisualEncoder(
            model_name=vit_model,
            out_dim=vit_hidden_dim,
            freeze=True,           # start frozen (warm-up phase)
        )

        self.audio_encoder = AudioEncoder(
            mel_bins=80,
            out_dim=audio_hidden_dim,
            freeze=True,           # start frozen (warm-up phase)
        )

        self.fusion = CrossAttentionFusion(
            embed_dim=vit_hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
        )

        self.classifier = ClassifierHead(
            in_dim=vit_hidden_dim + audio_hidden_dim,  # 1024
            hidden_dim=ffn_hidden_dim,
            dropout=dropout,
        )

    # ── Training phase control ─────────────────────────────────────────────────

    def set_warmup_mode(self):
        """
        Phase 1 (Epochs 1–5): Freeze backbones, train fusion + classifier only.
        IMPORTANT: frozen modules stay in eval() to protect BN running stats.
        """
        self.visual_encoder.freeze_backbone()
        self.audio_encoder.freeze_backbone()
        self.fusion.train()
        self.classifier.train()
        print("[Detector] Warm-up mode: backbones frozen, fusion+head trainable.")

    def set_finetune_mode(self):
        """
        Phase 2 (Epochs 6–30): Unfreeze all layers for end-to-end fine-tuning.
        """
        self.visual_encoder.unfreeze_backbone()
        self.audio_encoder.unfreeze_backbone()
        self.train()
        print("[Detector] Fine-tune mode: all layers trainable.")

    # ── Differential learning rate groups ─────────────────────────────────────

    def get_param_groups(
        self,
        lr_backbone: float = 1e-4,
        lr_fusion:   float = 1e-3,
        lr_audio:    float | None = None,
    ) -> list[dict]:
        """
        Returns AdamW parameter groups with differential learning rates.

        Usage:
            optimizer = AdamW(model.get_param_groups(), weight_decay=1e-2)
        """
        audio_lr = lr_audio if lr_audio is not None else lr_backbone
        return [
            # ViT backbone
            {
                "params": self.visual_encoder.vit.parameters(),
                "lr": lr_backbone,
                "name": "visual_backbone",
            },
            # ViT projection head
            {
                "params": self.visual_encoder.proj.parameters(),
                "lr": lr_fusion,
                "name": "visual_proj",
            },
            # CNN-6 backbone
            {
                "params": list(self.audio_encoder.conv1.parameters()) +
                          list(self.audio_encoder.conv2.parameters()) +
                          list(self.audio_encoder.conv3.parameters()) +
                          list(self.audio_encoder.conv4.parameters()) +
                          list(self.audio_encoder.conv5.parameters()) +
                          list(self.audio_encoder.conv6.parameters()),
                "lr": audio_lr,
                "name": "audio_backbone",
            },
            # CNN projection head
            {
                "params": self.audio_encoder.proj.parameters(),
                "lr": lr_fusion,
                "name": "audio_proj",
            },
            # Fusion module + Classifier
            {
                "params": list(self.fusion.parameters()) +
                          list(self.classifier.parameters()),
                "lr": lr_fusion,
                "name": "fusion_classifier",
            },
        ]

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        frames: torch.Tensor,
        mel:    torch.Tensor,
        return_embeddings: bool = False,
    ):
        """
        Args:
            frames            : (B, T, 6, 224, 224)
            mel               : (B, T, 80, F)
            return_embeddings : If True, also return v and a embeddings
                                (required for LSE-D loss computation).

        Returns:
            prob   : (B, 1)  deepfake probability
            (optional) embeddings dict with keys "v" and "a"
        """
        # ── Encode each modality ──────────────────────────────────────────────
        v = self.visual_encoder(frames)   # (B, T, 512)
        a = self.audio_encoder(mel)       # (B, T, 512)

        # ── Fuse with bidirectional cross-attention ───────────────────────────
        fused = self.fusion(v, a)         # (B, 1024)

        # ── Classify ──────────────────────────────────────────────────────────
        prob = self.classifier(fused)     # (B, 1)

        if return_embeddings:
            # Mean-pool over T for LSE-D loss
            return prob, {"v": v.mean(dim=1), "a": a.mean(dim=1)}

        return prob


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building DeepfakeDetector (downloads ViT weights on first run)…")

    model = DeepfakeDetector()
    model.set_warmup_mode()

    # Dummy inputs — no real data needed
    B, T, F_audio = 2, 16, 32
    dummy_frames = torch.randn(B, T, 6, 224, 224)
    dummy_mel    = torch.randn(B, T, 80, F_audio)
    dummy_labels = torch.randint(0, 2, (B, 1)).float()

    print("\n── Forward pass (prob only) ──")
    prob = model(dummy_frames, dummy_mel)
    print(f"  frames shape : {dummy_frames.shape}")
    print(f"  mel    shape : {dummy_mel.shape}")
    print(f"  output shape : {prob.shape}")          # (2, 1)
    assert prob.shape == (B, 1), "Prob shape mismatch!"
    assert (prob >= 0).all() and (prob <= 1).all(), "Prob not in [0,1]!"

    print("\n── Forward pass (with embeddings for LSE-D) ──")
    prob, embs = model(dummy_frames, dummy_mel, return_embeddings=True)
    print(f"  v embedding  : {embs['v'].shape}")     # (2, 512)
    print(f"  a embedding  : {embs['a'].shape}")     # (2, 512)
    assert embs["v"].shape == (B, 512)
    assert embs["a"].shape == (B, 512)

    print("\n── Parameter group summary ──")
    for group in model.get_param_groups():
        n_params = sum(p.numel() for p in group["params"] if p.requires_grad)
        print(f"  {group['name']:<25} lr={group['lr']}  params={n_params:,}")

    print("\n✅  DeepfakeDetector full smoke test passed.")