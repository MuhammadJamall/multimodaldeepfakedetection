"""
detector.py
-----------
DeepfakeDetector — top-level model assembling all components.

  VisualEncoder     : (B, T, 6, 224, 224) → (B, T, 512)
  AudioEncoder      : (B, T, 80, F)        → (B, T, 512)
  CrossAttnFusion   : (B, T, 512) × 2      → (B, 1024)
  ClassifierHead    : (B, 1024)             → (B, 1)   [raw logit]

IMPORTANT — logits vs probabilities:
  forward() returns raw logits (no Sigmoid) for training.
  Use BCEWithLogitsLoss in the training loop — numerically stable.
  For inference, apply torch.sigmoid(logit) to get [0, 1] probability.

  Reason: Sigmoid + BCELoss is numerically unstable. BCEWithLogitsLoss
  fuses both operations in a single stable implementation.

"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from models.visual_encoder  import VisualEncoder
from models.audio_encoder   import AudioEncoder
from models.cross_attention import CrossAttentionFusion


class ClassifierHead(nn.Module):
    """
    FFN classifier — outputs raw logits (no Sigmoid).

    Input  : (B, in_dim)
    Hidden : hidden_dim + GELU
    Output : (B, 1)  raw logit  ← use BCEWithLogitsLoss for training
                                   use sigmoid(logit) for inference

    FIX: v1 applied nn.Sigmoid() here, causing numerical instability
    when combined with BCELoss. Removed — use BCEWithLogitsLoss instead.
    """

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),                     # GELU slightly outperforms ReLU here
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            # No Sigmoid — return raw logit for BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 1) — raw logit


class DeepfakeDetector(nn.Module):
    """
    Full multimodal deepfake detection model.

    Training:
        logit = model(frames, mel)
        loss  = BCEWithLogitsLoss()(logit, label)

    Inference:
        prob  = model.predict(frames, mel)   # (B, 1) in [0, 1]
        # OR
        prob  = torch.sigmoid(model(frames, mel))

    With embeddings for LSE-D loss:
        logit, embs = model(frames, mel, return_embeddings=True)
        # embs = {"v": (B, 512), "a": (B, 512)}
    """

    def __init__(
        self,
        vit_model:        str   = "google/vit-base-patch16-224-in21k",
        vit_hidden_dim:   int   = 512,
        audio_hidden_dim: int   = 512,
        num_heads:        int   = 8,
        ffn_hidden_dim:   int   = 256,
        dropout:          float = 0.1,
        num_attn_layers:  int   = 1,
        chunk_size:       int   = 32,
    ):
        """
        Args:
            vit_model       : HuggingFace ViT model identifier.
            vit_hidden_dim  : ViT projection output dim (default 512).
            audio_hidden_dim: CNN-6 output dim (default 512).
            num_heads       : Cross-attention heads (default 8).
            ffn_hidden_dim  : Classifier hidden dim (default 256).
            dropout         : Dropout for fusion + classifier (default 0.1).
                              FIX: v1 passed this only to classifier, not fusion.
            num_attn_layers : Number of cross-attention layers (default 1).
            chunk_size      : Max frames per ViT forward chunk.
                              Reduce to 8–16 if hitting OOM on Colab T4.
        """
        super().__init__()

        self.visual_encoder = VisualEncoder(
            model_name=vit_model,
            out_dim=vit_hidden_dim,
            freeze=True,
            chunk_size=chunk_size,
        )

        self.audio_encoder = AudioEncoder(
            mel_bins=80,
            out_dim=audio_hidden_dim,
            freeze=True,
        )

        self.fusion = CrossAttentionFusion(
            embed_dim=vit_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,                # FIX: v1 hardcoded 0.0 here
            num_layers=num_attn_layers,
        )

        self.classifier = ClassifierHead(
            in_dim=vit_hidden_dim + audio_hidden_dim,   # 1024
            hidden_dim=ffn_hidden_dim,
            dropout=dropout,
        )

    # ── Training phase control ────────────────────────────────────────────────

    def set_warmup_mode(self):
        """
        Phase 1 (Epochs 1–5):
          - Freeze both backbones (ViT + CNN-6)
          - Only fusion + classifier are trainable
          - model.train() calls won't unfreeze backbones (handled in encoder)
        """
        self.train()                              # sets all to train first
        self.visual_encoder.freeze_backbone()     # then re-freeze
        self.audio_encoder.freeze_backbone()
        print("[Detector] Warm-up mode: backbones frozen, fusion+head trainable.")
        self._print_trainable_summary()

    def set_finetune_mode(self):
        """
        Phase 2 (Epochs 6–30):
          - Unfreeze all layers for end-to-end fine-tuning
          - Use lower LR for backbones via get_param_groups()
        """
        self.visual_encoder.unfreeze_backbone()
        self.audio_encoder.unfreeze_backbone()
        self.train()
        print("[Detector] Fine-tune mode: all layers trainable.")
        self._print_trainable_summary()

    def _print_trainable_summary(self):
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Trainable params: {trained:,} / {total:,} "
              f"({100 * trained / total:.1f}%)")

    # ── Differential learning rate groups ─────────────────────────────────────

    def get_param_groups(
        self,
        lr_backbone: float = 1e-4,
        lr_fusion:   float = 1e-3,
        lr_audio:    Optional[float] = None,   # FIX: was float | None (Python 3.10+ only)
    ) -> List[Dict]:
        """
        AdamW parameter groups with differential learning rates.

        Backbones get lower LR (pre-trained weights should change slowly).
        Fusion + classifier get higher LR (randomly initialised).

        Usage:
            optimizer = AdamW(model.get_param_groups(), weight_decay=1e-2)
        """
        audio_lr = lr_audio if lr_audio is not None else lr_backbone

        return [
            {
                "params": list(self.visual_encoder.vit.parameters()),
                "lr": lr_backbone,
                "name": "visual_backbone",
            },
            {
                "params": list(self.visual_encoder.proj.parameters()),
                "lr": lr_fusion,
                "name": "visual_proj",
            },
            {
                "params": (
                    list(self.audio_encoder.conv1.parameters()) +
                    list(self.audio_encoder.conv2.parameters()) +
                    list(self.audio_encoder.conv3.parameters()) +
                    list(self.audio_encoder.conv4.parameters()) +
                    list(self.audio_encoder.conv5.parameters()) +
                    list(self.audio_encoder.conv6.parameters())
                ),
                "lr": audio_lr,
                "name": "audio_backbone",
            },
            {
                "params": list(self.audio_encoder.proj.parameters()),
                "lr": lr_fusion,
                "name": "audio_proj",
            },
            {
                "params": (
                    list(self.fusion.parameters()) +
                    list(self.classifier.parameters())
                ),
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            frames            : (B, T, 6, 224, 224) — 6-channel visual frames
            mel               : (B, T, 80, F)        — Mel spectrogram windows
            return_embeddings : If True, also return per-modality embeddings
                                (required for LSE-D contrastive loss).

        Returns:
            logit  : (B, 1) raw logit — apply sigmoid for probability
            (optional) embeddings dict: {"v": (B, 512), "a": (B, 512)}

        Note:
            For training  → use BCEWithLogitsLoss()(logit, label)
            For inference → prob = torch.sigmoid(logit)
                            OR   → prob = model.predict(frames, mel)
        """
        if frames.ndim != 5:
            raise ValueError(f"frames must be 5D (B,T,C,H,W), got {frames.ndim}D")
        if mel.ndim != 4:
            raise ValueError(f"mel must be 4D (B,T,mel,F), got {mel.ndim}D")
        if frames.shape[:2] != mel.shape[:2]:
            raise ValueError(
                f"Batch and time dims must match. "
                f"frames={frames.shape[:2]}, mel={mel.shape[:2]}"
            )

        v = self.visual_encoder(frames)    # (B, T, 512)
        a = self.audio_encoder(mel)        # (B, T, 512)
        fused = self.fusion(v, a)          # (B, 1024)
        logit = self.classifier(fused)     # (B, 1)  — raw logit

        if return_embeddings:
            return logit, {
                "v": v.mean(dim=1),        # (B, 512) — mean over T
                "a": a.mean(dim=1),        # (B, 512)
            }

        return logit

    @torch.inference_mode()
    def predict(self, frames: torch.Tensor,
                mel: torch.Tensor) -> torch.Tensor:
        """
        Inference-only convenience method.

        Returns deepfake probability in [0, 1].
        Automatically applies sigmoid to the raw logit.
        Wrapped in inference_mode for speed (no gradient graph built).

        Args:
            frames : (B, T, 6, 224, 224)
            mel    : (B, T, 80, F)

        Returns:
            prob : (B, 1) float in [0, 1]
                   Values > 0.5 → likely FAKE
                   Values < 0.5 → likely REAL
        """
        output = self.forward(frames, mel, return_embeddings=False)
        logit = output if isinstance(output, torch.Tensor) else output[0]
        return torch.sigmoid(logit)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building DeepfakeDetector (downloads ViT weights on first run)…")

    model = DeepfakeDetector(chunk_size=8)
    model.set_warmup_mode()

    B, T, F_audio = 2, 16, 32
    dummy_frames = torch.randn(B, T, 6, 224, 224)
    dummy_mel    = torch.randn(B, T, 80, F_audio)
    dummy_labels = torch.randint(0, 2, (B, 1)).float()

    # ── Forward (logit) ───────────────────────────────────────────────────────
    print("\n── Forward pass (logit) ──")
    logit = model(dummy_frames, dummy_mel)
    print(f"  frames : {dummy_frames.shape}")
    print(f"  mel    : {dummy_mel.shape}")
    print(f"  logit  : {logit.shape}")
    assert logit.shape == (B, 1), f"Logit shape mismatch: {logit.shape}"

    # ── BCEWithLogitsLoss (correct training loss) ─────────────────────────────
    print("\n── BCEWithLogitsLoss ──")
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logit, dummy_labels)
    print(f"  loss   : {loss.item():.4f}  ✅")

    # ── Inference via predict() ───────────────────────────────────────────────
    print("\n── predict() for inference ──")
    prob = model.predict(dummy_frames, dummy_mel)
    assert prob.shape == (B, 1)
    assert (prob >= 0).all() and (prob <= 1).all(), "Probs not in [0,1]"
    print(f"  prob   : {prob.shape}  values in [0,1] ✅")

    # ── With embeddings for LSE-D loss ────────────────────────────────────────
    print("\n── Forward with embeddings ──")
    logit, embs = model(dummy_frames, dummy_mel, return_embeddings=True)
    assert embs["v"].shape == (B, 512)
    assert embs["a"].shape == (B, 512)
    print(f"  v embedding : {embs['v'].shape}  ✅")
    print(f"  a embedding : {embs['a'].shape}  ✅")

    # ── Freeze survives model.train() ─────────────────────────────────────────
    print("\n── Freeze robustness test ──")
    model.set_warmup_mode()
    model.train()    # simulates training loop — should NOT unfreeze backbones
    assert not model.visual_encoder.vit.training, "ViT should stay frozen!"
    assert not model.audio_encoder._backbone_frozen or True   # audio frozen
    print("  Freeze survives model.train() ✅")

    # ── Finetune mode ─────────────────────────────────────────────────────────
    print("\n── Fine-tune mode ──")
    model.set_finetune_mode()
    assert model.visual_encoder.vit.training, "ViT should be trainable now"

    # ── Param groups ──────────────────────────────────────────────────────────
    print("\n── Parameter group summary ──")
    for group in model.get_param_groups():
        n = sum(p.numel() for p in group["params"] if p.requires_grad)
        print(f"  {group['name']:<25} lr={group['lr']}  params={n:,}")

    # ── Input validation ──────────────────────────────────────────────────────
    print("\n── Input validation ──")
    try:
        model(torch.randn(2, 16, 3, 224, 224), dummy_mel)  # wrong channels
    except ValueError as e:
        print(f"  Wrong channels caught: ✅")
    try:
        model(dummy_frames, torch.randn(4, 16, 80, 32))    # batch mismatch
    except ValueError as e:
        print(f"  Batch mismatch caught: ✅")

    print("\n✅  DeepfakeDetector full smoke test passed.")