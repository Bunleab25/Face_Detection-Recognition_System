"""ArcFace face recognition model architecture.

Uses ResNet backbone for feature extraction + ArcFace loss for training.

Key differences from the original paper (which used VGG16):
1. Backbone: ResNet18/ResNet50 (lighter, better for 112×112 aligned faces)
2. Input: 112×112 (ArcFace standard) instead of 224×224
3. Loss: ArcFace angular margin instead of standard softmax
4. Embeddings: 512-d L2-normalized features for face verification

Architecture:
    Input (B, 3, 112, 112)
      ↓
    ResNet backbone (modified for 112×112)
      ↓
    Embedding layer (512-d)
      ↓
    L2 normalization
      ↓
    ArcFace loss (training only)
      ↓
    Logits (B, num_classes)

Inference mode:
    Input → Backbone → Embedding (512-d) → L2 normalize → Compare with cosine similarity
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from scripts.arcface_layer import ArcFace, CosFace


class FaceRecognitionModel(nn.Module):
    """Face recognition model with ResNet backbone and ArcFace loss.

    Args:
        num_classes: Number of identity classes in training set.
        embedding_size: Size of feature embeddings (default: 512).
        backbone: ResNet variant ("resnet18", "resnet34", "resnet50", default: "resnet18").
        pretrained: Use ImageNet-pretrained weights (default: True).
        loss_type: Metric loss type ("arcface" or "cosface", default: "arcface").
        scale: ArcFace scale parameter (default: 64.0).
        margin: ArcFace angular margin in radians (default: 0.5) or CosFace margin (default: 0.35).
        dropout: Dropout probability before embedding layer (default: 0.0).
    """

    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 512,
        backbone: Literal["resnet18", "resnet34", "resnet50"] = "resnet18",
        pretrained: bool = True,
        loss_type: Literal["arcface", "cosface"] = "arcface",
        scale: float = 64.0,
        margin: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.loss_type = loss_type

        # 1. Backbone: ResNet feature extractor
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_out_features = 512
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            backbone_out_features = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_out_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove original FC layer (we'll add our own embedding layer).
        self.backbone.fc = nn.Identity()

        # Optional: Modify first conv layer if input is different from ImageNet.
        # For ArcFace 112×112, we keep the standard 7×7 conv (works fine).
        # If you want to optimize for 112×112, you could use:
        # self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.backbone.maxpool = nn.Identity()

        # 2. Optional dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # 3. Embedding layer (maps backbone features to embedding_size)
        self.embedding = nn.Linear(backbone_out_features, embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.constant_(self.embedding.bias, 0.0)

        # 4. Batch normalization on embeddings (common in ArcFace)
        self.bn = nn.BatchNorm1d(embedding_size, eps=1e-05, momentum=0.1, affine=True)

        # 5. Metric loss layer (ArcFace or CosFace)
        if loss_type == "arcface":
            self.metric_loss = ArcFace(
                in_features=embedding_size,
                out_features=num_classes,
                scale=scale,
                margin=margin,
            )
        elif loss_type == "cosface":
            self.metric_loss = CosFace(
                in_features=embedding_size,
                out_features=num_classes,
                scale=scale,
                margin=margin,
            )
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input images (B, 3, 112, 112).
            labels: Ground-truth class labels (B,). If None, inference mode (return embeddings only).

        Returns:
            If training (labels provided):
                logits: (B, num_classes) ArcFace logits for classification loss.
                embeddings: (B, embedding_size) L2-normalized feature embeddings.
            If inference (labels=None):
                embeddings: (B, embedding_size) L2-normalized feature embeddings.
        """
        # Extract features from backbone
        features = self.backbone(x)  # (B, backbone_out_features)

        # Apply dropout
        features = self.dropout(features)

        # Map to embedding space
        embeddings = self.embedding(features)  # (B, embedding_size)

        # Batch normalization
        embeddings = self.bn(embeddings)

        # Training mode: compute ArcFace logits
        if labels is not None:
            logits = self.metric_loss(embeddings, labels)
            return logits, embeddings

        # Inference mode: return normalized embeddings for verification
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized embeddings (inference mode).

        Args:
            x: Input images (B, 3, 112, 112).

        Returns:
            embeddings: (B, embedding_size) L2-normalized feature embeddings.
        """
        with torch.no_grad():
            return self.forward(x, labels=None)


def create_arcface_model(
    num_classes: int,
    embedding_size: int = 512,
    backbone: str = "resnet18",
    pretrained: bool = True,
    **kwargs,
) -> FaceRecognitionModel:
    """Factory function to create an ArcFace model.

    Args:
        num_classes: Number of identity classes.
        embedding_size: Embedding dimension (default: 512).
        backbone: ResNet variant (default: "resnet18").
        pretrained: Use ImageNet pretrained weights (default: True).
        **kwargs: Additional arguments passed to FaceRecognitionModel.

    Returns:
        model: FaceRecognitionModel instance.
    """
    return FaceRecognitionModel(
        num_classes=num_classes,
        embedding_size=embedding_size,
        backbone=backbone,  # type: ignore
        pretrained=pretrained,
        **kwargs,
    )


if __name__ == "__main__":
    # Test model instantiation and forward pass
    num_identities = 1000
    batch_size = 8

    model = create_arcface_model(
        num_classes=num_identities,
        embedding_size=512,
        backbone="resnet18",
        pretrained=False,  # Faster for testing
    )

    print(f"Model architecture:\n{model}\n")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # Test training forward pass
    x = torch.randn(batch_size, 3, 112, 112)
    labels = torch.randint(0, num_identities, (batch_size,))

    logits, embeddings = model(x, labels)
    print(f"Training mode:")
    print(f"  Input shape: {x.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Embeddings shape: {embeddings.shape}\n")

    # Test inference forward pass
    embeddings_inf = model(x, labels=None)
    print(f"Inference mode:")
    print(f"  Input shape: {x.shape}")
    print(f"  Embeddings shape: {embeddings_inf.shape}")
    print(f"  Embeddings L2 norm: {torch.norm(embeddings_inf, p=2, dim=1)}")  # Should be ~1.0
