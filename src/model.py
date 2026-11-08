"""
Model architecture builder for PanNuke Object Detection baseline.
Uses torchvision Faster R-CNN with ResNet50-FPN backbone customized for 5 cell classes + background.
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn_model(
    num_classes: int = 6,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3
) -> torchvision.models.detection.FasterRCNN:
    """
    Constructs a Faster R-CNN model with ResNet-50-FPN backbone.

    Args:
        num_classes: Total number of classes (5 cell types + 1 background = 6).
        pretrained_backbone: Whether to use pre-trained COCO/ImageNet weights.
        trainable_backbone_layers: Number of trainable ResNet layers (0 to 5).

    Returns:
        torchvision FasterRCNN model.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None

    # Instantiate model
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers
    )

    # Get input feature dimension for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one for PanNuke classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


if __name__ == "__main__":
    test_model = build_faster_rcnn_model(num_classes=6, pretrained_backbone=False)
    dummy_input = [torch.rand(3, 256, 256)]
    test_model.eval()
    with torch.no_grad():
        output = test_model(dummy_input)
    print("Model successfully constructed. Dummy output keys:", output[0].keys())
