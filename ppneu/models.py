"""Backbones (all ImageNet-pretrained via timm) with a single-logit head."""
from __future__ import annotations

# short name -> timm model id with pretrained tag
ARCHS = {
    # the thesis trio
    "resnet50": "resnet50.tv_in1k",
    "densenet121": "densenet121.tv_in1k",
    "inception_v3": "inception_v3.tv_in1k",
    # deployable models for the on-device study
    "mobilenetv3": "mobilenetv3_large_100.ra_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
}


def create(arch: str, pretrained: bool = True):
    """Return (model, mean, std). The model outputs one logit (pneumonia)."""
    import timm

    model = timm.create_model(ARCHS[arch], pretrained=pretrained, num_classes=1)
    cfg = timm.data.resolve_data_config({}, model=model)
    return model, tuple(cfg["mean"]), tuple(cfg["std"])
