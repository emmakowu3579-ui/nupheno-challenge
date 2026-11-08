"""
Unit tests for PanNuke COCO Dataset loader.
"""

import json
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

src_file = Path(__file__).resolve().parent.parent / "src" / "dataset.py"
spec = importlib.util.spec_from_file_location("pannuke_dataset_mod", src_file)
pannuke_ds_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pannuke_ds_mod)
PanNukeCOCODataset = pannuke_ds_mod.PanNukeCOCODataset


@pytest.fixture
def mock_dataset_dir(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create dummy image
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    img.save(images_dir / "test_01.png")

    coco_dict = {
        "images": [{"id": 1, "file_name": "test_01.png", "width": 256, "height": 256}],
        "annotations": [
            {
                "id": 101,
                "image_id": 1,
                "category_id": 1,
                "bbox": [50.0, 50.0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0
            }
        ],
        "categories": [{"id": 1, "name": "Neoplastic"}]
    }

    ann_path = tmp_path / "annotations.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f)

    return images_dir, ann_path


def test_dataset_loading(mock_dataset_dir):
    images_dir, ann_path = mock_dataset_dir
    ds = PanNukeCOCODataset(images_dir=str(images_dir), annotations_file=str(ann_path))

    assert len(ds) == 1
    img_tensor, target = ds[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 256, 256)
    assert "boxes" in target
    assert "labels" in target
    assert target["boxes"].shape == (1, 4)
    assert torch.allclose(target["boxes"][0], torch.tensor([50.0, 50.0, 70.0, 70.0]))
    assert target["labels"][0].item() == 1
