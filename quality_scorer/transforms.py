from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CHANNEL_JITTER = {
    "normal_map": None,
    "roughness": transforms.ColorJitter(brightness=0.2, contrast=0.2),
    "metallic": transforms.ColorJitter(brightness=0.2, contrast=0.2),
    "base_color": transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1,
    ),
}


def get_transforms(is_train: bool, channel: str):
    if not is_train:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    aug = [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ]
    jitter = CHANNEL_JITTER.get(channel)
    if jitter is not None:
        aug.append(jitter)
    aug.extend(
        [
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ]
    )
    return transforms.Compose(aug)
