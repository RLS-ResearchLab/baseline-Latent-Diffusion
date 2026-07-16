import torch
from torchvision import datasets, transforms


def get_dataloader(config):
    img_size = config["img_size"]
    batch_size = config["batch_size"]
    root = config.get("data_root")

    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet mean & std
    ])

    if root:
        # expects images laid out as root/<class>/*.JPEG (torchvision ImageFolder format)
        dataset = datasets.ImageFolder(root, transform=tfm)

    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=config.get("num_workers", 4), 
        drop_last=True
    )