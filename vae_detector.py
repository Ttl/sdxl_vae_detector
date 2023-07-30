#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from PIL.ImageOps import exif_transpose
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

device = "cuda"
steps = 20000

if 1:
    validation_vae_path = "validation/sdxl10"
    vae_path = "test/sdxl10"
else:
    validation_vae_path = "validation/sdxl09"
    vae_path = "test/sdxl09"
reg_path = "test/orig"
validation_reg_path = "validation/orig"

class VAEDataset(Dataset):
    def __init__(
        self,
        vae_path,
        reg_path
    ):
        self.vae_path = Path(vae_path)
        self.reg_path = Path(reg_path)

        if not self.vae_path.exists():
            raise ValueError("VAE path does not exist")

        if not self.reg_path.exists():
            raise ValueError("Reg path does not exist")

        self.vae_images_path = list(Path(self.vae_path).iterdir())
        self.num_vae_images = len(self.vae_images_path)
        self.reg_images_path = list(Path(self.reg_path).iterdir())
        self.num_reg_images = len(self.reg_images_path)

        self._length = self.num_vae_images + self.num_reg_images

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        is_vae = index % 2 == 0
        index = index//2
        if is_vae:
            image = Image.open(self.vae_images_path[index % self.num_vae_images])
        else:
            image = Image.open(self.reg_images_path[index % self.num_reg_images])
        image = exif_transpose(image)
        image = image.convert("RGB")
        example["images"] = self.image_transforms(image)
        example["targets"] = is_vae
        return example


def collate_fn(examples):
    pixel_values = [example["images"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    targets = torch.FloatTensor([example["targets"] for example in examples])
    targets = torch.unsqueeze(targets, 1)

    batch = {
        "pixel_values": pixel_values,
        "targets": targets,
    }

    return batch

class VAEDetector(nn.Module):
    def __init__(self, conv_chs=64, linear_chs=16):
        super().__init__()
        self.conv_in = nn.Conv2d(3, conv_chs, 9, padding="same")
        self.linear1 = nn.Linear(conv_chs, linear_chs)
        self.linear2 = nn.Linear(linear_chs, 1)

    def forward(self, x):
        z = self.conv_in(x)
        z = torch.max(torch.max(z, -1)[0], -1)[0]
        z = F.relu(self.linear1(z))
        z = F.sigmoid(self.linear2(z))
        return z

def get_loss(model, images, targets):
    pred = detector(images)
    loss = F.mse_loss(pred, targets)
    # This is probably more correct
    #loss = F.binary_cross_entropy(pred, targets)
    with torch.no_grad():
        correct = torch.sum(torch.round(pred) == targets) / targets.shape[0]
    return loss, correct

if __name__ == "__main__":
    detector = VAEDetector().to(device)
    dataset = VAEDataset(vae_path, reg_path)
    dataloader = DataLoader(dataset,
            batch_size=64,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=12)
    validation_dataset = VAEDataset(validation_vae_path, validation_reg_path)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps//2.2, gamma=0.1)
    params = 0
    for p in detector.parameters():
        params += p.numel()
    print(params, 'Parameters')
    writer = SummaryWriter(comment=vae_path.split('/')[-1])
    detector.train()
    epoch = 0
    step = 0
    progress_bar = tqdm(range(steps))
    progress_bar.set_description("Steps")
    while step < steps:
        epoch += 1
        for batch in dataloader:
            step += 1
            targets = batch["targets"].to(device)
            images = batch["pixel_values"].to(device)
            loss, correct = get_loss(detector, images, targets)
            l = loss.detach().cpu().item()
            c = correct.detach().cpu().item()
            writer.add_scalar('Loss/train', l, step)
            writer.add_scalar('Accuracy/train', c, step)
            progress_bar.set_postfix(loss=round(l,2), correct=round(c,2), lr=scheduler.get_last_lr()[0])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            scheduler.step()
            if step >= steps:
                break

    torch.save(detector.state_dict(), 'sdxl_vae_detector.pt')
    print('Model saved')
    validation_dataloader = DataLoader(validation_dataset,
            batch_size=64,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=12)
    losses = []
    corrects = []
    print('Running validation')
    with torch.inference_mode():
        for batch in validation_dataloader:
            targets = batch["targets"].to(device)
            images = batch["pixel_values"].to(device)
            loss, correct = get_loss(detector, images, targets)
            l = loss.detach().cpu().item()
            c = correct.detach().cpu().item()
            losses.append(l)
            corrects.append(c)
    l = sum(losses) / len(losses)
    c = sum(corrects) / len(corrects)
    print('Validation loss', l, 'correct', c)
    writer.add_scalar('Loss/validation', l, step)
    writer.add_scalar('Accuracy/validation', c, step)
