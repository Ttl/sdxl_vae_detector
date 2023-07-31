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
import argparse

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
    def __init__(self, conv_chs=64, linear_chs=16, linear2_chs=8):
        super().__init__()
        self.patches = 8
        self.conv_in = nn.Conv2d(3, conv_chs, 9, padding="valid")
        self.linear1 = nn.Linear(conv_chs, linear_chs)
        self.linear2 = nn.Linear(linear_chs, 1)
        self.linear3 = nn.Linear(4, linear2_chs)
        self.linear4 = nn.Linear(linear2_chs, 1)

    def forward(self, z):
        z = self.conv_in(z)
        # Split conv output into patches.
        kernel_size_w = z.shape[2] // self.patches
        kernel_size_h = z.shape[3] // self.patches
        z = z.unfold(3, kernel_size_h, kernel_size_h).unfold(2, kernel_size_w, kernel_size_w)
        # Max of every patch.
        z = torch.max(torch.max(z, -1)[0], -1)[0]
        z = z.view(*z.shape[:-2], -1)
        z = torch.transpose(z, 1, 2)
        # Apply same linear net to every patch.
        z = F.relu(self.linear1(z))
        z = self.linear2(z)
        # Calculate mean, max, min and RMS of patches.
        m1 = torch.mean(z, 1)
        m2 = torch.max(z, 1)[0]
        m3 = torch.min(z, 1)[0]
        m4 = torch.sqrt(torch.mean(torch.square(z), 1))
        # Linear net to calculate prediction from previous stats.
        z = torch.cat([m1, m2, m3, m4], -1)
        z = F.relu(self.linear3(z))
        z = F.sigmoid(self.linear4(z))
        return z

def get_loss(model, images, targets):
    pred = detector(images)
    loss = F.binary_cross_entropy(pred, targets)
    with torch.no_grad():
        correct = torch.sum(torch.round(pred) == targets) / targets.shape[0]
    return loss, correct

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE artifact detector trainer.")
    parser.add_argument(
        "--train_path",
        type=str,
        default=None,
        required=True,
        help="Training data path for VAE processed images",
    )
    parser.add_argument(
        "--reg_path",
        type=str,
        default=None,
        required=True,
        help="Training data path for non processed images",
    )
    parser.add_argument(
        "--validation_path",
        type=str,
        default=None,
        required=False,
        help="Validation data path for VAE processed images",
    )
    parser.add_argument(
        "--validation_reg_path",
        type=str,
        default=None,
        required=False,
        help="Validation data path for non processed images",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="sdxl_vae_detector.pt",
        required=False,
        help="Output filename",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20000,
        help="Number of steps to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="CPU workers",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    args = parser.parse_args()
    detector = VAEDetector().to(args.device)
    dataset = VAEDataset(args.train_path, args.reg_path)
    dataloader = DataLoader(dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers)
    if args.validation_path is not None:
        validation_dataset = VAEDataset(args.validation_path, args.validation_reg_path)
    else:
        validation_dataset = None
    optimizer = torch.optim.AdamW(detector.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.steps//2.2, gamma=0.1)
    params = 0
    for p in detector.parameters():
        params += p.numel()
    print(params, 'Parameters')
    writer = SummaryWriter(comment=args.train_path.split('/')[-1])
    detector.train()
    epoch = 0
    step = 0
    progress_bar = tqdm(range(args.steps))
    progress_bar.set_description("Steps")
    while step < args.steps:
        epoch += 1
        for batch in dataloader:
            step += 1
            targets = batch["targets"].to(args.device)
            images = batch["pixel_values"].to(args.device)
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
            if step >= args.steps:
                break

    torch.save(detector.state_dict(), args.output_filename)
    print('Model saved')
    if validation_dataset is not None:
        validation_dataloader = DataLoader(validation_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=args.num_workers)
        losses = []
        corrects = []
        print('Running validation')
        with torch.inference_mode():
            for batch in validation_dataloader:
                targets = batch["targets"].to(args.device)
                images = batch["pixel_values"].to(args.device)
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
