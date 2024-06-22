import argparse

import numpy
import torch
import os
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Normalize
from tqdm import tqdm

from model import ViSemanticCommunicationSystem

torch.set_float32_matmul_precision('high')

class ViVQADataset(Dataset):
    def __init__(self, split="train", img_size=32):
        self.dataset = load_dataset("Graphcore/vqa", split=split)
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = item['image_id']
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
            image.save(image_path)
        processed_img = self.img_transform(image)
        return processed_img


class ViCLEVRDataset(Dataset):
    def __init__(self, split="train", img_size=32):
        self.image_path = os.path.join("CLEVR_v1.0", "images", split)
        self.img_files = os.listdir(self.image_path)

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        image = Image.open(os.path.join(self.image_path, img_file))
        if image.mode != "RGB":
            image = image.convert("RGB")
            image.save(os.path.join(self.image_path, img_file))
        processed_img = self.img_transform(image)
        return processed_img



def parse_args():
    parser = argparse.ArgumentParser(description='Train text model')
    parser.add_argument('--saved_path', type=str, default='saved_model', help='Path to save model')
    parser.add_argument('--num_epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument("--snr", type=int, default=12, help="Signal to noise ratio")
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on (cpu or cuda)')
    parser.add_argument("--depth", type=int, default=3, help="Depth of the encoder/decoder module")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping norm value")
    parser.add_argument('--embedding_dim', type=int, default=48, help='Embedding dimension')

    return parser.parse_args()


def train_step(net, inputs, optim, criterion, scaler):
    optim.zero_grad()
    with autocast():
        outputs = net(inputs)
        loss = criterion(outputs, inputs)
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()
    return loss


def main():
    args = parse_args()
    # if args.saved_path not existed create it
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)
    run = wandb.init(
        name=f"MambaOutViModel_SNR_{args.snr}_depth_{args.depth}", project='MambaOutViModel', config=vars(args)
    )
    print("Config:", dict(run.config))

    train_vi_vqa_dataset = ViVQADataset(img_size=32)
    train_vi_clevr_dataset = ViCLEVRDataset("train", img_size=32)
    test_vi_vqa_dataset = ViVQADataset(split="test", img_size=32)
    test_vi_clevr_dataset = ViCLEVRDataset("test", img_size=32)
    train_concat_dataset = torch.utils.data.ConcatDataset([train_vi_vqa_dataset, train_vi_clevr_dataset])
    test_concat_dataset = torch.utils.data.ConcatDataset([test_vi_vqa_dataset, test_vi_clevr_dataset])

    vi_model = ViSemanticCommunicationSystem(embed_dim=args.embedding_dim, img_size=32).to(args.device)

    sem_optim = torch.optim.AdamW(list(vi_model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sem_optim, mode='min', factor=0.5, patience=3, verbose=True)

    train_dataloader = DataLoader(train_concat_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_concat_dataset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=4)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
    best_loss = float('inf')
    mse_loss = nn.MSELoss().to('cuda')
    scaler = GradScaler()
    for epoch in range(args.num_epoch):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epoch}")
        ssim_loss = 0.0
        mae_loss = 0.0
        for data in train_bar:
            inputs = data.to(args.device)
            # loss, component_losses = train_step(vi_model, inputs, sem_optim, mse_loss)
            # epoch_loss += loss.item()
            # ssim_loss += component_losses['ssim_loss'].item()
            # mae_loss += component_losses['mse_loss'].item()
            loss = train_step(vi_model, inputs, sem_optim, mse_loss, scaler)
            epoch_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        epoch_loss /= len(train_dataloader)
        ssim_loss /= len(train_dataloader)
        mae_loss /= len(train_dataloader)

        test_epoch_loss = 0.0
        test_psnr = 0.0
        test_ssim = 0.0

        test_bar = tqdm(test_dataloader, desc=f"Test Epoch {epoch + 1}/{args.num_epoch}")
        for data in test_bar:
            inputs = data.to(args.device)
            with torch.no_grad():
                outputs = vi_model(inputs)
            loss = mse_loss(outputs, inputs)
            test_epoch_loss += loss.item()
            test_ssim += ssim(outputs, inputs)
            test_psnr += compute_psnr(inputs, outputs)
        test_epoch_loss /= len(test_dataloader)
        test_psnr /= len(test_dataloader)
        test_ssim /= len(test_dataloader)
        scheduler.step(test_epoch_loss)
        epoch_info = {"epoch_loss": epoch_loss,
                      'epoch_time': time.time() - epoch_start_time,
                      'test_epoch_loss': test_epoch_loss,
                      'test_psnr': test_psnr,
                      'test_ssim': test_ssim}
        print(f"Epoch: {epoch + 1}/{args.num_epoch}: {epoch_info}")
        run.log(epoch_info)
        torch.save(vi_model.state_dict(), os.path.join(args.saved_path, f'vi_model_{epoch}.pth'))
        wandb.save(os.path.join(args.saved_path, f'vi_model_{epoch}.pth'))

        if test_epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(vi_model.state_dict(), os.path.join(args.saved_path, f'best_vi_{args.snr}_model.pth'))
            wandb.save(os.path.join(args.saved_path, f'best_vi_{args.snr}_model.pth'))
    run.finish()
    print("Training completed")


if __name__ == '__main__':
    main()
