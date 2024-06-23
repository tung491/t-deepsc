import argparse
import os
import pickle
import time

import nltk
import numpy
import torch
import torch.nn.functional as F
import torchinfo
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, BatchSampler, RandomSampler, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

import wandb
from dataset import QADataset, VQADataset, CLEVRDataset
from udeepsc import UDeepSC

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser(description='Train text model')
    parser.add_argument('--saved_path', type=str, default='saved_model', help='Path to save model')
    parser.add_argument('--num_epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument("--snr", type=int, default=12, help="Signal to noise ratio")
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on (cpu or cuda)')
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")

    return parser.parse_args()


class TextDataset(Dataset):
    def __init__(self, split="train"):
        with open(f'{split}_text.pkl', 'rb') as f:
            self.tokenized_tensor = torch.tensor(numpy.array(pickle.load(f)), dtype=torch.long)
            self.tokenizer = tokenizer
            self.num_classes = self.tokenizer.vocab_size
            self.max_sentence_length = self.tokenized_tensor.size(1)
            self.word2id_dict = self.tokenizer.vocab

    def __len__(self):
        return len(self.tokenized_tensor)

    def __getitem__(self, idx):
        items = self.tokenized_tensor[idx]
        # Calculate the length of the sentence (number of non-padding tokens)
        sentence_length = (items != 0).sum().item()
        return items, sentence_length

    def get_sentence_as_word(self, index):
        sentence_as_id, sentence_length = self.__getitem__(index)
        # Convert token IDs to words
        sentence_as_word = self.tokenizer.decode(sentence_as_id[:sentence_length], skip_special_tokens=True)
        return sentence_as_word

    def convert_id_sentence_to_word(self, sentence_as_id, sentence_length):
        # Convert token IDs to words
        sentence_as_word = self.tokenizer.decode(sentence_as_id[:sentence_length], skip_special_tokens=True)
        return sentence_as_word


class LossFn(nn.Module):  # Loss function
    def __init__(self):
        super(LossFn, self).__init__()

    def forward(self, output, label, length_sen, mi_lb=0.0, lambda_n=0.009):
        delta = 1e-7  # used to avoid vanishing gradient
        device = output.device
        # Create a mask for valid lengths
        max_length = output.size(1)
        mask = torch.arange(max_length).expand(len(length_sen), max_length).to(device) < length_sen.unsqueeze(1)

        # Mask the output and label
        output_masked = output[mask]
        label_masked = label[mask]

        # Compute the loss using masked values
        loss_mine = -mi_lb
        loss = -torch.sum(
            label_masked * torch.log(output_masked + delta)) / length_sen.float().sum() + lambda_n * loss_mine
        return loss


class VQALossFn(nn.Module):
    def __init__(self):
        super(VQALossFn, self).__init__()
        self.img_loss = nn.MSELoss()
        self.text_loss = LossFn()

    def forward(self, output, label, length_sen, img_predicted, img_input):
        text_loss = self.text_loss(output, label, length_sen)
        img_loss = self.img_loss(img_predicted, img_input)
        return text_loss + img_loss


def train_step(sem_net, inputs, label, optim, criterion, sentence_length, scaler, multimodal=False):
    sem_net.train()
    optim.zero_grad()
    with autocast():
        if multimodal:
            img, text = inputs
            predicted_text, predicted_img = sem_net(text, img, multi_modal=multimodal)
            loss = criterion(predicted_text, label, sentence_length, predicted_img, img)
        else:
            text = inputs
            predicted_text, _ = sem_net(text, multi_modal=multimodal)
            loss = criterion(predicted_text, label, sentence_length)

    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()
    return loss.item()


@torch.jit.script
def calculate_sem_similarity(bert_input, bert_output):
    return torch.sum(bert_input * bert_output) / (
            torch.sqrt(torch.sum(bert_input * bert_input))
            * torch.sqrt(torch.sum(bert_output * bert_output)))


def calBLEU(n_gram, s_predicted, s, length):
    weights = [1 / n_gram] * n_gram
    BLEU = nltk.translate.bleu_score.sentence_bleu([s[:length]], s_predicted[:length], weights=weights)
    return BLEU


def main():
    args = parse_args()
    run = wandb.init(
        name=f"UDeepSC_training",
        project='UDeepSC', config=vars(args))

    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)

    print("Config:", dict(run.config))
    # Load dataset
    qa_dataset = QADataset(split="train")

    graphcore_vqa_dataset = VQADataset(split="train")
    clevr_dataset = CLEVRDataset(split="train")
    vqa_dataset = torch.utils.data.ConcatDataset([graphcore_vqa_dataset, clevr_dataset])

    vqa_batch_sampler = BatchSampler(RandomSampler(vqa_dataset), batch_size=64, drop_last=False)
    vqa_dataloader = DataLoader(vqa_dataset, batch_sampler=vqa_batch_sampler)

    qa_batch_sampler = BatchSampler(RandomSampler(qa_dataset), batch_size=args.batch_size, drop_last=False)
    qa_dataloader = DataLoader(qa_dataset, batch_sampler=qa_batch_sampler)

    text_loss = LossFn().to(args.device)
    vqa_loss = VQALossFn().to(args.device)
    sem_net = UDeepSC().to(args.device)
    sem_optim = torch.optim.AdamW(list(sem_net.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sem_optim, mode='min', factor=0.1, patience=5,
                                                           verbose=True)
    img, text, sentence_length = next(iter(vqa_dataloader))
    img = img.to(args.device)
    text = text.to(args.device)
    torchinfo.summary(sem_net, input_data=(text, img, True))
    scaler = GradScaler()
    best_loss = float('inf')
    sem_net.train()

    for epoch in range(args.num_epoch):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        mutual_info_total = 0.0
        # vqa_train_bar = tqdm(vqa_dataloader, desc=f'Epoch {epoch + 1}/{args.num_epoch} VQA')
        # for data in vqa_train_bar:
        #     input_img, input_text, sentence_length = data
        #     input_text = input_text.to(args.device)
        #     input_img = input_img.to(args.device)
        #     inputs = (input_img, input_text)
        #     sentence_length = sentence_length.to(args.device)
        #     label = F.one_hot(input_text, num_classes=30522).float().to(args.device)
        #     loss = train_step(sem_net, inputs, label, sem_optim, vqa_loss, sentence_length, scaler, multimodal=True)
        #     epoch_loss += loss
        #     vqa_train_bar.set_postfix(loss=loss)

        qa_train_bar = tqdm(qa_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epoch} QA")
        for data in qa_train_bar:
            input_text, sentence_length = data
            input_text = input_text.to(args.device)
            sentence_length = sentence_length.to(args.device)
            label = F.one_hot(input_text, num_classes=30522).float().to(args.device)
            loss = train_step(sem_net, input_text, label, sem_optim, text_loss, sentence_length, scaler)
            epoch_loss += loss
            qa_train_bar.set_postfix(loss=loss)
        epoch_loss /= (len(vqa_dataloader) + len(qa_dataloader))
        scheduler.step(epoch_loss)

        run.log({'loss': epoch_loss, 'mutual_information': mutual_info_total, 'lr': sem_optim.param_groups[0]['lr'],
                 'epoch_time': time.time() - epoch_start_time})
        torch.save(sem_net.state_dict(), os.path.join(args.saved_path, f'udeepsc_text_model_{epoch}.pth'))
        wandb.save(os.path.join(args.saved_path, f'udeepsc_text_model_{epoch}.pth'))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(sem_net.state_dict(), os.path.join(args.saved_path, 'udeepsc_best_model.pth'))
            wandb.save(os.path.join(args.saved_path, 'udeepsc_best_model.pth'))
    run.finish()


if __name__ == '__main__':
    main()
