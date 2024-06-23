import itertools
import json
import os
import pickle

import numpy
import torch
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Normalize


class VQADataset(Dataset):
    def __init__(self, split="train", img_size=32):
        self.dataset = load_dataset("Graphcore/vqa", split=split)
        with open(f"dataset/vqa/{split}.pkl", 'rb') as f:
            self.tokenized_text = pickle.load(f)

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
        image = self.img_transform(image)
        tokenized_ques = self.tokenized_text[idx]
        sentence_length = (tokenized_ques != 0).sum().item()
        return image, tokenized_ques, sentence_length


class CLEVRDataset(Dataset):
    def __init__(self, split="train", img_size=32):
        with open(f"dataset/clevr/{split}.pkl", 'rb') as f:
            self.tokenized_text = pickle.load(f)
        self.image_path = os.path.join("/home/tung491/lightweight_deepsc/CLEVR_v1.0", "images", split)
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
        tokenized_ques = self.tokenized_text[idx]
        sentence_length = (tokenized_ques != 0).sum().item()
        return processed_img, tokenized_ques, sentence_length


class QADataset(Dataset):
    def __init__(self, split="train"):
        with open(f'dataset/qa_{split}.pkl', 'rb') as f:
            self.tokenized_text = pickle.load(f)

    def __len__(self):
        return len(self.tokenized_text)

    def __getitem__(self, idx):
        tokenized_ques = self.tokenized_text[idx]
        sentence_length = (tokenized_ques != 0).sum().item()
        return tokenized_ques, sentence_length
