import argparse
import os
import pickle
import time

import numpy
import nltk
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from monai.data import DataLoader
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, BatchSampler, RandomSampler
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from model import TextSemanticCommunicationSystem

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


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


def train_step(sem_net, inputs, label, optim, criterion, sentence_length, scaler):
    sem_net.train()
    optim.zero_grad()
    with autocast():
        s_predicted = sem_net(inputs)
        loss = criterion(s_predicted, label, sentence_length)

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
        name=f"Text_T-DeepSC_{args.snr}_depth_{args.depth}_embed_{args.embedding_dim}",
        project='MambaTextModel', config=vars(args))

    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)

    print("Config:", dict(run.config))
    # Load dataset
    # corpus_data = CorpusData(args.data_path)
    # dataloader = DataLoader(corpus_data, batch_size=args.batch_size, shuffle=True)
    dataset = TextDataset(split="train")
    batch_sampler = BatchSampler(RandomSampler(dataset), batch_size=args.batch_size, drop_last=False)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    loss_fn = LossFn().to(args.device)
    # loss_fn = TextLossFn(0).to(args.device)
    sem_net = TextSemanticCommunicationSystem(input_size=30522, output_size=args.embed_dim).to(args.device)
    # channel_mi_estimator = MutualInformationEstimator(input_size=2, hidden_size=10).to(args.device)
    # channel_mi_estimator = torch.compile(channel_mi_estimator, mode='reduce-overhead')
    sem_optim = torch.optim.AdamW(list(sem_net.parameters()), lr=args.lr)
    # mi_optim = torch.optim.AdamW(list(channel_mi_estimator.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sem_optim, mode='min', factor=0.1, patience=args.patience,
                                                           verbose=True)
    scaler = GradScaler()
    best_loss = float('inf')
    sem_net.train()

    for epoch in range(args.num_epoch):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        mutual_info_total = 0.0
        train_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{args.num_epoch}')
        for data in train_bar:
            inputs, sentence_length = data
            inputs = inputs.to(args.device)
            sentence_length = sentence_length.to(args.device)
            label = F.one_hot(inputs, num_classes=30522).float().to(args.device)
            # mi_lb = train_mi(sem_net, channel_mi_estimator, inputs, mi_optim, scaler)
            loss = train_step(sem_net, inputs, label, sem_optim, loss_fn, sentence_length, scaler)
            epoch_loss += loss
            # mutual_info_total += mi_lb.item()
            # train_bar.set_postfix(loss=loss, mutual_info=mi_lb.item())
            train_bar.set_postfix(loss=loss)
        epoch_loss /= len(dataloader)
        mutual_info_total /= len(dataloader)
        scheduler.step(epoch_loss)

        run.log({'loss': epoch_loss, 'mutual_information': mutual_info_total, 'lr': sem_optim.param_groups[0]['lr'],
                 'epoch_time': time.time() - epoch_start_time})
        torch.save(sem_net.state_dict(), os.path.join(args.saved_path, f'tdeepsc_text_model_{epoch}.pth'))
        wandb.save(os.path.join(args.saved_path, f'tdeepsc_text_model_{epoch}.pth'))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(sem_net.state_dict(), os.path.join(args.saved_path, 'tdeepsc_best_model.pth'))
            wandb.save(os.path.join(args.saved_path, 'tdeepsc_best_model.pth'))
    run.finish()

    # Start eval run
    # Load the best model
    sem_net.load_state_dict(torch.load(os.path.join(args.saved_path, 'tdeepsc_best_model.pth')))
    corpus_data = TextDataset(split='test')
    batch_sampler = BatchSampler(RandomSampler(corpus_data), batch_size=args.batch_size, drop_last=False)
    dataloader = DataLoader(corpus_data, batch_sampler=batch_sampler)
    BLEU_scores = {1: [], 2: [], 3: [], 4: []}
    semantic_similarity_scores = []
    inference_latencies = []
    BLEU_1_SS_per_testSNR = []
    BLEU_2_SS_per_testSNR = []
    BLEU_3_SS_per_testSNR = []
    BLEU_4_SS_per_testSNR = []
    SS_per_testSNR = []

    tokenizer = BertTokenizer.from_pretrained('sentence-transformers/msmarco-bert-base-dot-v5')
    bert_model = BertModel.from_pretrained('sentence-transformers/msmarco-bert-base-dot-v5').to(args.device)
    test_snr_range = np.arange(-6, 19, 3)
    for test_snr in test_snr_range:
        print("Evaluating model with SNR:", test_snr)

        sem_net.set_snr(test_snr)

        BLEU_1_list = []
        BLEU_2_list = []
        BLEU_3_list = []
        BLEU_4_list = []
        semantic_similarity_list = []

        train_bar = tqdm(dataloader)
        for batch_idx, data in enumerate(train_bar):
            # if batch_idx >= 8:
            #     break

            sentence_list = []
            sentence_length_list = []

            inputs, sentence_lenths = data
            for i in range(len(inputs)):
                sentence_ids = inputs[i]
                sentence_length = sentence_lenths[i]
                sentence = corpus_data.convert_id_sentence_to_word(sentence_ids, sentence_length)
                sentence_list.append(sentence)
                sentence_length_list.append(sentence_length)
            inputs = inputs.to(args.device)

            s_predicted = sem_net(inputs)
            s_predicted = torch.argmax(s_predicted, dim=2)

            for i in range(len(inputs)):
                sentence = sentence_list[i]
                sentence_length = sentence_length_list[i]

                output_as_id = s_predicted[i, :]  # get the id list of most possible word
                origin_sentence_as_id = inputs[i, :]

                BLEU1 = calBLEU(1, output_as_id.cpu().detach().numpy(), origin_sentence_as_id.cpu().detach().numpy(),
                                sentence_length)
                BLEU_1_list.append(BLEU1)

                if sentence_length >= 2:
                    BLEU2 = calBLEU(2, output_as_id.cpu().detach().numpy(),
                                    origin_sentence_as_id.cpu().detach().numpy(), sentence_length)
                    BLEU_2_list.append(BLEU2)

                    if sentence_length >= 3:
                        BLEU3 = calBLEU(3, output_as_id.cpu().detach().numpy(),
                                        origin_sentence_as_id.cpu().detach().numpy(), sentence_length)
                        BLEU_3_list.append(BLEU3)

                        if sentence_length >= 4:
                            BLEU4 = calBLEU(4, output_as_id.cpu().detach().numpy(),
                                            origin_sentence_as_id.cpu().detach().numpy(),
                                            sentence_length)  # calculate BLEU
                            BLEU_4_list.append(BLEU4)

                output_sentence = corpus_data.convert_id_sentence_to_word(sentence_as_id=output_as_id,
                                                                          sentence_length=sentence_length)
                encoded_input = tokenizer(sentence, return_tensors='pt').to(args.device)  # encode sentence to fit bert model
                bert_input = bert_model(**encoded_input).pooler_output  # get semantic meaning of the sentence
                encoded_output = tokenizer(output_sentence, return_tensors='pt').to(args.device)
                bert_output = bert_model(**encoded_output).pooler_output
                semantic_similarity = calculate_sem_similarity(bert_input, bert_output)
                semantic_similarity_list.append(semantic_similarity.cpu().detach().numpy())

        avg_BLEU_1 = np.mean(BLEU_1_list)
        avg_BLEU_2 = np.mean(BLEU_2_list)
        avg_BLEU_3 = np.mean(BLEU_3_list)
        avg_BLEU_4 = np.mean(BLEU_4_list)
        avg_SS = np.mean(semantic_similarity_list)

        BLEU_1_SS_per_testSNR.append(avg_BLEU_1)
        BLEU_2_SS_per_testSNR.append(avg_BLEU_2)
        BLEU_3_SS_per_testSNR.append(avg_BLEU_3)
        BLEU_4_SS_per_testSNR.append(avg_BLEU_4)
        SS_per_testSNR.append(avg_SS)

        print("Result of SNR:", test_snr)
        print('BLEU 1 = {}'.format(avg_BLEU_1))
        print('BLEU 2 = {}'.format(avg_BLEU_2))
        print('BLEU 3 = {}'.format(avg_BLEU_3))
        print('BLEU 4 = {}'.format(avg_BLEU_4))
        print('Semantic Similarity = {}'.format(avg_SS))
        run.log(
            {'BLEU_1': avg_BLEU_1, 'BLEU_2': avg_BLEU_2, 'BLEU_3': avg_BLEU_3, 'BLEU_4': avg_BLEU_4, "SNR": test_snr})





if __name__ == '__main__':
    main()
