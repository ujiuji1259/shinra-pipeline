import sys
import os
sys.path.append('../')
import json

import mlflow
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import apex
from apex import amp
import numpy as np

from dataset import my_collate_fn, my_collate_fn_json
from searcher import NearestNeighborSearch
from utils.util import get_scheduler, to_fp16, save_model, to_parallel

class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.NLLLoss()

    # `x`と`y`はそれぞれモデル出力の値とGTの値
    def forward(self, x, y, eps=1e-24):
        x = torch.softmax(x, dim=1)
        # (BATCH, C, H, W) から (C, BATCH * H * W) に並べ直す
        #NUM_CLASSES = x.size(1)
        #x = x.contiguous().view(-1, NUM_CLASSES)
        # 微小値を足してからlogる※1
        x = torch.log(x + eps)

        #y = y.contiguous().view(-1, NUM_CLASSES)
        #_, y = torch.max(y, -1)
        return self.loss_fn(x, y)


class BertCrossEncoder(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.model = bert
        self.linear_layer = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        bertrep = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        bertrep = bertrep[:, 0, :]
        output = self.linear_layer(bertrep)

        return output

class BertCandidateRanker(object):
    def __init__(self, cross_encoder, device="cpu", model_path=None, use_mlflow=False, logger=None):
        self.model = cross_encoder.to(device)
        self.device = device

        self.model_path = model_path
        self.use_mlflow = use_mlflow
        self.logger = logger

    def predict(self,
        input_ids,
        pages,
        candidate_dataset,
        max_title_len=50,
        max_desc_len=100,
        ):

        overall_scores = []
        overall_tokens = []

        with torch.no_grad():
            for input_id, page in zip(input_ids, pages):
                candidate_input_ids = candidate_dataset.get_pages(page, max_title_len=max_title_len, max_desc_len=max_desc_len)
                inputs = self.merge_mention_candidate([input_id], candidate_input_ids)
                overall_tokens.extend(inputs)
                inputs = pad_sequence([torch.LongTensor(token)
                                        for token in inputs], padding_value=0).t().to(self.device)
                input_mask = inputs > 0
                scores = self.model(inputs, input_mask).view(-1).detach().cpu().tolist()
                overall_scores.append(scores)

        return overall_scores, overall_tokens

    def merge_mention_candidate(self, mention_ids, candidate_ids):
        page_per_cand = len(candidate_ids) // len(mention_ids)

        mention_ids = [[mention_id]*page_per_cand for mention_id in mention_ids]
        mention_ids = [m for mm in mention_ids for m in mm]
        results = [mention_id + candidate_id[1:] for candidate_id, mention_id in zip(candidate_ids, mention_ids)]
        return results

    def create_training_samples(self, labels, NNs, scores, negatives=1):
        pages = []
        output_labels = []
        for label, NN, score in zip(labels, NNs, scores):
            pages.append(label)
            output_labels.append(1)

            if isinstance(NN[0], int):
                NN = [str(n) for n in NN]

            negative_samples = [[n, s] for n, s in zip(NN, score) if n != label]

            #negative_scores = torch.tensor([s[1] for s in negative_samples])
            #negative_idx = torch.multinomial(negative_scores, negatives).tolist()
            negative_idx = list(range(len(negative_samples)))

            pages.extend([negative_samples[n][0] for n in negative_idx])
            output_labels.extend([0]*negatives)

        return pages[:negatives], output_labels[:negatives]

    def calculate_intraining_accuracy(self, scores):
        idx = torch.argmax(scores, dim=-1).detach().cpu().tolist()
        accuracy = sum([int(i == 0) for i in idx]) / len(idx)
        return accuracy

    def train(self,
            mention_dataset,
            candidate_dataset,
            tokenizer=None,
            args=None,
        ):


        optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        #loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        loss_fn = CrossEntropyLoss2d()

        if args.fp16:
            assert args.fp16_opt_level is not None
            self.model, optimizer = to_fp16(self.model, optimizer, args.fp16_opt_level)

        if args.parallel:
            self.model = to_parallel(self.model)

        for e in range(args.epochs):
            dataloader = DataLoader(mention_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn_json, num_workers=os.cpu_count())
            bar = tqdm(total=args.traindata_size)
            for step, (input_ids, labels, lines) in enumerate(dataloader):
                if self.logger:
                    self.logger.debug("%s step", step)
                    self.logger.debug("%s data in batch", len(input_ids))
                    self.logger.debug("%s unique labels in %s labels", len(set(labels)), len(labels))

                NNs = [l["nearest_neighbors"] for l in lines]
                scores = [l['similarity'] for l in lines]
                pages, output_label = self.create_training_samples(labels, NNs, scores, args.negatives)
                # pages = list(labels)
                # for nn in lines[0]["nearest_neighbors"]:
                #     if str(nn) not in pages:
                #         pages.append(str(nn))
                candidate_input_ids = candidate_dataset.get_pages(pages, max_title_len=args.max_title_len, max_desc_len=args.max_desc_len)

                inputs = self.merge_mention_candidate(input_ids, candidate_input_ids)

                inputs = pad_sequence([torch.LongTensor(token)
                                        for token in inputs], padding_value=0).t().to(self.device)
                input_mask = inputs > 0

                scores = self.model(inputs, input_mask).view(-1, args.negatives)
                accuracy = self.calculate_intraining_accuracy(scores)

                target = torch.LongTensor([0]*scores.size(0)).to(self.device)
                #loss = F.cross_entropy(scores, target, reduction="mean")
                loss = loss_fn(scores, target)
                if loss.item() < 1e-24:
                    continue
                #target = torch.tensor(output_label, dtype=float).to(self.device)
                #loss = loss_fn(scores, target.unsqueeze(1))
                if torch.isnan(loss):
                    print(inputs)
                    print(scores)
                    save_model(self.model, self.model_path)

                if self.logger:
                    self.logger.debug("Train loss: %s", loss.item())


                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()


                if (step + 1) % args.grad_acc_step == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), args.max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), args.max_grad_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad()

                    if self.logger:
                        self.logger.debug("Back propagation in step %s", step+1)

                if self.use_mlflow:
                    mlflow.log_metric("train loss", loss.item(), step=step)
                    mlflow.log_metric("train accuracy", accuracy, step=step)

                if self.model_path is not None and step % args.model_save_interval == 0:
                    #torch.save(self.model.state_dict(), self.model_path)
                    save_model(self.model, self.model_path)

                bar.update(len(input_ids))
                bar.set_description(f"Loss: {loss.item()}, Accuracy {accuracy}")


