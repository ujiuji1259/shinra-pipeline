import sys
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

from entity_linking.dataset import my_collate_fn, my_collate_fn_json
from entity_linking.searcher import NearestNeighborSearch
from utils.util import get_scheduler, to_fp16, save_model, to_parallel


class BertCrossEncoder(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.model = bert
        self.linear_layer = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        bertrep, _ = self.model(input_ids, attention_mask=attention_mask)
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
            negative_scores = torch.tensor([s[1] for s in negative_samples])
            negative_idx = torch.multinomial(negative_scores, negatives).tolist()

            pages.extend([negative_samples[n][0] for n in negative_idx])
            output_labels.extend([1]*negatives)

        return pages, output_labels

    def calculate_intraining_accuracy(self, scores):
        idx = torch.argmax(scores, dim=-1).detach().cpu().tolist()
        accuracy = sum([int(i == 0) for i in idx]) / len(idx)
        return accuracy

    def train(self,
              mention_dataset,
              candidate_dataset,
              lr=1e-5,
              batch_size=16,
              max_ctxt_len=32,
              max_title_len=50,
              max_desc_len=100,
              traindata_size=1000000,
              model_save_interval=10000,
              grad_acc_step=1,
              max_grad_norm=1.0,
              epochs=1,
              warmup_propotion=0.1,
              fp16=False,
              fp16_opt_level=None,
              parallel=False,
              negatives=1,
              debug=False,
              tokenizer=None,
             ):


        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = get_scheduler(
            1, grad_acc_step, epochs, warmup_propotion, optimizer, traindata_size)

        loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

        if fp16:
            assert fp16_opt_level is not None
            self.model, optimizer = to_fp16(self.model, optimizer, fp16_opt_level)

        if parallel:
            self.model = to_parallel(self.model)

        for e in range(epochs):
            dataloader = DataLoader(mention_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn_json, num_workers=2)
            bar = tqdm(total=traindata_size)
            for step, (input_ids, labels, lines) in enumerate(dataloader):
                if step * batch_size > traindata_size:
                    break

                if self.logger:
                    self.logger.debug("%s step", step)
                    self.logger.debug("%s data in batch", len(input_ids))
                    self.logger.debug("%s unique labels in %s labels", len(set(labels)), len(labels))

                NNs = [l["nearest_neighbors"] for l in lines]
                scores = [l['similarity'] for l in lines]
                pages, output_label = self.create_training_samples(labels, NNs, scores, negatives)
                # pages = list(labels)
                # for nn in lines[0]["nearest_neighbors"]:
                #     if str(nn) not in pages:
                #         pages.append(str(nn))
                candidate_input_ids = candidate_dataset.get_pages(pages, max_title_len=max_title_len, max_desc_len=max_desc_len)

                inputs = self.merge_mention_candidate(input_ids, candidate_input_ids)

                inputs = pad_sequence([torch.LongTensor(token)
                                        for token in inputs], padding_value=0).t().to(self.device)
                input_mask = inputs > 0

                if debug:
                    cnt = 0
                    original_tokens = [tokenizer.convert_ids_to_tokens(i) for i in inputs]
                    print(original_tokens)
                    while cnt < 1000:
                        cnt += 1
                        scores = self.model(inputs, input_mask).view(-1, negatives+1)
                        print(scores)

                        target = torch.LongTensor([0]*scores.size(0)).to(self.device)
                        loss = F.cross_entropy(scores, target, reduction="mean")
                        #target = torch.tensor(output_label, dtype=float).to(self.device)
                        #loss = loss_fn(scores, target.unsqueeze(1))
                        print(loss)

                        if self.logger:
                            self.logger.debug("Train loss: %s", loss.item())


                        if fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()


                        if (step + 1) % grad_acc_step == 0:
                            if fp16:
                                torch.nn.utils.clip_grad_norm_(
                                    amp.master_params(optimizer), max_grad_norm
                                )
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_grad_norm
                                )
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()

                            if self.logger:
                                self.logger.debug("Back propagation in step %s", step+1)
                                self.logger.debug("LR: %s", scheduler.get_lr())

                scores = self.model(inputs, input_mask).view(-1, negatives+1)
                accuracy = self.calculate_intraining_accuracy(scores)

                target = torch.LongTensor([0]*scores.size(0)).to(self.device)
                loss = F.cross_entropy(scores, target, reduction="mean")
                #target = torch.tensor(output_label, dtype=float).to(self.device)
                #loss = loss_fn(scores, target.unsqueeze(1))

                if self.logger:
                    self.logger.debug("Train loss: %s", loss.item())


                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()


                if (step + 1) % grad_acc_step == 0:
                    if fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if self.logger:
                        self.logger.debug("Back propagation in step %s", step+1)
                        self.logger.debug("LR: %s", scheduler.get_lr())

                if self.use_mlflow:
                    mlflow.log_metric("train loss", loss.item(), step=step)
                    mlflow.log_metric("train accuracy", accuracy, step=step)

                if self.model_path is not None and step % model_save_interval == 0:
                    #torch.save(self.model.state_dict(), self.model_path)
                    save_model(self.model, self.model_path)

                bar.update(len(input_ids))
                bar.set_description(f"Loss: {loss.item()}, Accuracy {accuracy}")

