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
from utils.util import get_scheduler, to_fp16, save_model, to_parallel, calculate_recall


class BertBiEncoder(nn.Module):
    def __init__(self, mention_bert, candidate_bert):
        super().__init__()
        self.mention_bert = mention_bert
        self.candidate_bert = candidate_bert

    def forward(self, input_ids, attention_mask, is_mention=True, shard_bsz=None):
        if is_mention:
            model = self.mention_bert
        else:
            model = self.candidate_bert

        if shard_bsz is None:
            bertrep, _ = model(input_ids, attention_mask=attention_mask)
            bertrep = bertrep[:, 0, :]
        return bertrep


class BertCandidateGenerator(object):
    def __init__(self, biencoder, device="cpu", model_path=None, use_mlflow=False, builder_gpu=False, logger=None):
        self.model = biencoder.to(device)
        self.device = device
        self.searcher = NearestNeighborSearch(768, use_gpu=builder_gpu)

        self.model_path = model_path
        self.use_mlflow = use_mlflow
        self.logger = logger

    def save_index(self, save_dir):
        self.searcher.save_index(save_dir)

    def load_index(self, save_dir):
        self.searcher.load_index(save_dir)

    def generate_candidates(self, mention_dataset):
        overall_input_ids = []
        results = []
        overall_scores = []
        trues = []

        dataloader = DataLoader(mention_dataset, batch_size=256, collate_fn=my_collate_fn, num_workers=2)
        with torch.no_grad():
            for input_ids, labels in tqdm(dataloader):
                if self.logger:
                    self.logger.debug("%s step", step)
                    self.logger.debug("%s data in batch", len(input_ids))
                    self.logger.debug("%s unique labels in %s labels", len(set(labels)), len(labels))

                inputs = pad_sequence([torch.LongTensor(token)
                                for token in input_ids], padding_value=0).t().to(self.device)
                input_mask = inputs > 0
                mention_reps = self.model(inputs, input_mask, is_mention=True).detach().cpu().numpy()

                preds, scores = self.searcher.search(mention_reps, 100)
                results.extend(preds)
                trues.extend(labels)
                overall_scores.extend(scores.tolist())
                overall_input_ids.extend(input_ids)
        return results, overall_scores, trues, overall_input_ids

    def evaluate(self, mention_dataset):
        results = []
        trues = []
        all_scores = []

        dataloader = DataLoader(mention_dataset, batch_size=256, collate_fn=my_collate_fn, num_workers=2)
        with torch.no_grad():
            for input_ids, labels in tqdm(dataloader):
                if self.logger:
                    self.logger.debug("%s step", step)
                    self.logger.debug("%s data in batch", len(input_ids))
                    self.logger.debug("%s unique labels in %s labels", len(set(labels)), len(labels))

                inputs = pad_sequence([torch.LongTensor(token)
                                for token in input_ids], padding_value=0).t().to(self.device)
                input_mask = inputs > 0
                mention_reps = self.model(inputs, input_mask, is_mention=True).detach().cpu().numpy()

                preds, scores = self.searcher.search(mention_reps, 100)
                results.extend(preds)
                trues.extend(labels)
                all_scores.extend(scores)
        recall = calculate_recall(trues, results)
        return recall, results, all_scores

    def build_searcher(self,
                       candidate_dataset,
                       max_title_len=50,
                       max_desc_len=100):
        page_ids = list(candidate_dataset.data.keys())
        batch_size = 512

        with torch.no_grad():
            for start in tqdm(range(0, len(page_ids), batch_size)):
                end = min(start+batch_size, len(page_ids))
                pages = page_ids[start:end]
                input_ids = candidate_dataset.get_pages(
                    pages,
                    max_title_len=max_title_len,
                    max_desc_len=max_desc_len,
                )

                inputs = pad_sequence([torch.LongTensor(token)
                                      for token in input_ids], padding_value=0).t().to(self.device)
                masks = inputs > 0
                reps = self.model(inputs, masks, is_mention=False)
                reps = reps.detach().cpu().numpy()

                self.searcher.add_entries(reps, pages)

    def save_traindata_with_negative_samples(self,
          mention_dataset,
          output_file,
          index_output_file,
          batch_size=32,
          max_ctxt_len=32,
          max_title_len=50,
          max_desc_len=100,
          traindata_size=1000000,
          NNs=64,
         ):
        dataloader = DataLoader(mention_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn_json, num_workers=2)

        bar = tqdm(total=traindata_size)
        total = 0
        trues = 0
        index = [0]
        steps = 0
        with open(output_file, 'w') as fout:
            with torch.no_grad():
                for input_ids, labels, lines in dataloader:
                    inputs = pad_sequence([torch.LongTensor(token)
                                          for token in input_ids], padding_value=0).t().to(self.device)
                    input_mask = inputs > 0

                    mention_reps = self.model(inputs, input_mask, is_mention=True).detach().cpu().numpy()

                    candidates_pageids, sims = self.searcher.search(mention_reps, 100)

                    for i in range(len(lines)):
                        lines[i]['nearest_neighbors'] = candidates_pageids[i]
                        lines[i]['similarity'] = sims[i].tolist()
                        output = json.dumps(lines[i]) + '\n'
                        fout.write(output)
                        index.append(index[-1] + len(output))

                        total += 1
                        trues += int(lines[i]['linkpage_id'] in lines[i]['nearest_neighbors'])

                    bar.update(len(lines))
                    bar.set_description(f"Recall@10: {trues/total}")

                    steps += len(lines)
                    if steps >= traindata_size:
                        break
        index = np.array(index)
        np.save(index_output_file, index)

    def train_hard_negative(self,
          mention_dataset,
          candidate_dataset,
          lr=1e-5,
          batch_size=32,
          random_bsz=100000,
          max_ctxt_len=32,
          max_title_len=50,
          max_desc_len=100,
          traindata_size=1000000,
        ):

        mention_batch = mention_dataset.batch(batch_size=batch_size, random_bsz=random_bsz, max_ctxt_len=max_ctxt_len, return_json=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        all_loss = []
        steps = 0
        bar = tqdm(total=traindata_size)
        for input_ids, labels, lines in mention_batch:
            inputs = pad_sequence([torch.LongTensor(token)
                                  for token in input_ids], padding_value=0).t().to(self.device)
            input_mask = inputs > 0

            mention_reps = self.model(inputs, input_mask, is_mention=True).detach().cpu().numpy()

            candidate_input_ids = candidate_dataset.get_pages(labels, max_title_len=max_title_len, max_desc_len=max_desc_len)
            candidate_inputs = pad_sequence([torch.LongTensor(token)
                                            for token in candidate_input_ids], padding_value=0).t().to(self.device)
            candidate_mask = candidate_inputs > 0
            candidate_reps = self.model(candidate_inputs, candidate_mask, is_mention=False)

    def calculate_inbatch_accuracy(self, scores):
        preds = torch.argmax(scores, dim=1).tolist()
        result = sum([int(i == p) for i, p in enumerate(preds)])
        return result / scores.size(0)

    def train(self,
              mention_dataset,
              candidate_dataset,
              inbatch=True,
              lr=1e-5,
              batch_size=32,
              random_bsz=100000,
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
              hard_negative=False,
             ):


        if inbatch:

            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            scheduler = get_scheduler(
                batch_size, grad_acc_step, epochs, warmup_propotion, optimizer, traindata_size)

            if fp16:
                assert fp16_opt_level is not None
                self.model, optimizer = to_fp16(self.model, optimizer, fp16_opt_level)

            if parallel:
                self.model = to_parallel(self.model)

            for e in range(epochs):
                #mention_batch = mention_dataset.batch(batch_size=batch_size, random_bsz=random_bsz, max_ctxt_len=max_ctxt_len)
                dataloader = DataLoader(mention_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn_json, num_workers=2)
                bar = tqdm(total=traindata_size)
                #for step, (input_ids, labels) in enumerate(mention_batch):
                for step, (input_ids, labels, lines) in enumerate(dataloader):
                    if self.logger:
                        self.logger.debug("%s step", step)
                        self.logger.debug("%s data in batch", len(input_ids))
                        self.logger.debug("%s unique labels in %s labels", len(set(labels)), len(labels))

                    inputs = pad_sequence([torch.LongTensor(token)
                                          for token in input_ids], padding_value=0).t().to(self.device)
                    input_mask = inputs > 0

                    mention_reps = self.model(inputs, input_mask, is_mention=True)

                    pages = list(labels[:])
                    if hard_negative:
                        for label, line in zip(labels, lines):
                            for i in line["nearest_neighbors"]:
                                if str(i) == label:
                                    break
                                pages.append(str(i))

                    candidate_input_ids = candidate_dataset.get_pages(pages, max_title_len=max_title_len, max_desc_len=max_desc_len)
                    candidate_inputs = pad_sequence([torch.LongTensor(token)
                                                    for token in candidate_input_ids], padding_value=0).t().to(self.device)
                    candidate_mask = candidate_inputs > 0
                    candidate_reps = self.model(candidate_inputs, candidate_mask, is_mention=False)

                    scores = mention_reps.mm(candidate_reps.t())
                    accuracy = self.calculate_inbatch_accuracy(scores)

                    target = torch.LongTensor(torch.arange(scores.size(0))).to(self.device)
                    loss = F.cross_entropy(scores, target, reduction="mean")

                    if self.logger:
                        self.logger.debug("Accurac: %s", accuracy)
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
                        mlflow.log_metric("accuracy", accuracy, step=step)

                    if self.model_path is not None and step % model_save_interval == 0:
                        #torch.save(self.model.state_dict(), self.model_path)
                        save_model(self.model, self.model_path)

                    bar.update(len(input_ids))
                    bar.set_description(f"Loss: {loss.item()}, Accuracy: {accuracy}")


