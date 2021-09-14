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

from dataset import my_collate_fn, my_collate_fn_json, MentionDataset
from searcher import NearestNeighborSearch
from utils.util import get_scheduler, to_fp16, save_model, to_parallel
from utils.util import calculate_recall


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
            bertrep = model(input_ids, attention_mask=attention_mask).last_hidden_state
            bertrep = bertrep[:, 0, :]
        return bertrep


class BertCandidateGenerator(object):
    def __init__(self, biencoder, device="cpu", model_path=None, use_mlflow=False, logger=None):
        self.model = biencoder.to(device)
        self.device = device

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
                       builder_gpu=False,
                       faiss_gpu_id=None,
                       max_title_len=50,
                       max_desc_len=100):
        page_ids = list(candidate_dataset.data.keys())
        batch_size = 1024

        self.searcher = NearestNeighborSearch(768, len(page_ids), use_gpu=builder_gpu, gpu_id=faiss_gpu_id)

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
        self.searcher.finish_add_entry()

    def save_traindata_with_negative_samples(self,
          mention_dataset,
          output_file,
          index_output_file,
          batch_size=32,
          NNs=100,
          traindata_size=1000000,
         ):
        dataloader = DataLoader(mention_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn_json, num_workers=2)

        bar = tqdm(total=traindata_size)
        total = 0
        trues = 0
        index = [0]
        with open(output_file, 'w') as fout:
            with torch.no_grad():
                for input_ids, labels, lines in dataloader:
                    inputs = pad_sequence([torch.LongTensor(token)
                                          for token in input_ids], padding_value=0).t().to(self.device)
                    input_mask = inputs > 0

                    mention_reps = self.model(inputs, input_mask, is_mention=True).detach().cpu().numpy()

                    candidates_pageids, sims = self.searcher.search(mention_reps, NNs)

                    for i in range(len(lines)):
                        lines[i]['nearest_neighbors'] = candidates_pageids[i]
                        lines[i]['similarity'] = sims[i].tolist()
                        output = json.dumps(lines[i]) + '\n'
                        fout.write(output)
                        index.append(index[-1] + len(output))

                        total += 1
                        #trues += int(lines[i]['linkpage_id'] in lines[i]['nearest_neighbors'])

                    bar.update(len(lines))
                    #bar.set_description(f"Recall@10: {trues/total}")

        index = np.array(index)
        np.save(index_output_file, index)

    def calculate_inbatch_accuracy(self, scores):
        preds = torch.argmax(scores, dim=1).tolist()
        result = sum([int(i == p) for i, p in enumerate(preds)])
        return result / scores.size(0)

    def train(self,
              mention_dataset,
              index,
              candidate_dataset,
              args=None,
              mention_tokenizer=None,
             ):

        optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        self.model.load_state_dict(torch.load(args.model_path + "_1.model"))

        if args.fp16:
            assert args.fp16_opt_level is not None
            self.model, optimizer = to_fp16(self.model, optimizer, args.fp16_opt_level)

        if args.parallel:
            self.model = to_parallel(self.model)

        for e in range(args.epochs):
            if e < 2:
                continue

            if args.hard_negative and args is not None:
                if e > 0:
                    self.build_searcher(candidate_dataset, builder_gpu=args.builder_gpu, faiss_gpu_id=args.faiss_gpu_id, max_title_len=args.max_title_len, max_desc_len=args.max_desc_len)
                    self.save_traindata_with_negative_samples(
                        mention_dataset,
                        args.path_for_NN + f"/{e}.jsonl",
                        args.path_for_NN + f"/{e}_index.npy",
                        batch_size=args.bsz,
                        # batch_size=512,
                        NNs=100,
                        traindata_size=args.traindata_size,
                    )
                    index = np.load(args.path_for_NN + f"/{e}_index.npy")
                    mention_dataset = MentionDataset(args.path_for_NN + f"/{e}.jsonl", index, mention_tokenizer, preprocessed=args.mention_preprocessed, return_json=True, without_context=args.without_context, use_index=False)

            #mention_batch = mention_dataset.batch(batch_size=batch_size, random_bsz=random_bsz, max_ctxt_len=max_ctxt_len)
            dataloader = DataLoader(mention_dataset, batch_size=args.bsz, shuffle=False, collate_fn=my_collate_fn_json, num_workers=8)
            bar = tqdm(total=args.traindata_size)
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
                if args.hard_negative and e > 0:
                    hard_pages = []
                    for label, line in zip(labels, lines):
                        _hard_pages = []
                        for i in line["nearest_neighbors"]:
                            if len(_hard_pages) >= args.num_negs:
                                break
                            if str(i) == label:
                                continue
                            _hard_pages.append(str(i))
                        hard_pages.extend(_hard_pages)

                    candidate_input_ids = candidate_dataset.get_pages(pages, max_title_len=args.max_title_len, max_desc_len=args.max_desc_len)
                    candidate_inputs = pad_sequence([torch.LongTensor(token)
                                                    for token in candidate_input_ids], padding_value=0).t().to(self.device)
                    candidate_mask = candidate_inputs > 0
                    candidate_reps = self.model(candidate_inputs, candidate_mask, is_mention=False)
                    scores = mention_reps.mm(candidate_reps.t())

                    hard_candidate_input_ids = candidate_dataset.get_pages(hard_pages, max_title_len=args.max_title_len, max_desc_len=args.max_desc_len)
                    shard_bsz = args.bsz
                    for bsz in range(0, len(hard_candidate_input_ids), shard_bsz):
                        hard_candidate_inputs = pad_sequence([torch.LongTensor(token)
                                                                for token in hard_candidate_input_ids[bsz:bsz+shard_bsz]], padding_value=0).t().to(self.device)
                        hard_candidate_mask = hard_candidate_inputs > 0
                        if bsz == 0:
                            hard_candidate_reps = self.model(hard_candidate_inputs, hard_candidate_mask, is_mention=False)
                        else:
                            hard_candidate_reps = torch.cat([hard_candidate_reps, self.model(hard_candidate_inputs, hard_candidate_mask, is_mention=False)], dim=0)
                    hard_scores = torch.bmm(mention_reps.unsqueeze(1), torch.transpose(hard_candidate_reps.view(-1, args.num_negs, 768), 1, 2)).view(-1, args.num_negs)

                    scores = torch.cat([scores, hard_scores], dim=-1)
                    accuracy = self.calculate_inbatch_accuracy(scores)

                    target = torch.LongTensor(torch.arange(scores.size(0))).to(self.device)
                    loss = F.cross_entropy(scores, target, reduction="mean")
                else:
                    candidate_input_ids = candidate_dataset.get_pages(pages, max_title_len=args.max_title_len, max_desc_len=args.max_desc_len)
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
                    # scheduler.step()
                    optimizer.zero_grad()

                    if self.logger:
                        self.logger.debug("Back propagation in step %s", step+1)
                        # self.logger.debug("LR: %s", scheduler.get_lr())

                if self.use_mlflow:
                    mlflow.log_metric("train loss", loss.item(), step=step)
                    mlflow.log_metric("accuracy", accuracy, step=step)

                #if self.model_path is not None and step % model_save_interval == 0:
                    #torch.save(self.model.state_dict(), self.model_path)
                    # save_model(self.model, self.model_path + f'_{e}.model')

                bar.update(len(input_ids))
                bar.set_description(f"Loss: {loss.item()}, Accuracy: {accuracy}")

            if self.model_path is not None:
                save_model(self.model, self.model_path + f'_{e}.model')


