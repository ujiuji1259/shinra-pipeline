import argparse
import sys
from pathlib import Path
import json
import random
import os

from torch.optim.optimizer import Optimizer

sys.path.append("../")

from apex import amp
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from seqeval.metrics import f1_score, classification_report
import mlflow

from utils.dataset import ShinraData
from utils.util import get_scheduler, to_parallel, save_model, decode_iob, to_fp16
from dataset import NerDataset, ner_collate_fn, create_batch_dataset_for_ner
from model import BertForMultilabelNER, create_pooler_matrix
from predict import predict

device = "cuda" if torch.cuda.is_available() else "cpu"

# seed固定
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping():
   def __init__(self, patience=0, verbose=0):
       self._step = 0
       self._score = - float('inf')
       self.patience = patience
       self.verbose = verbose

   def validate(self, score):
       if self._score > score:
           self._step += 1
           if self._step > self.patience:
               if self.verbose:
                   print('early stopping')
               return True
       else:
           self._step = 0
           self._score = score

       return False


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_name", type=str, help="Specify BERT name")
    parser.add_argument("--input_path", type=str, help="Specify input path in SHINRA2020")
    parser.add_argument("--data_split", type=str, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--model_path", type=str, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--lr", type=float, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--bsz", type=int, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--epoch", type=int, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--grad_acc", type=int, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--warmup", type=float, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--grad_clip", type=float, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--parallel", action="store_true", help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--fp16", action="store_true", help="whether using inbatch negative")
    parser.add_argument('--fp16_opt_level', type=str, default="O1")
    parser.add_argument("--seed", type=int, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--note", type=str, help="Specify attribute_list path in SHINRA2020")

    args = parser.parse_args()

    return args

def evaluate(model, dataset, attributes, args):
    total_preds, total_trues = predict(model, dataset, device)
    total_preds = decode_iob(total_preds, attributes)
    total_trues = decode_iob(total_trues, attributes)

    f1 = f1_score(total_trues, total_preds)
    return f1


def train(model, train_dataset, valid_dataset, attributes, args):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = get_scheduler(
    #     args.bsz, args.grad_acc, args.epoch, args.warmup, optimizer, len(train_dataset))

    early_stopping = EarlyStopping(patience=10, verbose=1)

    category = Path(args.data_split).stem

    if args.fp16:
        assert args.fp16_opt_level is not None
        model, optimizer = to_fp16(model, optimizer, args.fp16_opt_level)

    if args.parallel:
        model = to_parallel(model)

    losses = []
    for e in range(args.epoch):
        train_dataloader = DataLoader(train_dataset, batch_size=args.bsz, collate_fn=ner_collate_fn, shuffle=True, num_workers=2)
        bar = tqdm(total=len(train_dataset))

        total_loss = 0
        model.train()
        for step, inputs in enumerate(train_dataloader):
            input_ids = inputs["tokens"]
            word_idxs = inputs["word_idxs"]
            labels = inputs["labels"]

            labels = [pad_sequence([torch.tensor(l) for l in label], padding_value=-1, batch_first=True)
                for label in labels]
            # for parallel
            labels = torch.tensor([[labels[attr_idx][bsz].tolist() for attr_idx in range(len(labels))] for bsz in range(labels[0].size(0))]).to(device)

            input_ids = pad_sequence([torch.tensor(t) for t in input_ids], padding_value=0, batch_first=True).to(device)
            attention_mask = input_ids > 0
            pooling_matrix = create_pooler_matrix(input_ids, word_idxs, pool_type="head").to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pooling_matrix=pooling_matrix)

            loss = outputs[0].mean()
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # loss.backward()

            total_loss += loss.item()
            mlflow.log_metric("Trian batch loss", loss.item(), step=(e+1) * step)

            bar.set_description(f"[Epoch] {e + 1}")
            bar.set_postfix({"loss": loss.item()})
            bar.update(args.bsz)

            if (step + 1) % args.grad_acc == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.grad_clip
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                # torch.nn.utils.clip_grad_norm_(
                #     model.parameters(), args.grad_clip
                # )
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()

        losses.append(total_loss / (step+1))
        mlflow.log_metric("Trian loss", losses[-1], step=e)

        valid_f1 = evaluate(model, valid_dataset, attributes, args)
        mlflow.log_metric("Valid F1", valid_f1, step=e)

        if early_stopping._score < valid_f1:
            # torch.save(model.state_dict(), args.model_path + f"{category}_best.model")
            save_model(model, args.model_path + f"/{category}_best.model")


        if early_stopping.validate(valid_f1) and e + 1 > 30:
            break


if __name__ == "__main__":
    args = parse_arg()

    set_seed(args.seed)

    bert = AutoModel.from_pretrained(args.bert_name)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)

    # dataset = [ShinraData(), ....]
    dataset = ShinraData.from_shinra2020_format(Path(args.input_path))


    data_split = Path(args.data_split)
    train_dataset, valid_dataset, test_dataset = [], [], []
    train_ids, valid_ids, test_ids = set(), set(), set()
    attributes = None
    if (data_split / "train.txt").exists():
        with open(data_split / "train.txt", "r") as f:
            train_ids = set([int(l) for l in f.read().split("\n") if l != ""])
        # train_dataset = [d for d in dataset if d.page_id in train_ids]
        # train_dataset = NerDataset(create_batch_dataset_for_ner(train_dataset), tokenizer)

    if (data_split / "valid.txt").exists():
        with open(data_split / "valid.txt", "r") as f:
            valid_ids = set([int(l) for l in f.read().split("\n") if l != ""])
        # valid_dataset = [d for d in dataset if d.page_id in valid_ids]
        # valid_dataset = NerDataset(create_batch_dataset_for_ner(valid_dataset), tokenizer)

    if (data_split / "test.txt").exists():
        with open(data_split / "test.txt", "r") as f:
            test_ids = set([int(l) for l in f.read().split("\n") if l != ""])
        # test_dataset = [d for d in dataset if d.page_id in test_ids]
        # test_dataset = NerDataset(create_batch_dataset_for_ner(test_dataset), tokenizer)
    for idx, data in enumerate(dataset):
        if idx == 0:
            attributes = data.attributes.copy()
        if data.page_id in train_ids:
            train_dataset.extend(data.ner_inputs)
        if data.page_id in valid_ids:
            valid_dataset.extend(data.ner_inputs)
        if data.page_id in test_ids:
            test_dataset.extend(data.ner_inputs)

    assert len(train_dataset) != 0, "Please specify the path for the training data"
    assert len(valid_dataset) != 0, "Please specify the path for the validation data"

    train_dataset = NerDataset(train_dataset, tokenizer)
    valid_dataset = NerDataset(valid_dataset, tokenizer)

    model = BertForMultilabelNER(bert, len(attributes)).to(device)

    mlflow.start_run()
    mlflow.log_params(vars(args))
    train(model, train_dataset, valid_dataset, attributes, args)
    # torch.save(model.state_dict(), args.model_path + f"{data_split.stem}_last.model")
    save_model(model, args.model_path + f"/{data_split.stem}_last.model")
    mlflow.end_run()
