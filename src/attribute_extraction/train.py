import argparse
import sys
from pathlib import Path
sys.path.append("..")

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from seqeval.metrics import f1_score
import mlflow

from utils.dataset import ShinraData
from utils.util import get_scheduler, to_parallel, save_model, decode_iob
from dataset import NerDataset, ner_collate_fn
from model import BertForMultilabelNER

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_name", type=str, help="Specify BERT name")
    parser.add_argument("--input_path", type=str, help="Specify input path in SHINRA2020")
    parser.add_argument("--attribute_list", type=str, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--data_split", type=str, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--model_path", type=str, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--lr", type=float, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--bsz", type=int, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--epoch", type=int, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--grad_acc", type=int, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--warmup", type=float, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--grad_clip", type=float, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--parallel", action="store_true", help="Specify attribute_list path in SHINRA2020")

    args = parser.parse_args()

    return args


def evaluate(model, dataset, attributes, args):
    dataloader = DataLoader(dataset, batch_size=args.bsz, collate_fn=ner_collate_fn, shuffle=True)
    bar = tqdm(total=len(dataset))

    total_preds = []
    total_trues = []
    for step, inputs in enumerate(dataloader):
        input_ids = inputs["tokens"]
        word_idxs = inputs["word_idxs"]

        labels = inputs["labels"]
        label_ids = [pad_sequence([torch.tensor(l) for l in label], padding_value=-1, batch_first=True).to(device)
            for label in labels]

        input_ids = pad_sequence([torch.tensor(t) for t in input_ids], padding_value=0, batch_first=True).to(device)
        attention_mask = input_ids > 0

        preds = model.predict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_idxs=word_idxs,
        )
        pred_iobs = decode_iob(preds, attributes)
        true_iobs = decode_iob(labels, attributes)

        total_preds.extend(pred_iobs)
        total_trues.extend(true_iobs)

    f1 = f1_score(true_iobs, pred_iobs)
    return f1


def train(model, train_dataset, valid_dataset, attributes, args):
    valid_dataloader = DataLoader(train_dataset, batch_size=args.bsz, collate_fn=ner_collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_scheduler(
        args.bsz, args.grad_acc, args.epoch, args.warmup, optimizer, len(train_dataset))

    if args.parallel:
        model = to_parallel(model)

    losses = []
    valid_scores = [-float("inf"), -float("inf")]
    for e in range(args.epoch):
        train_dataloader = DataLoader(train_dataset, batch_size=args.bsz, collate_fn=ner_collate_fn, shuffle=True)
        bar = tqdm(total=len(train_dataset))

        total_loss = 0
        for step, inputs in enumerate(train_dataloader):
            input_ids = inputs["tokens"]
            word_idxs = inputs["word_idxs"]
            labels = inputs["labels"]

            labels = [pad_sequence([torch.tensor(l) for l in label], padding_value=-1, batch_first=True).to(device)
                for label in labels]

            input_ids = pad_sequence([torch.tensor(t) for t in input_ids], padding_value=0, batch_first=True).to(device)
            attention_mask = input_ids > 0

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_idxs=word_idxs,
                labels=labels)

            loss = outputs[0]
            loss.backward()

            total_loss += loss.item()
            mlflow.log_metric("Trian batch loss", loss.item(), step=(e+1) * step)
            bar.set_description(f"[Epoch] {e + 1}")
            bar.set_postfix({"loss":loss.item()})
            bar.update(1)

            if (step + 1) % args.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        losses.append(total_loss / len(train_dataset))
        mlflow.log_metric("Trian loss", losses[-1], step=e)

        with torch.zero_grad():
            valid_f1 = evaluate(model, valid_dataset, attributes, args)
        valid_scores.append(valid_f1)
        mlflow.log_metric("Valid F1", valid_scores[-1], step=e)

        if valid_scores[-1] < valid_scores[-2] and valid_scores[-1] < valid_scores[-3]:
            break


if __name__ == "__main__":
    args = parse_arg()

    bert = AutoModel.from_pretrained(args.bert_name)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)

    dataset = ShinraData.from_shinra2020_format(Path(args.attribute_list), Path(args.input_path))

    model = BertForMultilabelNER(bert, len(dataset[0].attributes), device).to(device)

    data_split = Path(args.data_split)
    if (data_split / "train.txt").exists():
        with open(data_split / "train.txt", "r") as f:
            train_ids = set([int(l) for l in f.read().split("\n") if l != ""])
        train_dataset = NerDataset([d for d in dataset if d.page_id in train_ids], tokenizer)

    if (data_split / "valid.txt").exists():
        with open(data_split / "valid.txt", "r") as f:
            valid_ids = set([int(l) for l in f.read().split("\n") if l != ""])
        valid_dataset = NerDataset([d for d in dataset if d.page_id in valid_ids], tokenizer)

    if (data_split / "test.txt").exists():
        with open(data_split / "test.txt", "r") as f:
            test_ids = set([int(l) for l in f.read().split("\n") if l != ""])
        test_dataset = NerDataset([d for d in dataset if d.page_id in test_ids], tokenizer)

    mlflow.start_run()
    train(model, train_dataset, valid_dataset, dataset[0].attributes, args)
    mlflow.end_run()
