import argparse
import json
import sys
from pathlib import Path

sys.path.append("../")

import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

from utils.dataset import ShinraData
from utils.util import to_parallel, to_fp16
from attribute_extraction.dataset import NerDataset, ner_collate_fn, create_dataset_for_ner
from attribute_extraction.model import BertForMultilabelNER, create_pooler_matrix


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def ner_for_shinradata(model, tokenizer, shinra_dataset, device, bsz=8):
    processed_data = shinra_dataset.ner_inputs
    dataset = NerDataset(processed_data, tokenizer)
    total_preds, _ = predict(model, dataset, device, sent_wise=True)

    shinra_dataset.add_nes_from_iob(total_preds)

    return shinra_dataset


def predict(model, dataset, device, sent_wise=False):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=ner_collate_fn)

    total_preds = []
    total_trues = []
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            input_ids = inputs["tokens"]
            word_idxs = inputs["word_idxs"]

            labels = inputs["labels"]

            input_ids = pad_sequence([torch.tensor(t) for t in input_ids], padding_value=0, batch_first=True).to(device)
            attention_mask = input_ids > 0
            pooling_matrix = create_pooler_matrix(input_ids, word_idxs, pool_type="head").to(device)

            preds = model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_idxs=word_idxs,
                pooling_matrix=pooling_matrix
            )

            total_preds.append(preds)
            # test dataの場合truesは使わないので適当にpredsを入れる
            total_trues.append(labels if labels[0] is not None else preds)

    attr_num = len(total_preds[0])
    total_preds = [[pred for preds in total_preds for pred in preds[attr]] for attr in range(attr_num)]
    total_trues = [[true for trues in total_trues for true in trues[attr]] for attr in range(attr_num)]

    if sent_wise:
        total_preds = [[total_preds[attr][idx] for attr in range(attr_num)] for idx in range(len(total_preds[0]))]
        total_trues = [[total_trues[attr][idx] for attr in range(attr_num)] for idx in range(len(total_trues[0]))]

    return total_preds, total_trues

def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_name", type=str, help="Specify BERT name")
    parser.add_argument("--input_path", type=str, help="Specify input path in SHINRA2020")
    parser.add_argument("--model_path", type=str, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--attribute_path", type=str, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--output_path", type=str, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--bsz", type=int, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--parallel", action="store_true", help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--fp16", action="store_true", help="whether using inbatch negative")
    parser.add_argument('--fp16_opt_level', type=str, default="O1")
    parser.add_argument("--seed", type=int, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--note", type=str, help="Specify attribute_list path in SHINRA2020")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arg()

    bert = AutoModel.from_pretrained(args.bert_name)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)

    # dataset = [ShinraData(), ....]
    dataset = ShinraData.from_shinra2020_format(Path(args.input_path), attribute_path=args.attribute_path)

    with open(args.attribute_path, "r") as f:
        attributes = [attr for attr in f.read().split("\n") if attr != '']

    model = BertForMultilabelNER(bert, len(attributes)).to(device)
    model.load_state_dict(torch.load(args.model_path))

    if args.fp16:
        assert args.fp16_opt_level is not None
        model = to_fp16(model, fp16_opt_level=args.fp16_opt_level)

    with open(args.output_path, "w") as f:
        for data in dataset:
            output_dataset = ner_for_shinradata(model, tokenizer, data, device, bsz=args.bsz)
            f.write("\n".join([json.dumps(n, ensure_ascii=False) for n in output_dataset.nes]))
