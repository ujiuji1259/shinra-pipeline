import sys
from pathlib import Path
sys.path.append("..")

import torch
from torch.utils.data import Dataset, DataLoader
from utils.dataset import ShinraData
from transformers import AutoTokenizer


def create_batch_dataset_for_ner(datasets):
    ner_dataset = [create_dataset_for_ner(d) for d in datasets]
    outputs = {}
    outputs["tokens"] = [tokens for d in ner_dataset for tokens in d["tokens"]]
    outputs["word_idxs"] = [word_idxs for d in ner_dataset for word_idxs in d["word_idxs"]]

    if ner_dataset[0]["labels"] is not None:
        outputs["labels"] = [labels for d in ner_dataset for labels in d["labels"]]
    else:
        outputs["labels"] = None

    return outputs


def create_dataset_for_ner(dataset):
    outputs = {}
    ner_inputs = dataset.ner_inputs

    valid_line_id = []
    tokens = []
    for idx, data in enumerate(ner_inputs["input_ids"]):
        if len(data) > 0:
            tokens.append(data)
            valid_line_id.append(idx)

    outputs["tokens"] = tokens
    outputs["valid_line_id"] = valid_line_id

    outputs["word_idxs"] = [ner_inputs["word_idxs"][i] for i in valid_line_id]

    if "labels" not in ner_inputs:
        outputs["labels"] = None
    else:
        outputs["labels"] = [ner_inputs["labels"][i] for i in valid_line_id]

    return outputs


class NerDataset(Dataset):
    label2id = {
        "O": 0,
        "B": 1,
        "I": 2
    }
    # datas = [{"tokens": , "word_idxs": , "labels": }, ...]
    def __init__(self, datas, tokenizer):
        self.tokenizer = tokenizer
        self.tokens = datas["tokens"]
        self.word_idxs = datas["word_idxs"]
        self.labels = datas["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        input_ids = ["[CLS]"] + self.tokens[item][:510] + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_ids)
        word_idxs = [idx+1 for idx in self.word_idxs[item] if idx <= 510]

        if self.labels is not None:
            labels = self.labels[item]
            # truncate label using zip(_, word_idxs)
            labels = [[self.label2id[l] for l, _ in zip(label, word_idxs)] for label in labels]
        else:
            labels = None

        return input_ids, word_idxs, labels

def ner_collate_fn(batch):
    tokens, word_idxs, labels = list(zip(*batch))
    if labels is not None:
        labels = [[label[idx] for label in labels] for idx in range(len(labels[0]))]
    else:
        labels = None

    return {"tokens": tokens, "word_idxs": word_idxs, "labels": labels}

if __name__ == "__main__":
    dataset = ShinraData.from_shinra2020_format("/data1/ujiie/shinra/tohoku_bert/attributes.pickle", Path("/data1/ujiie/shinra/tohoku_bert/Event/Event_Other"))

    ner = NerDataset(dataset)
    dataloader = DataLoader(ner, collate_fn=ner_collate_fn, batch_size=8)
    for inputs in dataloader:
        print(inputs["labels"])
        break
