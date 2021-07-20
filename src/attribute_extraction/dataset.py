import sys
from pathlib import Path
sys.path.append("..")

import torch
from torch.utils.data import Dataset, DataLoader
from utils.dataset import ShinraData
from transformers import AutoTokenizer


def create_batch_dataset_for_ner(datasets):
    outputs = [output for d in datasets for output in create_dataset_for_ner(d)[0]]
    return outputs


def create_dataset_for_ner(dataset):
    outputs = []
    ner_inputs = dataset.ner_inputs

    valid_line_id = []
    for idx, data in enumerate(ner_inputs["input_ids"]):
        if len(data) > 0:
            outputs.append({
                "tokens": data,
                "word_idxs": ner_inputs["word_idxs"][idx],
                "labels": ner_inputs["labels"][idx],
            })
            valid_line_id.append(idx)

    return outputs, valid_line_id


class NerDataset(Dataset):
    label2id = {
        "O": 0,
        "B": 1,
        "I": 2
    }
    # datas = [{"tokens": , "word_idxs": , "labels": }, ...]
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        input_ids = ["[CLS]"] + self.data[item]["tokens"][:510] + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_ids)
        word_idxs = [idx+1 for idx in self.data[item]["word_idxs"] if idx <= 510]

        labels = self.data[item]["labels"]
        if labels is not None:
            # truncate label using zip(_, word_idxs)
            labels = [[self.label2id[l] for l, _ in zip(label, word_idxs)] for label in labels]

        return input_ids, word_idxs, labels

def ner_collate_fn(batch):
    tokens, word_idxs, labels = list(zip(*batch))
    if labels[0] is not None:
        labels = [[label[idx] for label in labels] for idx in range(len(labels[0]))]

    return {"tokens": tokens, "word_idxs": word_idxs, "labels": labels}

if __name__ == "__main__":
    dataset = ShinraData.from_shinra2020_format("/data1/ujiie/shinra/tohoku_bert/attributes.pickle", Path("/data1/ujiie/shinra/tohoku_bert/Event/Event_Other"))

    ner = NerDataset(dataset)
    dataloader = DataLoader(ner, collate_fn=ner_collate_fn, batch_size=8)
    for inputs in dataloader:
        print(inputs["labels"])
        break
