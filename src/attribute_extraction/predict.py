import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

from utils.dataset import ShinraData
from attribute_extraction.dataset import NerDataset, ner_collate_fn, create_dataset_for_ner
from attribute_extraction.model import BertForMultilabelNER


def ner_for_shinradata(model, tokenizer, shinra_dataset, device):
    processed_data, valid_line_id = create_dataset_for_ner(shinra_dataset)
    dataset = NerDataset(processed_data, tokenizer)
    total_preds, _ = predict(model, dataset, device)

    shinra_dataset.add_nes_from_iob(total_preds, valid_line_ids=valid_line_id)

    return shinra_dataset


def predict(model, dataset, device):
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

            preds = model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_idxs=word_idxs,
            )

            total_preds.extend(preds)
            total_trues.extend(labels if labels is not None else [None])

    return total_preds, total_trues
