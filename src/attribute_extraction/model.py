import torch
import torch.nn as nn
from transformers import AutoModel

class BertForMultilabelNER(nn.Module):
    def __init__(self, bert, attribute_num, device, dropout=0.1, pooler="head"):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)

        # classifier that classifies token into IOB tag (B, I, O) for each attribute
        classifiers = [nn.Linear(768, 3) for i in range(attribute_num)]
        self.classifiers = nn.ModuleList(classifiers)

        # pooler type, "head" "average", which specify how to pool subword representations into word representation
        self.pooler = pooler
        self.device = device

    def _create_pooler_matrix(self, input_ids, word_idxs):
        bsz, subword_len = input_ids.size()
        max_word_len = max([len(w) for w in word_idxs])
        pooler_matrix = torch.zeros(bsz * max_word_len * subword_len)

        if self.pooler == "head":
            pooler_idxs = [subword_len * max_word_len * batch_offset +  subword_len * word_offset + w
                for batch_offset, word_idx in enumerate(word_idxs) for word_offset, w in enumerate(word_idx)]
            pooler_matrix.scatter_(0, torch.LongTensor(pooler_idxs), 1)
            return pooler_matrix.view(bsz, max_word_len, subword_len).to(self.device)

        """
        elif self.pooler == "average":
            pooler_idxs = [subword_len * max_word_len * batch_offset +  subword_len * word_offset + w
                for batch_offset, word_idx in enumerate(word_idxs)
                for word_offset, w in enumerate(word_idx)]
            pooler_matrix.scatter_(0, torch.LongTensor(pooler_idxs), 1)
            return pooler_matrix.view(bsz, max_word_len, subword_len)
        """

    def predict(
        self,
        input_ids=None,
        attention_mask=None,
        word_idxs=None
    ):
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_idxs=word_idxs)[0]
        labels = [torch.argmax(logit.detach().cpu(), dim=-1) for logit in logits]

        truncated_labels = [[label[:len(word_idx)].tolist() for label, word_idx in zip(attr_labels, word_idxs)] for attr_labels in labels]

        return truncated_labels


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        word_idxs=None
    ):
        pooler_matrix = self._create_pooler_matrix(input_ids, word_idxs)

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        # create word-level representations using pooler matrix
        sequence_output = torch.bmm(pooler_matrix, sequence_output)
        sequence_output = self.dropout(sequence_output)

        logits = [classifier(sequence_output) for classifier in self.classifiers]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = 0

            for label, logit in zip(labels, logits):
                loss += loss_fct(logit.view(-1, 3), label.view(-1))

        output = (logits, ) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

if __name__ == "__main__":
    bert = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")

    inputs = torch.tensor([[1,2,3,4,5,6], [2,3,4,5,6,7]])
    word_idxs = [[1,2,3], [2,3,4,5]]

    model = BertForMultilabelNER(bert, 2)

    attention_mask = inputs > 0

    outputs = model(inputs, attention_mask=attention_mask, word_idxs=word_idxs, labels=labels)
    print(outputs)
    #pooler = model._create_pooler_matrix(inputs, word_idxs)
