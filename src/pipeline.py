import argparse

from transformers import AutoTokenizer, AutoModel

from attribute_extraction.model import BertForMultilabelNER
from attribute_extraction.predict import ner_for_shinradata

def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_name", type=str, help="Specify BERT name")
    parser.add_argument("--note", type=str, help="Specify attribute_list path in SHINRA2020")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arg()

    bert = AutoModel.from_pretrained(args.bert_name)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)