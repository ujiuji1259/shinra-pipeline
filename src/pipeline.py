import argparse
from pathlib import Path
import sys
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
from transformers import AutoTokenizer, AutoModel

from attribute_extraction.model import BertForMultilabelNER
from attribute_extraction.predict import ner_for_shinradata
from entity_linking.entity_linking import entity_linking_for_shinradata
from entity_linking.dataset import CandidateDataset
from entity_linking.bert_generator import BertBiEncoder, BertCandidateGenerator
from entity_linking.bert_ranking import BertCrossEncoder, BertCandidateRanker
from utils.dataset import ShinraData
from utils.util import to_fp16, to_parallel


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_arg():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--input", type=str, help="bert-name used for biencoder")
    parser.add_argument("--attribute", type=str, help="bert-name used for biencoder")

    # args for entity linking
    parser.add_argument("--model_name", type=str, help="bert-name used for biencoder")
    parser.add_argument("--generator_path", type=str, help="model save path")
    parser.add_argument("--ranker_path", type=str, help="model save path")
    parser.add_argument("--index_path", type=str, help="model save path")
    parser.add_argument("--load_index", action="store_true", help="model save path")
    parser.add_argument("--candidate_dataset", type=str, help="candidate dataset path")
    parser.add_argument("--candidate_preprocessed", action="store_true", help="whether candidate_dataset is preprocessed")
    parser.add_argument("--builder_gpu", action="store_true", help="bert-name used for biencoder")
    parser.add_argument("--max_ctxt_len", type=int, help="maximum context length")
    parser.add_argument("--max_title_len", type=int, help="maximum title length")
    parser.add_argument("--max_desc_len", type=int, help="maximum description length")
    parser.add_argument("--parallel", action="store_true", help="whether using inbatch negative")
    parser.add_argument("--fp16", action="store_true", help="whether using inbatch negative")
    parser.add_argument('--fp16_opt_level', type=str, default="O1")

    parser.add_argument("--note", type=str, help="Specify attribute_list path in SHINRA2020")

    args = parser.parse_args()

    return args

def init_entity_linking(args):
    mention_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    mention_tokenizer.add_special_tokens({"additional_special_tokens": ["[M]", "[/M]"]})

    candidate_dataset = CandidateDataset(args.candidate_dataset, mention_tokenizer, preprocessed=args.candidate_preprocessed)

    mention_bert = AutoModel.from_pretrained(args.model_name)
    mention_bert.resize_token_embeddings(len(mention_tokenizer))
    candidate_bert = AutoModel.from_pretrained(args.model_name)

    biencoder = BertBiEncoder(mention_bert, candidate_bert)
    biencoder.load_state_dict(torch.load(args.generator_path))

    cross_bert = AutoModel.from_pretrained(args.model_name)
    cross_bert.resize_token_embeddings(len(mention_tokenizer))
    crossencoder = BertCrossEncoder(cross_bert)
    crossencoder.load_state_dict(torch.load(args.ranker_path))


    generator = BertCandidateGenerator(
        biencoder,
        device,
        model_path=args.generator_path,
        use_mlflow=False,
        builder_gpu=args.builder_gpu,
        logger=None)


    ranker = BertCandidateRanker(
        crossencoder,
        device,
        model_path=args.ranker_path,
        use_mlflow=False,
        logger=None)

    if args.fp16:
        generator.model = to_fp16(generator.model, fp16_opt_level=args.fp16_opt_level)
        ranker.model = to_fp16(ranker.model, fp16_opt_level=args.fp16_opt_level)

    if args.parallel:
        generator.model = to_parallel(generator.model)
        ranker.model = to_parallel(ranker.model)

    if args.load_index:
        generator.load_index(args.index_path)
    else:
        generator.build_searcher(candidate_dataset, max_title_len=args.max_title_len, max_desc_len=args.max_desc_len)
        generator.save_index(args.index_path)

    return {
        "tokenizer": mention_tokenizer,
        "generator": generator,
        "ranker": ranker,
        "candidate_dataset": candidate_dataset
    }


if __name__ == "__main__":
    args = parse_arg()

    dataset = ShinraData.from_shinra2020_format(args.attribute, Path(args.input))[5]

    entity_linking = init_entity_linking(args)

    dataset = entity_linking_for_shinradata(
        generator=entity_linking["generator"],
        ranker=entity_linking["ranker"],
        dataset=dataset,
        candidate_dataset=entity_linking["candidate_dataset"],
        tokenizer=entity_linking["tokenizer"],
        args=args,
        debug=False,
    )
    with open('log.json', "w") as f:
        json.dump(dataset.nes, f, ensure_ascii=False, indent=4)
