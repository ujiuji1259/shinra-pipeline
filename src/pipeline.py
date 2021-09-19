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
    # for input data
    parser.add_argument("--input_path", type=str, help="mention dataset path")
    parser.add_argument("--category", type=str, help="mention dataset path")
    parser.add_argument("--output_path", type=str, help="model save path")

    # for ner
    parser.add_argument("--ner_path", type=str, help="Specify attribute_list path in SHINRA2020")
    parser.add_argument("--attribute_path", type=str, help="Specify attribute_list path in SHINRA2020")

    # for entity linking
    # for model
    parser.add_argument("--model_name", type=str, help="bert-name used for biencoder")
    parser.add_argument("--biencoder_path", type=str, help="model save path")
    parser.add_argument("--crossencoder_path", type=str, help="model save path")

    parser.add_argument("--without_context", action="store_true", help="bert-name used for biencoder")
    parser.add_argument("--max_ctxt_len", type=int, help="maximum context length")
    parser.add_argument("--max_title_len", type=int, help="maximum title length")
    parser.add_argument("--max_desc_len", type=int, help="maximum description length")

    # for data
    parser.add_argument("--candidate_dataset", type=str, help="candidate dataset path")
    parser.add_argument("--candidate_preprocessed", action="store_true", help="whether candidate_dataset is preprocessed")

    # for faiss
    parser.add_argument("--index_path", type=str, help="model save path")
    parser.add_argument("--load_index", action="store_true", help="model save path")
    parser.add_argument("--builder_gpu", action="store_true", help="bert-name used for biencoder")
    parser.add_argument("--faiss_gpu_id", type=int, help="bert-name used for biencoder")

    # for config
    parser.add_argument("--parallel", action="store_true", help="whether using inbatch negative")
    parser.add_argument("--fp16", action="store_true", help="whether using inbatch negative")
    parser.add_argument('--fp16_opt_level', type=str, default="O1")
    args = parser.parse_args()


    return args


def init_ner(args, attributes):
    model = BertForMultilabelNER(bert, len(attributes)).to(device)
    model.load_state_dict(torch.load(args.ner_path))

    if args.fp16:
        assert args.fp16_opt_level is not None
        model = to_fp16(model, fp16_opt_level=args.fp16_opt_level)

    if args.parallel:
        model = to_parallel(model)

    return model


def init_entity_linking(args):
    mention_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    mention_tokenizer.add_special_tokens({"additional_special_tokens": ["[M]", "[/M]"]})

    candidate_dataset = CandidateDataset(args.candidate_dataset, mention_tokenizer, preprocessed=args.candidate_preprocessed, without_context=args.without_context)


    mention_bert = AutoModel.from_pretrained(args.model_name)
    mention_bert.resize_token_embeddings(len(mention_tokenizer))
    candidate_bert = AutoModel.from_pretrained(args.model_name)

    biencoder = BertBiEncoder(mention_bert, candidate_bert)
    biencoder.load_state_dict(torch.load(args.biencoder_path))

    cross_bert = AutoModel.from_pretrained(args.model_name)
    cross_bert.resize_token_embeddings(len(mention_tokenizer))
    crossencoder = BertCrossEncoder(cross_bert)
    crossencoder.load_state_dict(torch.load(args.crossencoder_path))


    generator = BertCandidateGenerator(
        biencoder,
        device,
        model_path=args.biencoder_path,
        use_mlflow=args.mlflow,
        logger=logger)


    ranker = BertCandidateRanker(
        crossencoder,
        device,
        model_path=args.crossencoder_path,
        use_mlflow=args.mlflow,
        logger=logger)

    if args.fp16:
        generator.model = to_fp16(generator.model, fp16_opt_level=args.fp16_opt_level)
        ranker.model = to_fp16(ranker.model, fp16_opt_level=args.fp16_opt_level)

    if args.parallel:
        generator.model = to_parallel(generator.model)
        ranker.model = to_parallel(ranker.model)
        ranker.model = to_parallel(ranker.model)

    load_path = None
    if args.load_index:
        load_path = args.index_path

    generator.build_searcher(
        candidate_dataset,
        builder_gpu=args.builder_gpu,
        faiss_gpu_id=args.faiss_gpu_id,
        max_title_len=args.max_title_len,
        max_desc_len=args.max_desc_len,
        load_path=load_path,
    )
    if not args.load_index:
        generator.save_index(args.index_path)

    return {
        "tokenizer": mention_tokenizer,
        "generator": generator,
        "ranker": ranker,
        "candidate_dataset": candidate_dataset
    }


if __name__ == "__main__":
    args = parse_arg()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = ShinraData.from_plain_wiki(args.attribute_path, args.category, args.input_path, tokenizer)
    attributes = next(dataset)

    ner_model = init_ner(args. attributes)
    el = init_entity_linking(args)
    biencoder = el['generator']
    crossencoder = el['ranker']
    candidate_dataset = el['candidate_dataset']
    mention_tokenizer = el['mention_tokenizer']

    outputs = []
    for data in dataset:
        data = ner_for_shinradata(ner_model, tokenizer, data, device)
        output = entity_linking_for_shinradata(
            biencoder,
            crossencoder,
            mention_tokenizer,
            data,
            candidate_dataset,
            args.max_ctxt_len,
            args.max_title_len,
            args.desc_len
        )
        outputs.append(output)


    with open(args.output_path, "w") as f:
        f.write("\n".join(
            [
                json.dumps(output, ensure_ascii=False) for output in outputs
            ]
        ))
