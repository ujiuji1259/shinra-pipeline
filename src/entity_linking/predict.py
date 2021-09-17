import sys
sys.path.append('../')
from line_profiler import LineProfiler
import argparse
from logging import getLogger, StreamHandler, DEBUG, Formatter, FileHandler
import random
import json

import numpy as np
import mlflow
import torch
from transformers import AutoTokenizer, AutoModel
import apex
from apex import amp

from bert_generator import BertBiEncoder, BertCandidateGenerator
from bert_ranking import BertCrossEncoder, BertCandidateRanker
from utils.util import to_parallel, to_fp16, save_model
from utils.dataset import ShinraData, CandidateDataset
from utils.metric import calculate_recall
from dataset import EntityLinkingDataset

def parse_input_tokens_without_context(tokens):
    mention = ""
    overall_text = ""
    title = ""
    idx = 0

    while idx < len(tokens) and tokens[idx] != "[SEP]":
        mention += tokens[idx][2:] if tokens[idx].startswith("##") else tokens[idx]
        overall_text +=  tokens[idx][2:] if tokens[idx].startswith("##") else tokens[idx]
        idx += 1

    overall_text +=  tokens[idx][2:] if tokens[idx].startswith("##") else tokens[idx]
    idx += 1

    while idx < len(tokens) and tokens[idx] != "[SEP]":
        title += tokens[idx][2:] if tokens[idx].startswith("##") else tokens[idx]
        overall_text +=  tokens[idx][2:] if tokens[idx].startswith("##") else tokens[idx]
        idx += 1

    while idx < len(tokens):
        overall_text +=  tokens[idx][2:] if tokens[idx].startswith("##") else tokens[idx]
        idx += 1

    return mention, title, overall_text

def parse_input_tokens(tokens):
    mention = ""
    overall_text = ""
    title = ""
    idx = 0
    while idx < len(tokens) and tokens[idx] != "[M]":
        overall_text +=  tokens[idx][2:] if tokens[idx].startswith("##") else tokens[idx]
        idx += 1

    overall_text +=  tokens[idx][2:] if tokens[idx].startswith("##") else tokens[idx]
    idx += 1

    while idx < len(tokens) and tokens[idx] != "[/M]":
        mention += tokens[idx][2:] if tokens[idx].startswith("##") else tokens[idx]
        overall_text +=  tokens[idx][2:] if tokens[idx].startswith("##") else tokens[idx]
        idx += 1

    while idx < len(tokens) and tokens[idx] != "[SEP]":
        overall_text +=  tokens[idx][2:] if tokens[idx].startswith("##") else tokens[idx]
        idx += 1

    overall_text +=  tokens[idx][2:] if tokens[idx].startswith("##") else tokens[idx]
    idx += 1

    while idx < len(tokens) and tokens[idx] != "[SEP]":
        title += tokens[idx][2:] if tokens[idx].startswith("##") else tokens[idx]
        overall_text +=  tokens[idx][2:] if tokens[idx].startswith("##") else tokens[idx]
        idx += 1

    while idx < len(tokens):
        overall_text +=  tokens[idx][2:] if tokens[idx].startswith("##") else tokens[idx]
        idx += 1

    return mention, title, overall_text


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

fix_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

all_categories = ["Airport", "City", "Company", "Compound", "Conference", "Lake", "Person"]

def parse_args():
    parser = argparse.ArgumentParser()
    # for model
    parser.add_argument("--model_name", type=str, help="bert-name used for biencoder")
    parser.add_argument("--biencoder_path", type=str, help="model save path")
    parser.add_argument("--crossencoder_path", type=str, help="model save path")

    parser.add_argument("--without_context", action="store_true", help="bert-name used for biencoder")
    parser.add_argument("--max_ctxt_len", type=int, help="maximum context length")
    parser.add_argument("--max_title_len", type=int, help="maximum title length")
    parser.add_argument("--max_desc_len", type=int, help="maximum description length")

    # for data
    parser.add_argument("--input_path", type=str, help="mention dataset path")
    parser.add_argument("--category", type=str, help="mention dataset path")
    parser.add_argument("--output_path", type=str, help="model save path")
    parser.add_argument("--candidate_dataset", type=str, help="candidate dataset path")
    parser.add_argument("--candidate_preprocessed", action="store_true", help="whether candidate_dataset is preprocessed")

    # for faiss
    parser.add_argument("--index_path", type=str, help="model save path")
    parser.add_argument("--load_index", action="store_true", help="model save path")
    parser.add_argument("--builder_gpu", action="store_true", help="bert-name used for biencoder")

    # for config
    parser.add_argument("--mlflow", action="store_true", help="whether using inbatch negative")
    parser.add_argument("--parallel", action="store_true", help="whether using inbatch negative")
    parser.add_argument("--fp16", action="store_true", help="whether using inbatch negative")
    parser.add_argument('--fp16_opt_level', type=str, default="O1")
    parser.add_argument("--logging", action="store_true", help="whether using inbatch negative")
    parser.add_argument("--log_file", type=str, help="whether using inbatch negative")
    parser.add_argument("--debug", action="store_true", help="whether using inbatch negative")
    args = parser.parse_args()

    if args.mlflow:
        mlflow.start_run()
        arg_dict = vars(args)
        for key, value in arg_dict.items():
            mlflow.log_param(key, value)

    logger = None

    if args.logging:
        logger = getLogger(__name__)
        #handler = StreamHandler()

        logger.setLevel(DEBUG)
        #handler.setLevel(DEBUG)
        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #handler.setFormatter(formatter)
        #logger.addHandler(handler)

        if args.log_file:
            fh = FileHandler(filename=args.log_file)
            fh.setLevel(DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return args, logger


def main():
    args, logger = parse_args()

    mention_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    mention_tokenizer.add_special_tokens({"additional_special_tokens": ["[M]", "[/M]"]})

    candidate_dataset = CandidateDataset(args.candidate_dataset, mention_tokenizer, preprocessed=args.candidate_preprocessed, without_context=args.without_context)


    mention_bert = AutoModel.from_pretrained(args.model_name)
    mention_bert.resize_token_embeddings(len(mention_tokenizer))
    candidate_bert = AutoModel.from_pretrained(args.model_name)

    biencoder = BertBiEncoder(mention_bert, candidate_bert)
    biencoder.load_state_dict(torch.load(args.encoder_path))

    cross_bert = AutoModel.from_pretrained(args.model_name)
    cross_bert.resize_token_embeddings(len(mention_tokenizer))
    crossencoder = BertCrossEncoder(cross_bert)
    crossencoder.load_state_dict(torch.load(args.crossencoder_path))


    model = BertCandidateGenerator(
        biencoder,
        device,
        model_path=args.biencoder_path,
        use_mlflow=args.mlflow,
        logger=logger)


    cross_encoder_model = BertCandidateRanker(
        crossencoder,
        device,
        model_path=args.cross_model_path,
        use_mlflow=args.mlflow,
        logger=logger)

    if args.fp16:
        model.model = to_fp16(model.model, fp16_opt_level=args.fp16_opt_level)

    if args.parallel:
        model.model = to_parallel(model.model)

    if args.mlflow:
        mlflow.end_run()

    if args.load_index:
        model.load_index(args.index_path)
    else:
        model.build_searcher(
            candidate_dataset,
            builder_gpu=args.builder_gpu,
            max_title_len=args.max_title_len,
            max_desc_len=args.max_desc_len
        )
        model.save_index(args.index_path)

    dataset = ShinraData.from_linkjp_format(
        input_path=args.input_path,
        category=args.category,
        tokenizer=mention_tokenizer
    )

    outputs = []
    with torch.no_grad():
        for data in dataset:
            el_inputs = data.entity_linking_inputs
            mention_dataset = EntityLinkingDataset(
                el_inputs,
                mention_tokenizer,
                args.max_ctxt_len,
            )

            original_annotation = [d['annotation'] for a in el_inputs]

            # generate candidates
            preds, bi_scores, trues, input_ids = model.generate_candidates(mention_dataset)

            # rank candidates
            cross_scores, tokens = cross_encoder_model.predict(
                input_ids, preds, candidate_dataset,
                max_title_len=args.max_title_len,
                max_desc_len=args.max_desc_len)

            # sort rank-score
            rank = np.argsort(np.array(cross_scores), axis=1)[:, ::-1].tolist()
            cross_preds = [[[p[s], sc[s]] for s in ss] for ss, p, sc in zip(rank, preds, cross_scores)]

            assert len(cross_preds) == len(original_annotation)

            for data, preds in zip(original_annotation, cross_preds):
                data["link_page_id"] = str(preds[0][0])

                if args.debug:
                    data["score"] = str(preds[0][1])
                outputs.append(data)


    # if args.category == "all":
    #     for category in all_categories:
    #         mention_dataset = ShinraDataset(args.mention_dataset, category, mention_tokenizer, max_ctxt_len=args.max_ctxt_len, without_context=args.without_context, is_test=True)
    #         original_annotation = [a["annotation"] for a in mention_dataset.data]

    #         preds, bi_scores, trues, input_ids = model.generate_candidates(mention_dataset)
    #         cross_scores, tokens = cross_encoder_model.predict(
    #             input_ids, preds, candidate_dataset,
    #             max_title_len=args.max_title_len,
    #             max_desc_len=args.max_desc_len)
    #         #rank = np.argsort(np.array(cross_scores), axis=1).tolist()
    #         rank = np.argsort(np.array(cross_scores), axis=1)[:, ::-1].tolist()
    #         cross_preds = [[[p[s], sc[s]] for s in ss] for ss, p, sc in zip(rank, preds, cross_scores)]

    #         assert len(cross_preds) == len(original_annotation)

    #         for data, preds in zip(original_annotation, cross_preds):
    #             data["link_page_id"] = str(preds[0][0])
    #             data["score"] = str(preds[0][1])
    #             """
    #             data["link_type"] = {
    #                 "later_name": False,
    #                 "part_of": False,
    #                 "derivation_of": False
    #             }
    #             """

    #         with open(args.output_path + f'/{category}.jsonl', 'w') as f:
    #             f.write("\n".join([json.dumps(data, ensure_ascii=False) for data in original_annotation]))

    #         """

    #         tokens = [mention_tokenizer.convert_ids_to_tokens(t) for t in tokens]

    #         outputs = []
    #         cnt = 0
    #         for dd in cross_binary:
    #             output = {}
    #             output["data"] = []
    #             bi_true = False
    #             sc_true = False
    #             sc_score = -float("inf")
    #             for true, bi_sc, sc in dd:
    #                 token = tokens[cnt]
    #                 if args.without_context:
    #                     mention, title, overall_sent = parse_input_tokens_without_context(token)
    #                 else:
    #                     mention, title, overall_sent = parse_input_tokens(token)

    #                 output["data"].append(
    #                     {"true": true, "entity_title":title, "overall_text": overall_sent,
    #                      "BiEncoder": bi_sc, "CrossEncoder": sc}
    #                 )

    #             bi_true = bi_true or true
    #             if sc > sc_score:
    #                 sc_score = sc
    #                 sc_true = true

    #             output["meta"] = {
    #                 "mention": mention,
    #                 "generation": bi_true, "ranking": sc_true}

    #             output["data"].sort(key=lambda x: x["CrossEncoder"])
    #             outputs.append(json.dumps(output))


    #         with open(args.output_path + f'/{category}.jsonl', 'w') as f:
    #             f.write("\n".join(outputs))

    #         recall = calculate_recall(trues, cross_preds, ks=[1, 10, 30, 50, 100])
    #         print(category, recall)
    #         """


    # else:
    #     mention_dataset = ShinraDataset(args.mention_dataset, args.category, mention_tokenizer, max_ctxt_len=args.max_ctxt_len, without_context=args.without_context)

    #     recall = model.evaluate(mention_dataset)
    #     print(args.category, recall)


if __name__ == "__main__":
    """
    prf = LineProfiler()
    prf.add_function(BertCandidateGenerator.train)
    prf.runcall(main)
    prf.print_stats()
    #cProfile.run('main()', filename="main.prof")
    """
    main()

