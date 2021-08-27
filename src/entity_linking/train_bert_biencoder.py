#import cProfile
import sys
sys.path.append('../')
from line_profiler import LineProfiler
import argparse
from logging import getLogger, StreamHandler, DEBUG, Formatter, FileHandler

import numpy as np
import mlflow
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.trainer_utils import set_seed
import apex
from apex import amp

from dataset import MentionDataset, CandidateDataset
from bert_generator import BertBiEncoder, BertCandidateGenerator
from utils.util import to_parallel, save_model

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="bert-name used for biencoder")
    parser.add_argument("--load_model_path", type=str, help="bert-name used for biencoder")
    parser.add_argument("--hard_negative", action="store_true", help="bert-name used for biencoder")
    parser.add_argument("--mention_dataset", type=str, help="mention dataset path")
    parser.add_argument("--mention_index", type=str, help="mention dataset path")
    parser.add_argument("--candidate_dataset", type=str, help="candidate dataset path")
    parser.add_argument("--path_for_NN", type=str, help="candidate dataset path")
    parser.add_argument("--model_path", type=str, help="model save path")
    parser.add_argument("--mention_preprocessed", action="store_true", help="whether mention_dataset is preprocessed")
    parser.add_argument("--candidate_preprocessed", action="store_true", help="whether candidate_dataset is preprocessed")
    parser.add_argument("--inbatch", action="store_true", help="whether using inbatch negative")
    parser.add_argument("--epochs", type=int, help="epochs")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--warmup_propotion", type=float, help="learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="learning rate")
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--bsz", type=int, help="batch size")
    parser.add_argument("--model_save_interval", default=None, type=int, help="batch size")
    parser.add_argument("--random_bsz", type=int, help="data size loaded at one time")
    parser.add_argument("--seed", type=int, help="data size loaded at one time")
    parser.add_argument("--max_ctxt_len", type=int, help="maximum context length")
    parser.add_argument("--max_title_len", type=int, help="maximum title length")
    parser.add_argument("--max_desc_len", type=int, help="maximum description length")
    parser.add_argument("--traindata_size", type=int, help="training datasize (for progress bar)")
    parser.add_argument("--without_context", action="store_true", help="whether using inbatch negative")
    parser.add_argument("--mlflow", action="store_true", help="whether using inbatch negative")
    parser.add_argument("--parallel", action="store_true", help="whether using inbatch negative")
    parser.add_argument("--fp16", action="store_true", help="whether using inbatch negative")
    parser.add_argument('--fp16_opt_level', type=str, default="O1")
    parser.add_argument("--logging", action="store_true", help="whether using inbatch negative")
    parser.add_argument("--log_file", type=str, help="whether using inbatch negative")

    args = parser.parse_args()
    set_seed(args.seed)

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

    index = np.load(args.mention_index)
    mention_dataset = MentionDataset(args.mention_dataset, index, mention_tokenizer, preprocessed=args.mention_preprocessed, return_json=True, without_context=args.without_context)
    #mention_dataset = MentionDataset2(args.mention_dataset, mention_tokenizer, preprocessed=args.mention_preprocessed)
    candidate_dataset = CandidateDataset(args.candidate_dataset, mention_tokenizer, preprocessed=args.candidate_preprocessed, without_context=args.without_context)

    mention_bert = AutoModel.from_pretrained(args.model_name)
    mention_bert.resize_token_embeddings(len(mention_tokenizer))
    candidate_bert = AutoModel.from_pretrained(args.model_name)

    biencoder = BertBiEncoder(mention_bert, candidate_bert)

    if args.load_model_path:
        biencoder.load_state_dict(torch.load(args.load_model_path))

    model = BertCandidateGenerator(biencoder, device, model_path=args.model_path, use_mlflow=args.mlflow, logger=logger)

    try:
        model.train(
            mention_dataset,
            candidate_dataset,
            inbatch=args.inbatch,
            lr=args.lr,
            batch_size=args.bsz,
            random_bsz=args.random_bsz,
            max_ctxt_len=args.max_ctxt_len,
            max_title_len=args.max_title_len,
            max_desc_len=args.max_desc_len,
            traindata_size=args.traindata_size,
            model_save_interval=args.model_save_interval,
            grad_acc_step=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            epochs=args.epochs,
            warmup_propotion=args.warmup_propotion,
            fp16=args.fp16,
            fp16_opt_level=args.fp16_opt_level,
            parallel=args.parallel,
            hard_negative=args.hard_negative,
            args=args,
            mention_tokenizer=mention_tokenizer,
        )
    except KeyboardInterrupt:
        pass

    save_model(model.model, args.model_path)
    #torch.save(model.model.state_dict(), args.model_path)

    if args.mlflow:
        mlflow.end_run()

if __name__ == "__main__":
    """
    prf = LineProfiler()
    prf.add_function(BertCandidateGenerator.train)
    prf.runcall(main)
    prf.print_stats()
    #cProfile.run('main()', filename="main.prof")
    """
    main()
