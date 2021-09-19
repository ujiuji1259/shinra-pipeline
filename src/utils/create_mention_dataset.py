from pathlib import Path
from xml.sax.saxutils import unescape
from urllib.parse import unquote_to_bytes
import re
import json

import numpy as np
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    # for model
    parser.add_argument("--wiki_dir", type=str, help="bert-name used for biencoder")
    parser.add_argument("--output_path", type=str, help="model save path")
    parser.add_argument("--index_path", type=str, help="model save path")
    parser.add_argument("--preprocess", action="store_true", help="model save path")
    parser.add_argument("--bert_name", type=str, help="model save path")
    args = parser.parse_args()

    return args

args = parse_args()

if args.preprocess:
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
base_dir = Path(args.wiki_dir)
file_set = base_dir.glob("**/wiki_*")

cnt = 0
index = [0]
alias_set = set()
with open(args.output_path, "w") as fout:
    for fn in file_set:
        sent = []
        with open(fn, 'r') as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue

                line = json.loads(line)
                doc = line["text"].split("\n")
                doc = [''] + [unescape(s) for s in doc if s != ''] + [""]
                plain_doc = [re.sub('<a href="(.*?)">(.*?)</a>', r'\2', t) for t in doc]
                for idx in range(1, len(doc)-1):
                    sent = doc[idx]
                    ite = re.finditer('<a href="(.*?)">(.*?)</a>', sent)
                    for it in ite:
                        link = unquote_to_bytes(it.groups()[0]).decode()
                        if link not in title2id:
                            continue
                        link = title2id[link]
                        mention = it.groups()[1]
                        start, end = it.span()
                        left_context = sent[:start]
                        left_context = re.sub('<a href="(.*?)">(.*?)</a>', r'\2', left_context)
                        right_context = sent[end:]
                        right_context = re.sub('<a href="(.*?)">(.*?)</a>', r'\2', right_context)
                        output = {
                            "pre_sent": plain_doc[idx-1],
                            "left_context": left_context,
                            "right_context": right_context,
                            "post_sent": plain_doc[idx+1],
                            "mention": mention,
                            "linkpage_id": link
                        }

                        if args.preprocess:
                            output['left_ctxt_tokens'] = tokenizer.tokenize(left_context)
                            output['mention_tokens'] = tokenizer.tokenize(mention)
                            output['right_ctxt_tokens'] = tokenizer.tokenize(right_context)


                        output_str = json.dumps(output, ensure_ascii=False) + '\n'
                        fout.write(output_str)

                        index.append(index[-1] + len(output_str))

index = np.array(index)
np.save(args.index_path, index)