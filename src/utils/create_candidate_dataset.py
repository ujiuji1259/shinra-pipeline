from pathlib import Path
import csv
import re
import json
from xml.sax.saxutils import unescape
from urllib.parse import unquote_to_bytes


def parse_args():
    parser = argparse.ArgumentParser()
    # for model
    parser.add_argument("--wiki_dir", type=str, help="bert-name used for biencoder")
    parser.add_argument("--output_path", type=str, help="model save path")
    args = parser.parse_args()

    return args


base_dir = Path(args.wiki_dir)
file_set = base_dir.glob("**/wiki_*")


title2id = {}
with open(args.output_path, "w") as fout:
    for fn in file_set:
        with open(fn, 'r') as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue


                line = json.loads(line)

                # remove redirect page
                if int(line['id']) not in id_set:
                    continue

                _id = line['id']
                title = line['title']

                text = line['text']
                text = text.split("\n")[0]
                text = unescape(text)
                text = re.sub('<a href="(.*?)">(.*?)</a>', r'\2', text)

                title2id[line['title']] = line['id']

                fout.write(json.dumps({"id": _id, "title": title, "description": text}) + '\n')
