import argparse
import json

from transformers import AutoTokenizer

from link import link_using_atag, exact_match, find_reccursive_page
from utils.dataset import ShinraData
from utils.util import load_title2id

def parse_args():
    parser = argparse.ArgumentParser()

    # for dataset
    parser.add_argument("--input_path", type=str, help="mention dataset path")
    parser.add_argument("--category", type=str, help="mention dataset path")
    parser.add_argument("--output_path", type=str, help="mention dataset path")
    parser.add_argument("--title2id", type=str, help="mention dataset path")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    title2id = load_title2id(args.title2id)
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    dataset = ShinraData.from_linkjp_format(
        input_path=args.input_path,
        category=args.category,
        tokenizer=tokenizer
    )

    with open(args.output_path, "w") as f:
        for data in dataset:
            link_data = data.entity_linking_inputs
            for d in link_data:
                html_text = d['annotation']['html_offset']['text']
                page = link_using_atag(html_text, title2id)
                if page is None:
                    page = find_reccursive_page(d['annotation'], args.category, None)
                if page is None:
                    page = exact_match(d['annotation']['text_offset']['text'], title2id)

                if page is not None:
                    output_dict = d['annotation']
                    output_dict['page_id'] = str(output_dict['page_id'])
                    output_dict['link_page_id'] = str(page)

                    f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')
