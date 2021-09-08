import sys
import re
sys.path.append("../../")

from transformers import AutoTokenizer

from utils.dataset import ShinraData
from utils.util import load_title2id

pattern = '<a href=".*?" title="(.*?)">'

def link_using_atag(html_text, title2id):
    link_pages = re.findall(pattern, html_text)
    if link_pages:
        if len(link_pages) > 1:
            return None

        title = link_pages[-1]
        title = title.replace(" ", "_")
        if title in title2id:
            return title2id[title]
    return None


def exact_match(mention, title2id):
    mention = mention.replace(" ", "_")
    if mention in title2id:
        return title2id[mention]
    return None


def find_reccursive_page(ann, category, rec_list):
    attr = ann['attribute']
    if (category, attr) in rec_list:
        return ann['page_id']
    return None


if __name__ == "__main__":
    rec_list = [
        ("Airport", "別名"),
        ("City", "別名"),
        ("Company", "別名"),
        ("Conference", "別名"),
        ("Lake", "別名"),
        ("Person", "別名"),
        ("Compound", "別称"),
        ("Compound", "商標名"),
        ("Airport", "旧称"),
        ("City", "旧称"),
        ("Conference", "旧称・前身"),
        ("Company", "起源"),
        ("Company", "過去の社名"),
    ]
    title2id = load_title2id("/data1/ujiie/wiki_resource/title2page.csv")
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    dataset = ShinraData.from_linkjp_format(
        input_path="/data1/ujiie/shinra/EN/linkjp-sample-210402",
        category="Airport",
        tokenizer=tokenizer
    )
    s = 0
    l = 0
    for data in dataset:
        category = data.category
        link_data = data.entity_linking_inputs
        for d in link_data:
            html_text = d['annotation']['html_offset']['text']
            s += 1
            page = link_using_atag(html_text, title2id)
            if page is None:
                page = exact_match(d['annotation']['text_offset']['text'], title2id)
            if page is None:
                page = find_reccursive_page(d['annotation'], category, rec_list)
            l += int(page is not None)
    print(l, s)
