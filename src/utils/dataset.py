import sys
sys.path.append("..")
import json
from collections import defaultdict
from pathlib import Path
import pickle
import json

from tqdm import tqdm
from transformers import AutoTokenizer

from utils.util import decode_iob, is_chunk_start, is_chunk_end
from utils.shinra_tokenizer import tokenize_sent, annotation_mapper

def load_tokens(path, vocab):
    tokens = []
    text_offsets = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip().split()
            line = [l.split(",") for l in line]
            tokens.append([vocab[int(l[0])] for l in line])
            text_offsets.append([[l[1], l[2]] for l in line])

    return tokens, text_offsets


def load_vocab(path):
    vocab = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            vocab.append(line)
    return vocab


def load_annotation(path):
    ann = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            line = json.loads(line)
            line["page_id"] = int(line["page_id"])
            ann[line["page_id"]].append(line)
    return ann


def find_word_alignment(tokens):
    word_idxs = []
    sub2word = {}
    for idx, token in enumerate(tokens):
        if not token.startswith("##"):
            word_idxs.append(idx)
        sub2word[idx] = len(word_idxs) - 1

    # add word_idx for end offset
    if len(tokens) > 0:
        word_idxs.append(len(tokens))
        sub2word[len(tokens)] = len(word_idxs) - 1

    return word_idxs, sub2word


class ShinraData(object):
    def __init__(self, attributes, params={}):
        self.attributes = attributes
        if self.attributes is not None:
            self.attr2idx = {attr: idx for idx, attr in enumerate(self.attributes)}

        self.page_id = None
        self.page_title = None
        self.category = None
        self.plain_text = None
        self.tokens = None
        self.word_alignments = None
        self.sub2word = None
        self.text_offsets = None
        self.valid_line_ids = None
        self.nes = None
        self.link = None

        for key, value in params.items():
            setattr(self, key, value)

    @classmethod
    def from_plain_wiki(
        cls,
        attributes_path=None,
        category=None,
        input_path=None,
        tokenizer=None,
    ):
        base_attribute_path = Path(attribute_path)
        attribute_path = base_attribute_path / f"{category}.txt"
        with open(attribute_path, "r") as f:
            attributes = [attr for attr in f.read().split("\n") if attr != '']
        yield attributes

        files = Path(input_path).glob("*.txt")
        for fn in files:
            page_id = int(fn.stem)
            tokens = []
            text_offsets = []
            sub2words = []
            word_alignments = []
            with open(fn, 'r') as f:
                for line in f:
                    line = line.rstrip()
                    if not line:
                        tokens.append([])
                        text_offsets.append([])
                        sub2words.append([])
                        word_alignments.append([])

                    subwords, offset = tokenize_sent(sent, tokenizer)
                    word_alignment, sub2word = find_word_alignment(subwords)

                    tokens.append(subwords)
                    text_offsets.append(offset)
                    sub2words.append(sub2word)
                    word_alignments.append(word_alignment)

                    # find title
                    if line_id == 4:
                        pos = sent.find("-jawiki")
                        title = sent[:pos]

                data = {
                    "page_id": page_id,
                    "page_title": title,
                    "category": category,
                    "tokens": tokens,
                    "text_offsets": text_offsets,
                    "word_alignments": word_alignments,
                    "sub2word": sub2word,
                }

                yield cls(attributes, params=data)

    @classmethod
    def from_linkjp_format(
        cls,
        input_path=None,
        category=None,
        tokenizer=None):

        input_path = Path(input_path)
        text_path = input_path / "plain" / category

        anns = load_annotation(input_path / "ene_annotation" / f"{category}.json")

        for text_file in tqdm(text_path.glob("*.txt")):
            page_id = int(text_file.stem)

            tokens = []
            text_offsets = []
            sub2words = []
            word_alignments = []
            annotation_mappers = []
            with open(text_file, "r") as f:
                for line_id, sent in enumerate(f):
                    sent = sent.rstrip()
                    if not sent:
                        tokens.append([])
                        text_offsets.append([])
                        sub2words.append([])
                        word_alignments.append([])
                        annotation_mappers.append([])
                        continue

                    subwords, offset = tokenize_sent(sent, tokenizer)
                    annotation_mappers.append(list(zip(subwords, [o[0] for o in offset], [o[1] for o in offset])))
                    word_alignment, sub2word = find_word_alignment(subwords)

                    tokens.append(subwords)
                    text_offsets.append(offset)
                    sub2words.append(sub2word)
                    word_alignments.append(word_alignment)

                    # find title
                    if line_id == 4:
                        pos = sent.find("-jawiki")
                        title = sent[:pos]

                data = {
                    "page_id": page_id,
                    "page_title": title,
                    "category": category,
                    "tokens": tokens,
                    "text_offsets": text_offsets,
                    "word_alignments": word_alignments,
                    "sub2word": sub2word,
                }

                if page_id in anns:
                    nes, _ = annotation_mapper(anns[page_id], annotation_mappers)
                    data["nes"] = nes

            yield cls(None, params=data)

    @classmethod
    def from_shinra2020_format(
        cls,
        input_path=None,
        get_attributes=False,
        attribute_path=None):

        input_path = Path(input_path)
        category = input_path.stem

        if attribute_path is None:
            base_attribute_path = input_path.parent.parent / "attributes"
        else:
            base_attribute_path = Path(attribute_path)

        attribute_path = base_attribute_path / f"{category}.txt"

        anns = load_annotation(input_path / f"{category}_dist.json")
        vocab = load_vocab(input_path / "vocab.txt")

        # create attributes
        if attribute_path.exists():
            with open(attribute_path, "r") as f:
                attributes = [attr for attr in f.read().split("\n") if attr != '']
        else:
            attributes = set()
            for page_id, ann in anns.items():
                attributes.update([a["attribute"] for a in ann if "attribute" in a])
            attributes = list(attributes)
            with open(attribute_path, "w") as f:
                f.write("\n".join(attributes))

        if get_attributes:
            yield attributes

        docs = []
        for token_file in tqdm(input_path.glob("tokens/*.txt")):
            page_id = int(token_file.stem)
            tokens, text_offsets = load_tokens(token_file, vocab)
            valid_line_ids = [idx for idx, token in enumerate(tokens) if len(token) > 0]

            # find title
            title = "".join([t[2:] if t.startswith("##") else t for t in tokens[4]])
            pos = title.find("-jawiki")
            title = title[:pos]

            # find word alignments = start positions of words
            word_alignments = [find_word_alignment(t) for t in tokens]
            sub2word = [w[1] for w in word_alignments]
            word_alignments = [w[0] for w in word_alignments]

            data = {
                "page_id": page_id,
                "page_title": title,
                "category": category,
                "tokens": tokens,
                "text_offsets": text_offsets,
                "word_alignments": word_alignments,
                "sub2word": sub2word,
                "valid_line_ids": valid_line_ids,
            }

            if page_id in anns:
                data["nes"] = anns[page_id]

            yield cls(attributes, params=data)

    def add_linkpage(self, pages):
        for ne, page in zip(self.nes, pages):
            ne["link_page_id"] = int(page)

    # iobs = [sents1, sents2, ...]
    # sents1 = [[iob1_attr1, iob2_attr1, ...], [iob1_attr2, iob2_attr2, ...], ...]
    def add_nes_from_iob(self, iobs):
        assert len(iobs) == len(self.valid_line_ids), f"{len(iobs)}, {len(self.valid_line_ids)}"
        self.nes = []

        for line_id, sent_iob in zip(self.valid_line_ids, iobs):
            word2subword = self.word_alignments[line_id]
            tokens = self.tokens[line_id]
            text_offsets = self.text_offsets[line_id]
            for iob, attr in zip(sent_iob, self.attributes):
                ne = {}
                iob = [0] + iob + [0]
                for token_idx in range(1, len(iob)):
                    if is_chunk_end(iob[token_idx-1], iob[token_idx]):
                        assert ne != {}
                        # token_idx????????????????????????+2????????????????????????word2subword???ne?????????????????????token_id
                        end_offset = len(tokens) if token_idx - 1 >= len(word2subword) else word2subword[token_idx-1]
                        # end_offset = len(tokens) if token_idx >= len(word2subword) else word2subword[token_idx-1]
                        ne["token_offset"]["end"] = {
                            "line_id": line_id,
                            "offset": end_offset
                        }
                        ne["token_offset"]["text"] = " ".join(tokens[ne["token_offset"]["start"]["offset"]:ne["token_offset"]["end"]["offset"]])

                        ne["text_offset"]["end"] = {
                            "line_id": line_id,
                            "offset": text_offsets[end_offset-1][1]
                        }
                        ne["page_id"] = self.page_id
                        ne["title"] = self.page_title

                        self.nes.append(ne)
                        ne = {}

                    if is_chunk_start(iob[token_idx-1], iob[token_idx]):
                        ne["attribute"] = attr
                        ne["token_offset"] = {
                            "start": {
                                "line_id": line_id,
                                "offset": word2subword[token_idx-1]
                            }
                        }
                        ne["text_offset"] = {
                            "start": {
                                "line_id": line_id,
                                "offset": text_offsets[word2subword[token_idx-1]][0]
                            }
                        }

    @property
    def entity_linking_inputs(self):
        dataset = []
        for ne in self.nes:
            mention = ne['text_offset']['text']
            start_line, start_off = ne['token_offset']['start']['line_id'], ne['token_offset']['start']['offset']
            end_line, end_off = ne['token_offset']['end']['line_id'], ne['token_offset']['end']['offset']

            if start_line == end_line:
                mention_tokens = self.tokens[start_line][start_off:end_off]
                left_context = [token for tokens in self.tokens[:start_line] for token in tokens] + self.tokens[start_line][:start_off]
                right_context = self.tokens[start_line][end_off:] + [token for tokens in self.tokens[start_line+1:] for token in tokens]
            else:
                mention_tokens = self.tokens[start_line][start_off:] + self.tokens[end_line][:end_off]
                left_context = [token for tokens in self.tokens[:start_line] for token in tokens] + self.tokens[start_line][:start_off]
                right_context = self.tokens[end_line][end_off:] + [token for tokens in self.tokens[end_line+1:] for token in tokens]

            dataset.append({
                "left_context": left_context,
                "mention": mention_tokens,
                "right_context": right_context,
                "link_page_id": int(ne["link_page_id"]) if "link_page_id" in ne else None,
                "annotation": ne
            })

        return dataset


    @property
    def ner_inputs(self):
        outputs = []
        iobs = self.iob if self.nes is not None else None
        for idx in self.valid_line_ids:
            sent = {
                "tokens": self.tokens[idx],
                "word_idxs": self.word_alignments[idx],
                "labels": iobs[idx] if iobs is not None else None
            }
            outputs.append(sent)

        # outputs["input_ids"] = self.tokens
        # outputs["word_idxs"] = self.word_alignments.copy()

        # if self.nes is not None:
        #     outputs["labels"] = self.iob
        # else:
        #     outputs["labels"] = [None for i in range(len(self.tokens))]

        return outputs

    @property
    def words(self):
        all_words = []
        for tokens, word_alignments in zip(self.tokens, self.word_alignments):
            words = []
            prev_idx = 0
            for idx in word_alignments[1:] + [-1]:
                inword_subwords = tokens[prev_idx:idx]
                inword_subwords = [s[2:] if s.startswith("##") else s for s in inword_subwords]
                words.append("".join(inword_subwords))
                prev_idx = idx
            all_words.append(words)
        return all_words

    @property
    def iob(self):
        """
        %%% IOB for ** only word-level iob2 tag **
        iobs = [sent, sent, ...]
        sent = [[Token1_attr1_iob, Token2_attr1_iob, ...], [Token1_attr2_iob, Token2_attr2_iob, ...], ...]
        {"O": 0, "B": 1, "I": 2}
        """
        iobs = [[["O" for _ in range(len(tokens)-1)] for _ in range(len(self.attributes))] for tokens in self.word_alignments]
        for ne in self.nes:
            if "token_offset" not in ne:
                continue
            start_line = int(ne["token_offset"]["start"]["line_id"])
            start_offset = int(ne["token_offset"]["start"]["offset"])

            end_line = int(ne["token_offset"]["end"]["line_id"])
            end_offset = int(ne["token_offset"]["end"]["offset"])

            # ???????????????entity?????????
            if start_line != end_line:
                continue

            # ???????????????subword?????????word???????????????
            attr_idx = self.attr2idx[ne["attribute"]]
            ne_start = self.sub2word[start_line][start_offset]
            ne_end = self.sub2word[end_line][end_offset-1] + 1

            for idx in range(ne_start, ne_end):
                iobs[start_line][attr_idx][idx] = "B" if idx == ne_start else "I"

        return iobs


if __name__ == "__main__":
    #dataset = ShinraData.from_shinra2020_format("/data1/ujiie/shinra/tohoku_bert/attributes.pickle", Path("/data1/ujiie/shinra/tohoku_bert/Event/Event_Other"))
    #print(dataset[5].ner_inputs)
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    dataset = ShinraData.from_one_doc_json("/data1/ujiie/shinra/tohoku_bert/attributes.pickle", "sample.json", tokenizer)
    print(dataset.tokens)
    print(dataset.plain_text)
    print(dataset.word_alignments)
    print(dataset.text_offsets)
