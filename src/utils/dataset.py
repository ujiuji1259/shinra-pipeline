import sys
from collections import defaultdict
from pathlib import Path
import pickle
import json

from tqdm import tqdm
from transformers import AutoTokenizer

from utils.util import decode_iob, is_chunk_start

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

    return word_idxs, sub2word


class ShinraData(object):
    def __init__(self, attributes_path, params={}):
        with open(attributes_path, "rb") as f:
            self.attributes = pickle.load(f)
        self.attr2idx = {}
        for key, value in self.attributes.items():
            self.attr2idx[key] = {word: idx for idx, word in enumerate(value)}

        self.page_id = None
        self.page_title = None
        self.category = None
        self.plain_text = None
        self.tokens = None
        self.word_alignments = None
        self.sub2word = None
        self.text_offsets = None
        self.nes = None

        for key, value in params.items():
            setattr(self, key, value)

        if self.category is not None:
            self.attributes = self.attributes[self.category]
            self.attr2idx = self.attr2idx[self.category]

    @classmethod
    def from_shinra2020_format(
        cls,
        attributes_path=None,
        input_path=None):

        input_path = Path(input_path)
        category = input_path.stem

        anns = load_annotation(input_path / f"{category}_dist.json")
        vocab = load_vocab(input_path / "vocab.txt")

        docs = []
        for token_file in tqdm(input_path.glob("tokens/*.txt")):
            page_id = int(token_file.stem)
            tokens, text_offsets = load_tokens(token_file, vocab)

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
            }

            if page_id in anns:
                data["nes"] = anns[page_id]

            docs.append(cls(attributes_path, params=data))

        return docs

    # iobs = [sents1, sents2, ...]
    # sents1 = [[iob1_attr1, iob2_attr1, ...], [iob1_attr2, iob2_attr2, ...], ...]
    def add_nes_from_iob(self, iobs, valid_line_ids=None):
        self.nes = []
        if valid_line_ids is None:
            ite = enumerate(iob)
        else:
            ite = zip(valid_line_ids, iobs)

        for line_id, sent_iob in ite:
            word2subword = self.word_alignments[line_id]
            tokens = self.tokens[line_id]
            text_offsets = self.text_offsets[line_id]
            for iob, attr in zip(sent_iob, self.attributes):
                ne = {}
                iob = [0] + iob + [0]
                for token_idx in range(1, len(iob)):
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

                    if is_chunk_end(iob[token_idx-1], iob[token_idx]):
                        assert ne != {}
                        # token_idxは本来のものから+1されているので，word2subwordはneの外のはじめのtoken_id
                        end_offset = len(tokens) if token_idx == len(word2subword) else word2subword[token_idx]
                        ne["token_offset"]["end"] = {
                            "line_id": line_id,
                            "offset": end_offset
                        }
                        ne["token_offset"]["text"] = " ".join(tokens[ne["token_offset"]["start"]["offset"]:ne["token_offset"]["end"]["offset"]])

                        ne["text_offset"]["end"] = {
                            "line_id": line_id,
                            "offset": text_offsets[end_offset-1]
                        }

                        self.nes.append(ne)
                        ne = {}

    @property
    def ner_inputs(self):
        outputs = {}

        outputs["input_ids"] = self.tokens
        outputs["word_idxs"] = self.word_alignments.copy()

        if self.nes is not None:
            outputs["labels"] = self.iob

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
        iobs = [[["O" for _ in range(len(tokens))] for _ in range(len(self.attributes))] for tokens in self.word_alignments]
        for ne in self.nes:
            if "token_offset" not in ne:
                continue
            start_line = int(ne["token_offset"]["start"]["line_id"])
            start_offset = int(ne["token_offset"]["start"]["offset"])

            end_line = int(ne["token_offset"]["end"]["line_id"])
            end_offset = int(ne["token_offset"]["end"]["offset"])

            if start_line != end_line:
                continue

            # 正解となるsubwordを含むwordまでタグ付
            attr_idx = self.attr2idx[ne["attribute"]]
            ne_start = self.sub2word[start_line][start_offset]
            ne_end = self.sub2word[end_line][end_offset-1] + 1

            for idx in range(ne_start, ne_end):
                iobs[start_line][attr_idx][idx] = "B" if idx == ne_start else "I"

        return iobs


if __name__ == "__main__":
    dataset = ShinraData.from_shinra2020_format("/data1/ujiie/shinra/tohoku_bert/attributes.pickle", Path("/data1/ujiie/shinra/tohoku_bert/Event/Event_Other"))
    print(dataset[5].ner_inputs)
