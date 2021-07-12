import copy
import json
from collections import defaultdict


class IOB(object):
    def __init__(self, iob_items, page_id, line_id, is_train):
        self.iob_items = iob_items
        self.page_id = page_id
        self.line_id = line_id
        self.is_train = is_train

    @property
    def output_format(self):
        return "\n".join(
            [str(self.page_id), str(self.line_id)]
            + [i.output_format for i in self.iob_items]
        )


class IOBItem(object):
    def __init__(self, token, offset):
        self.token = token
        self.offset = offset
        self.label = []

    def set_label(self, label):
        self.label.append(label)

    def fill_label(self, is_train=True):
        fill_label = "O" if is_train else ""
        self.label = "|".join(self.label) if self.label != [] else fill_label

    @property
    def output_format(self):
        return "\t".join([self.token, self.offset[0], self.offset[1], self.label])


def split_iobs(iob_items, info, split_idx, is_train):
    results = []
    sent_iobs = []

    for iob_item in iob_items:
        sent_iobs.append(iob_item)
        if iob_item.token in split_idx:
            results.append(IOB(sent_iobs, *info, is_train))
            sent_iobs = []
    if sent_iobs != []:
        results.append(IOB(sent_iobs, *info, is_train))

    return results


def load_annotation(path):
    category = str(path.stem)
    fin = path / (category + "_dist.json")
    annotations = {}
    with open(fin, "r") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            line = json.loads(line)
            page_id = int(line["page_id"])
            if page_id not in annotations:
                annotations[page_id] = defaultdict(list)
            # annotations[line['token_offset']['start']['line_id']].append(line)

            if "token_offset" not in line:
                continue
            elif (
                line["token_offset"]["start"]["line_id"]
                != line["token_offset"]["end"]["line_id"]
            ):
                first_entity = copy.deepcopy(line)
                first_entity["token_offset"]["end"]["line_id"] = first_entity[
                    "token_offset"
                ]["start"]["line_id"]
                first_entity["token_offset"]["end"]["offset"] = -1
                annotations[page_id][line["token_offset"]["start"]["line_id"]].append(
                    first_entity
                )

                line["token_offset"]["start"]["line_id"] = line["token_offset"]["end"][
                    "line_id"
                ]
                line["token_offset"]["start"]["offset"] = 0
                line["continue"] = True
                annotations[page_id][line["token_offset"]["start"]["line_id"]].append(
                    line
                )
            else:
                annotations[page_id][line["token_offset"]["start"]["line_id"]].append(
                    line
                )

    return annotations


def load_tokens(path, annotations, split_idx=set()):
    page_id = int(path.stem)
    iobs = []

    if page_id in annotations:
        annotation = annotations[page_id]
    else:
        annotation = None
    is_train = annotation is not None

    with open(path, "r") as f:
        for line_id, line in enumerate(f):
            line = line.rstrip()
            if not line:
                continue
            line = line.split(" ")
            line = [l.split(",") for l in line]

            sent_iobs = [IOBItem(l[0], (l[1], l[2])) for l in line]
            # sent_iobs = [IOBItem(l[0], (l[1], l[2])) for l in line]

            if annotation is not None:
                if line_id in annotation:

                    for ann in annotation[line_id]:
                        for idx, i in enumerate(
                            range(
                                ann["token_offset"]["start"]["offset"],
                                ann["token_offset"]["end"]["offset"],
                            )
                        ):
                            prefix = (
                                "B-" if idx == 0 and "continue" not in ann else "I-"
                            )
                            sent_iobs[i].set_label(prefix + ann["attribute"])

            [i.fill_label(is_train) for i in sent_iobs]
            sent_iobs = split_iobs(sent_iobs, [page_id, line_id], split_idx, is_train)
            iobs.extend(sent_iobs)

    return iobs


def load_vocab(path):
    with open(path, "r") as f:
        vocab = [line for line in f.read().split("\n") if line != ""]
    return vocab


def output_iobs(tokens, labels, offsets, path):
    iobs = [
        "\n".join(
            ["\t".join([t, o[0], o[1], l]) for t, l, o in zip(token, label, offset)]
        )
        for token, label, offset in zip(tokens, labels, offsets)
    ]
    with open(path, "w") as f:
        f.write("\n\n".join(iobs))


def convert_tokenized_to_conll(path, output_dir=None, single=False):
    if output_dir is None:
        output_dir = path

    annotations = load_annotation(path)
    vocab = load_vocab(path / "vocab.txt")
    word2idx = {w: idx for idx, w in enumerate(vocab)}

    token_files = (path / "tokens").glob("*.txt")
    train_iobs = []
    test_iobs = []

    for file_name in token_files:
        iobs = load_tokens(file_name, annotations, set([word2idx["。"]]), vocab, single)
        if iobs[0].is_train:
            train_iobs.extend(iobs)
        else:
            test_iobs.extend(iobs)

    with open(output_dir / "train.iob", "w") as f:
        f.write("\n\n".join([iob.output_format for iob in train_iobs]))
    with open(output_dir / "test.iob", "w") as f:
        f.write("\n\n".join([iob.output_format for iob in test_iobs]))

