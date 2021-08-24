import sys
sys.path.append("../")
import json
import pickle
import random
import fasteners

from torch.utils.data import Dataset


class EntityLinkingDataset(Dataset):
    def __init__(self, data, tokenizer, max_ctxt_len):
        self.tokenizer = tokenizer
        self.max_ctxt_len = max_ctxt_len
        self.data = data

    def _preprocess(self, line):
        left_ctxt = line['left_context']
        mention_tokens = line['mention']
        right_ctxt = line['right_context']

        input_seq = ['[CLS]'] + left_ctxt[-self.max_ctxt_len:] + ['[M]'] + mention_tokens + ['[/M]'] + right_ctxt[:self.max_ctxt_len] + ['[SEP]']
        input_seq = self.tokenizer.convert_tokens_to_ids(input_seq)
        input_label = line['link_page_id']
        return input_seq, input_label

    def __getitem__(self, item):
        input_seq, input_label = self._preprocess(self.data[item])
        return input_seq, input_label

    def __len__(self):
        return len(self.data)


class CandidateDataset(object):
    def __init__(self, input_file, tokenizer, preprocessed=False, without_context=False):
        self.tokenizer = tokenizer
        self.data = self._read(input_file, preprocessed)
        self.CLS = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.SEP = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.without_context = without_context

    def save_preprocessed_data(self, fn):
        with open(fn, 'wb') as f:
            pickle.dump(self.data, f)

    def _preprocess(self, title, description):
        title = self.tokenizer.tokenize(title)
        description = self.tokenizer.tokenize(description)

        if len(title) > self.max_title_len:
            self.max_title_len = len(title)
        if len(description) > self.max_desc_len:
            self.max_desc_len = len(description)

        title = self.tokenizer.convert_tokens_to_ids(title)
        description = self.tokenizer.convert_tokens_to_ids(description)
        return title, description

    def _read(self, fn, preprocessed=False):
        data = {}
        if preprocessed:
            with open(fn, 'rb') as f:
                data = pickle.load(f)
            return data

        with open(fn, 'r') as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue

                line = json.loads(line)
                title, desc = self._preprocess(line['title'], line['description'])
                data[int(line['id'])] = {
                    "title": line['title'],
                    "description": line['description'],
                    'title_ids': title,
                    'description_ids': desc
                }

        return data

    def get_pages(self, page_ids, max_title_len=50, max_desc_len=100):
        input_seq =  [self.data[int(page_id)] for page_id in page_ids]
        results = []
        for seq in input_seq:
            result = [self.CLS]
            if max_title_len == -1:
                result += seq['title_ids']
            else:
                result += seq['title_ids'][:max_title_len]

            result.append(self.SEP)

            if not self.without_context:
                if max_desc_len == -1:
                    result += seq['description_ids'][:511-len(result)]
                else:
                    result += seq['description_ids'][:max_desc_len]

                result.append(self.SEP)

            #[result.append(0) for i in range(max(0,300-len(result)))]

            results.append(result)

        return results


class MentionDataset(Dataset):
    def __init__(self, fn, index, tokenizer, preprocessed, max_ctxt_len=32, return_json=False, without_context=False):
        self.fn = fn
        self.f = open(fn, 'r')
        self.index = index
        self.datasize = self.index.shape[0] - 1
        self.tokenizer = tokenizer
        self.preprocessed = preprocessed
        self.max_ctxt_len = max_ctxt_len
        self.return_json = return_json
        self.without_context = without_context

    def __del__(self):
        self.f.close()

    def __len__(self):
        return self.datasize

    def __getitem__(self, item):
        with fasteners.InterProcessLock(self.fn):
            self.f.seek(self.index[item])
            line = self.f.read(self.index[item+1] - self.index[item])[:-1]
        line = json.loads(line)
        input_seq, input_label = self._preprocess(line)
        if self.return_json:
            return input_seq, input_label, line

        return input_seq, input_label

    def _preprocess(self, line):
        if self.preprocessed:
            left_ctxt = line['left_ctxt_tokens']
            mention_tokens = line['mention_tokens']
            right_ctxt = line['right_ctxt_tokens']
        else:
            left_ctxt = self.tokenizer.tokenize(line['left_context'])
            mention_tokens = self.tokenizer.tokenize(line['mention'])
            right_ctxt = self.tokenizer.tokenize(line['right_context'])

        if self.without_context:
            input_seq = ['[CLS]'] + mention_tokens + ['[SEP]']
        else:
            input_seq = ['[CLS]'] + left_ctxt[-self.max_ctxt_len:] + ['[M]'] + mention_tokens + ['[/M]'] + right_ctxt[:self.max_ctxt_len] + ['[SEP]']
        input_seq = self.tokenizer.convert_tokens_to_ids(input_seq)
        #[input_seq.append(0) for i in range(max(0,200-len(input_seq)))]
        input_label = line['linkpage_id']
        return input_seq, input_label


class MentionDataset2(object):
    def __init__(self, input_file, tokenizer, preprocessed):
        self.input_file = input_file
        self.tokenizer = tokenizer
        self.preprocessed = preprocessed

    def _preprocess(self, line, max_ctxt_len=32):
        if self.preprocessed:
            left_ctxt = line['left_ctxt_tokens']
            mention_tokens = line['mention_tokens']
            right_ctxt = line['right_ctxt_tokens']
        else:
            left_ctxt = self.tokenizer.tokenize(line['left_context'])
            mention_tokens = self.tokenizer.tokenize(line['mention'])
            right_ctxt = self.tokenizer.tokenize(line['right_context'])

        input_seq = ['[CLS]'] + left_ctxt[-max_ctxt_len:] + ['[M]'] + mention_tokens + ['[/M]'] + right_ctxt[:max_ctxt_len] + ['[SEP]']
        input_seq = self.tokenizer.convert_tokens_to_ids(input_seq)
        #[input_seq.append(0) for i in range(max(0,200-len(input_seq)))]
        input_label = line['linkpage_id']
        return input_seq, input_label

    def batch(self, batch_size=16, random_bsz=100000, max_ctxt_len=32, return_json=False):
        batch_input, batch_labels = [], []
        if return_json:
            batch_lines = []
        with open(self.input_file, 'r') as f:
            for line in f:
                if len(batch_input) >= random_bsz:
                    random_idx = [i for i in range(len(batch_input))]
                    random.shuffle(random_idx)

                    for batch_idx in range(0, len(batch_input), batch_size):
                        end_batch_idx = min(batch_idx+batch_size, len(batch_input))
                        inbatch_input = [batch_input[random_idx[i]] for i in range(batch_idx, end_batch_idx)]
                        inbatch_labels = [batch_labels[random_idx[i]] for i in range(batch_idx, end_batch_idx)]

                        if return_json:
                            inbatch_lines = [batch_lines[random_idx[i]] for i in range(batch_idx, end_batch_idx)]
                            yield inbatch_input, inbatch_labels, inbatch_lines
                        else:
                            yield inbatch_input, inbatch_labels
                    batch_input, batch_labels = [], []
                    if return_json:
                        batch_lines = []

                line = line.rstrip()
                if not line:
                    continue
                line = json.loads(line)

                ids, labels = self._preprocess(line)
                batch_input.append(ids)
                batch_labels.append(labels)

                if return_json:
                    batch_lines.append(line)


            if len(batch_input) > 0:
                random_idx = [i for i in range(len(batch_input))]
                random.shuffle(random_idx)

                for batch_idx in range(0, len(batch_input), batch_size):
                    end_batch_idx = min(batch_idx+batch_size, len(batch_input))
                    inbatch_input = [batch_input[random_idx[i]] for i in range(batch_idx, end_batch_idx)]
                    inbatch_labels = [batch_labels[random_idx[i]] for i in range(batch_idx, end_batch_idx)]
                    if return_json:
                        inbatch_lines = [batch_lines[random_idx[i]] for i in range(batch_idx, end_batch_idx)]
                        yield inbatch_input, inbatch_labels, inbatch_lines
                    else:
                        yield inbatch_input, inbatch_labels

def my_collate_fn(batch):
    tokens, tags = list(zip(*batch))
    return tokens, tags


def my_collate_fn_json(batch):
    tokens, tags, lines = list(zip(*batch))
    return tokens, tags, lines
