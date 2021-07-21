from pathlib import Path

import numpy as np
import faiss

class NearestNeighborSearch(object):
    def __init__(self, dim, metric="dot", use_gpu=False):
        self.dim = dim
        self.use_gpu = use_gpu
        if metric == "dot":
            self.index = faiss.IndexFlatIP(self.dim)
        elif metric == "euclid":
            self.index = faiss.IndexFlatL2(self.dim)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.page_ids = []

    def load_index(self, save_dir):
        save_dir = Path(save_dir)
        self.index = faiss.read_index(str(save_dir / "index.model"))
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        with open(save_dir / "pages.txt", 'r') as f:
            self.page_ids = [int(l) for l in f.read().split("\n") if l != ""]

    def save_index(self, save_dir):
        save_dir = Path(save_dir)
        if self.use_gpu:
            self.index = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(self.index, str(save_dir / "index.model"))

        with open(save_dir / "pages.txt", 'w') as f:
            f.write('\n'.join([str(l) for l in self.page_ids]))

    def add_entries(self, reps, ids):
        self.index.add(reps)
        self.page_ids.extend(ids)

    def search(self, queries, k):
        D, I = self.index.search(queries, k)
        candidates = [[self.page_ids[i] for i in ii] for ii in I]
        return candidates, D

if __name__ == "__main__":
    xb = np.random.random((1000, 768)).astype("float32")
    xq = np.random.random((10, 768)).astype("float32")
    page_ids = [str(i) for i in range(1000)]

    searcher = NearestNeighborSearch(768, use_gpu=True)
    searcher.add_entries(xb, page_ids)
    candidates = searcher.search(xb[:10], 10)
    print(candidates)

    searcher.save_index("/home/is/ujiie/wiki_en/models/base_bert_index")
