from pathlib import Path

import numpy as np
import faiss

class NearestNeighborSearch(object):
    def __init__(self, dim, total_size, metric="dot", use_gpu=False, gpu_id=0, ivf=False):
        self.dim = dim
        self.use_gpu = use_gpu
        self.ivf = ivf
        if metric == "dot":
            index = faiss.IndexFlatIP(self.dim)
            index_metric = faiss.METRIC_INNER_PRODUCT
        elif metric == "euclid":
            index = faiss.IndexFlatL2(self.dim)
            index_metric = faiss.METRIC_L2

        if ivf:
            self.index = faiss.IndexIVFFlat(index, d, nlist, index_metric)
        else:
            self.index = index


        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, gpu_id, self.index)

        self.page_ids = []
        self.reps = np.empty((total_size, dim), dtype=np.float32)
        #self.reps = np.random.rand(total_size, dim).astype(np.float32)
        self.cnt = 0

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
        # self.index.add(reps)
        self.reps[self.cnt:self.cnt+reps.shape[0]] = reps
        self.cnt += reps.shape[0]
        self.page_ids.extend(ids)

    def finish_add_entry(self):
        if self.ivf:
            self.index.train(self.reps)
        self.index.add(self.reps)
        #self.page_ids.extend(["0" for _ in range(self.reps.shape[0] - len(self.page_ids))])

        del self.reps

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

    searcher.save_index("/home/is/ujiie/wiki_en/models/base_bert_index")
