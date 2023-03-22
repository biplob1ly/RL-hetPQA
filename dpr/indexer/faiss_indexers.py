import logging
import pickle
from typing import List, Tuple

import faiss
import numpy as np

logger = logging.getLogger()


class DenseIndexer(object):

    def __init__(self, buffer_size: int = 500):
        self.buffer_size = buffer_size
        self.index = None

    def index_data(self, ctx_vectors: np.array, ctx_ids: np.array):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_ctx_count: int) -> Tuple[np.array, np.array]:
        raise NotImplementedError

    # def serialize(self, file: str):
    #     logger.info('Serializing index to %s', file)
    #
    #     index_file = file + '.index.dpr'
    #     meta_file = file + '.index_meta.dpr'
    #
    #     faiss.write_index(self.index, index_file)
    #     with open(meta_file, mode='wb') as f:
    #         pickle.dump(self.index_id_to_db_id, f)
    #
    # def deserialize_from(self, file: str):
    #     logger.info('Loading index from %s', file)
    #
    #     index_file = file + '.index.dpr'
    #     meta_file = file + '.index_meta.dpr'
    #
    #     self.index = faiss.read_index(index_file)
    #     logger.info('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)
    #
    #     with open(meta_file, "rb") as reader:
    #         self.index_id_to_db_id = pickle.load(reader)
    #     assert len(
    #         self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'
    #
    # def _update_id_mapping(self, db_ids: List):
    #     self.index_id_to_db_id.extend(db_ids)


class DenseFlatIndexer(DenseIndexer):

    def __init__(self, vector_sz: int, buffer_size: int = 500):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(vector_sz))

    def index_data(self, ctx_vectors: np.array, ctx_ids: np.array):
        ctx_vectors = ctx_vectors.astype('float32')
        ctx_ids = ctx_ids.astype('int64')
        n = len(ctx_vectors)
        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            vectors = ctx_vectors[i:i+self.buffer_size]
            ids = ctx_ids[i:i+self.buffer_size]
            self.index.add_with_ids(vectors, ids)

    def search_knn(self, query_vectors: np.array, top_ctx_count: int) -> Tuple[np.array, np.array]:
        query_vectors = query_vectors.astype('float32')
        scores_arr, ctx_ids_arr = self.index.search(query_vectors, top_ctx_count)
        return scores_arr, ctx_ids_arr
