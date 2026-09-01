"""Swappable sources for group relevance scores (the vector `r` in the spatial-boost
prior), replacing hand-curated lookups like MalaCards gene-disease relevance with a
generic, pluggable interface that can be backed by an LLM or embedding API.
"""
from abc import ABC, abstractmethod
import numpy as np


class RelevanceSource(ABC):
    @abstractmethod
    def score(self, group_descriptions, query):
        """Return an array of relevance scores, one per entry of `group_descriptions`,
        for a natural-language `query` describing the trait/outcome of interest."""


class TextEmbeddingRelevance(RelevanceSource):
    """Relevance via cosine similarity between an embedding of each group's text
    description and an embedding of the query.

    `embed_fn(list[str]) -> (n, d) array` is any embedding function. This is exactly the
    slot where a production system would plug in an LLM embedding API (e.g. a hosted
    text-embedding model) or a retrieval-augmented LLM relevance judge scoring each
    group's abstract/description against the query. The default here is a local TF-IDF
    vectorizer purely so this class works fully offline for demos -- swap `embed_fn` for
    any embedding call with the same signature and nothing downstream (kernels, blocks,
    spike_slab) needs to change, since they only ever consume the resulting score array.
    """

    def __init__(self, embed_fn=None):
        self.embed_fn = embed_fn

    def score(self, group_descriptions, query):
        texts = list(group_descriptions) + [query]
        if self.embed_fn is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            emb = TfidfVectorizer().fit_transform(texts).toarray()
        else:
            emb = np.asarray(self.embed_fn(texts))

        group_emb, query_emb = emb[:-1], emb[-1]
        norms = np.linalg.norm(group_emb, axis=1) * np.linalg.norm(query_emb) + 1e-12
        return (group_emb @ query_emb) / norms
