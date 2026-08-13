from enum import Enum


class SearchType(str, Enum):
    KEYWORD = "keyword"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"


class ScoreNormalization(str, Enum):
    RAW = "raw"          # native scales: BM25 unbounded, cosine [-1,1], RRF small fractions
    NORMALIZED = "normalized"  # fixed [0,1] transform per method — scores are absolute, not relative
