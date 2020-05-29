import numpy as np


def topic_score(term_occurrences: int, term_citations: int) -> float:
    return np.log(term_citations + 1) * (term_citations / term_occurrences)
