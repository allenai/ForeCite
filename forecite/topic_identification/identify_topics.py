from typing import Dict, List, Set, Iterable

import os
import json
import multiprocessing
import numpy as np
import argparse
import math
from tqdm import tqdm
from collections import Counter

from forecite.consts import *
from forecite.topic_identification.generate_dataset import get_date_key_from_arxiv_id
from forecite.eval_utils import topic_score


def get_all_candidate_strings_for_n_gram(
    n_gram: str, normalization_dict: Dict[str, List[str]]
) -> Set[str]:
    """Given an n_gram, return the n_gram and all unnormalized versions of it"""
    unnormalized_versions = normalization_dict[n_gram]
    candidates = set([n_gram] + unnormalized_versions)
    return candidates


def merge_noun_phrase_dicts(
    title_noun_phrases: Dict[str, List[str]],
    abstract_noun_phrases: Dict[str, List[str]],
    body_noun_phrases: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Merge title, abstract, and body noun phrase dictionaries together
    """
    output_noun_phrases = {}
    all_noun_phrases = (
        set(title_noun_phrases.keys())
        .union(abstract_noun_phrases.keys())
        .union(body_noun_phrases.keys())
    )
    for noun_phrase in tqdm(all_noun_phrases, desc="Merging noun_phrase dictionaries"):
        output_noun_phrases[noun_phrase] = (
            set(title_noun_phrases.get(noun_phrase, {}))
            .union(abstract_noun_phrases.get(noun_phrase, {}))
            .union(body_noun_phrases.get(noun_phrase, {}))
        )

    return output_noun_phrases


def compute_citation_scores(
    noun_phrase_cluster: Iterable[str],
    combined_noun_phrases: Dict[str, List[str]],
    arxiv_to_s2_mapping: Dict[str, str],
    s2_id_to_citing_ids: Dict[str, List[str]],
    s2_id_to_references: Dict[str, List[str]],
    s2_id_to_canonical: Dict[str, str],
    s2_id_to_date_key: Dict[str, str],
    all_s2_ids_in_corpus_canonical: Set[str],
):
    """
    Compute ForeCite scores for arxiv candidate topics
    """
    # get all the s2 ids associated with any phrase in the noun phrase cluster
    all_s2_ids_for_candidates = []
    for noun_phrase in noun_phrase_cluster:
        all_s2_ids_for_candidates += list(combined_noun_phrases.get(noun_phrase, []))

    # sort the candidates by publication date
    all_s2_ids_for_candidates = sorted(
        all_s2_ids_for_candidates, key=lambda x: s2_id_to_date_key[x]
    )
    papers_checked = set()

    future_ids_with_counts = []

    for i, s2_id in enumerate(all_s2_ids_for_candidates):
        # if the same id somehow ends up in the list twice, we don't need to score it again
        if s2_id in papers_checked:
            continue
        else:
            papers_checked.add(s2_id)

        # restrict to papers with at least 3 citations
        citing_ids = s2_id_to_citing_ids.get(s2_id, [])
        if len(citing_ids) < 3:
            continue

        # only need to search through papers published after the current paper, as citations
        # can only occur in one temporal direction
        future_id_count = 0
        references_match_count = 0
        future_indices = list(range(i + 1, len(all_s2_ids_for_candidates)))

        # Can uncomment for efficiency
        # if len(set(citing_ids).intersection(all_s2_ids_in_corpus_canonical)) < 0.05*min(len(future_indices), 500):
        #     continue

        # NOTE: we sample 500 of the future occurrences for efficiency purposes
        # There is a small amount of randomness introduced here, the results in the paper were computed without setting
        # a random seed
        if len(future_indices) >= 500:
            sampled_indices = np.random.choice(
                list(range(i + 1, len(all_s2_ids_for_candidates))), 500, replace=False
            )
        else:
            sampled_indices = future_indices

        # iterate over the sampled future occurrences of the phrase, and see if they cite the current paper
        for s2_id_future_index in sampled_indices:
            s2_id_future = all_s2_ids_for_candidates[s2_id_future_index]
            references = set(s2_id_to_references.get(s2_id_future, []))
            if len(references) == 0:
                continue

            future_id_count += 1
            # references are aleady canonicalized
            if s2_id in s2_id_to_canonical and s2_id_to_canonical[s2_id] in references:
                references_match_count += 1

        if future_id_count == 0:
            continue

        ratio = references_match_count / future_id_count
        if ratio > 0:
            future_ids_with_counts.append(
                [s2_id, references_match_count, future_id_count]
            )

    return [
        list(noun_phrase_cluster),
        sorted(
            future_ids_with_counts,
            key=lambda x: topic_score(
                x[TERM_OCCURRENCES_INDEX], x[TERM_CITATIONS_INDEX]
            ),
            reverse=True,
        ),
    ]


def compute_cnlc_score(
    noun_phrase_cluster: Iterable[str],
    combined_noun_phrases: Dict[str, List[str]],
    s2_id_to_references: Dict[str, List[str]],
    s2_id_to_canonical: Dict[str, str],
    arxiv_to_s2_mapping: Dict[str, str],
    all_s2_ids_in_corpus_canonical: Set[str],
):
    """
    Compute the CNLC score for a noun phrase cluster.
    Variable names and algorithm come from https://dl.acm.org/citation.cfm?id=1998081
    """
    # get all s2 ids associated with any phrase in the noun phrase cluster
    all_s2_ids_for_candidates = []
    for noun_phrase in noun_phrase_cluster:
        all_s2_ids_for_candidates += list(combined_noun_phrases.get(noun_phrase, []))

    # the set of all s2 ids for this noun phrase cluster, these are the nodes of the term citation subgraph
    Q_p = set(all_s2_ids_for_candidates)

    # canonicalized version of Q_p
    canonical_Q_p = {s2_id_to_canonical.get(id, id) for id in Q_p}

    # size of the full corpus
    n = sum(
        [1 if s2_id != "" else 0 for arxiv_id, s2_id in arxiv_to_s2_mapping.items()]
    )

    # size of the term citation subgraph
    n_p = len(all_s2_ids_for_candidates)

    # counter keeping track of all in corpus references from any paper in the term citation subgraph
    k_i = Counter()
    for i in Q_p:
        references = s2_id_to_references[i]
        in_corpus_references = [
            reference
            for reference in references
            if reference in all_s2_ids_in_corpus_canonical
        ]
        k_i.update(in_corpus_references)

    # count of in corpus references that are also within the term citation subgraph
    a_p_i = {s2_id: count for s2_id, count in k_i.items() if s2_id in canonical_Q_p}

    # compute the final score
    sum_a_p_i = sum(a_p_i.values())
    sum_k_i = sum(k_i.values())
    cnlc = ((1 / n_p) * sum_a_p_i) - ((1 / n) * sum_k_i)
    return [list(noun_phrase_cluster), cnlc]


def compute_loor_score(
    noun_phrase_cluster: Iterable[str],
    combined_noun_phrases: Dict[str, List[str]],
    s2_id_to_references: Dict[str, List[str]],
    arxiv_to_s2_mapping: Dict[str, str],
    s2_id_to_canonical: Dict[str, str],
    s2_id_to_citing_ids: Dict[str, List[str]],
    all_s2_ids_in_corpus_canonical: Set[str],
):
    """
    Compute the LOOR score for a noun phrase cluster.
    Variable names and algorithm come from https://www.cs.cornell.edu/~ykjo/papers/frp751-jo.pdf
    """
    # get all s2 ids associated with any phrase in the noun phrase cluster
    all_s2_ids_for_candidates = []
    for noun_phrase in noun_phrase_cluster:
        all_s2_ids_for_candidates += list(combined_noun_phrases.get(noun_phrase, []))

    # filter to s2 ids in the term citation subgraph that have references
    all_s2_ids_for_candidates_with_references = set()
    for s2_id in all_s2_ids_for_candidates:
        if s2_id_to_references.get(s2_id, []) != []:
            all_s2_ids_for_candidates_with_references.add(s2_id)

    # canonicalized s2 ids in the term citation subgraph
    canonical_s2_ids_for_candidates_with_references = {
        s2_id_to_canonical.get(id, id)
        for id in all_s2_ids_for_candidates_with_references
    }

    # count the number of nodes in the term citation subgraph that have at least 1 reference in the term citation subgraph
    in_graph_links_ids = set()
    for s2_id in all_s2_ids_for_candidates_with_references:
        references = s2_id_to_references.get(s2_id, [])
        canonical_references = set(
            s2_id_to_canonical.get(reference, reference) for reference in references
        )

        citations = s2_id_to_citing_ids.get(s2_id, [])
        canonical_citations = set(
            s2_id_to_canonical.get(citation, citation) for citation in citations
        )

        reference_overlap = canonical_references.intersection(
            canonical_s2_ids_for_candidates_with_references
        )
        citation_overlap = canonical_citations.intersection(
            canonical_s2_ids_for_candidates_with_references
        )
        if len(reference_overlap) > 0 or len(citation_overlap) > 0:
            in_graph_links_ids.add(s2_id)

    # number of nodes in the term citation subgraph
    n_A = len(set(all_s2_ids_for_candidates_with_references))
    # a graph of size 0 or 1 results in undefined math later, score very low
    if n_A == 1 or n_A == 0:
        return [list(noun_phrase_cluster), -1000000000]

    # number of nodes in the term citation subgraph that have at least 1 reference in the term citation subgraph
    n_c_A = len(in_graph_links_ids)
    # magic number from the paper
    p_c = 0.9

    # nodes in the term citation graph, just making a new variable here to align with notation in paper
    s2_ids_in_V = all_s2_ids_for_candidates_with_references

    # accumulate the number of all links for a given node in l_i
    l_i = {}
    for s2_id in s2_ids_in_V:
        citing_ids = s2_id_to_citing_ids.get(s2_id, [])
        canonical_citing_ids = set(
            s2_id_to_canonical.get(citing_id, citing_id) for citing_id in citing_ids
        )
        references = s2_id_to_references.get(s2_id, [])
        canonical_references = set(
            s2_id_to_canonical.get(reference, reference) for reference in references
        )

        # include all citation and references that are in the corpus
        overlap = len(
            (canonical_citing_ids.union(canonical_references)).intersection(
                all_s2_ids_in_corpus_canonical
            )
        )
        l_i[s2_id] = overlap

    # nodes in the term citation graph that have at least one link within the term citation graph
    s2_ids_in_V_c = in_graph_links_ids

    # size of the corpus
    N = sum(
        [1 if s2_id != "" else 0 for arxiv_id, s2_id in arxiv_to_s2_mapping.items()]
    )

    # the first sum in the H0 term
    sum_1 = sum(
        [
            np.log(1 - (1 - ((n_A - 1) / (N - 1))) ** l_i[s2_id])
            for s2_id in s2_ids_in_V_c
        ]
    )

    # the second sum in the H0 term
    sum_2 = sum(
        [
            l_i[s2_id] * np.log(1 - ((n_A - 1) / (N - 1)))
            for s2_id in (s2_ids_in_V - s2_ids_in_V_c)
        ]
    )

    # compute the final score
    h_1 = n_c_A * np.log(p_c) + (n_A - n_c_A) * np.log(1 - p_c)
    h_0 = sum_1 + sum_2
    score = h_1 - h_0
    return [list(noun_phrase_cluster), score]


def identify_topics_arxiv_no_refs(method: str, candidates: str):
    """
    Function to compute topics for arxiv cs with references clipped
    """
    print("Loading title noun_phrases...")
    with open(NO_REFS_ARXIV_CS_TITLE_NPS_PATH) as _json_file:
        title_noun_phrases = json.load(_json_file)

    print("Loading abstract noun_phrases...")
    with open(NO_REFS_ARXIV_CS_ABSTRACT_NPS_PATH) as _json_file:
        abstract_noun_phrases = json.load(_json_file)

    print("Loading body noun_phrases...")
    with open(NO_REFS_ARXIV_CS_BODY_NPS_PATH) as _json_file:
        body_noun_phrases = json.load(_json_file)

    print("Loading normalization...")
    with open(NO_REFS_ARXIV_CS_NORMALIZATION_PATH) as _json_file:
        phrase_normalization = json.load(_json_file)

    print("Loading citng ids...")
    with open(NO_REFS_ARXIV_CS_CITING_IDS_PATH) as _json_file:
        s2_id_to_citing_ids = json.load(_json_file)

    print("Loading references...")
    with open(NO_REFS_ARXIV_CS_REFERENCES_PATH) as _json_file:
        s2_id_to_references = json.load(_json_file)

    print("Loading canonicalization...")
    with open(NO_REFS_ARXIV_CS_CANONICALIZATION_PATH) as _json_file:
        s2_id_to_canonical = json.load(_json_file)

    print("Loading arxiv to s2 mapping...")
    with open(NO_REFS_ARXIV_CS_TO_S2_MAPPING_PATH) as _json_file:
        arxiv_to_s2_mapping = json.load(_json_file)

    s2_id_to_date_key = {
        value: get_date_key_from_arxiv_id(key)
        for key, value in arxiv_to_s2_mapping.items()
    }

    combined_noun_phrases = merge_noun_phrase_dicts(
        title_noun_phrases, abstract_noun_phrases, body_noun_phrases
    )

    if candidates == "title":
        all_title_noun_phrases = {
            noun_phrase
            for noun_phrase in tqdm(
                title_noun_phrases.keys(), desc="Computing all candidate phrases"
            )
        }
        candidate_expanded_noun_phrases = [
            get_all_candidate_strings_for_n_gram(noun_phrase, phrase_normalization)
            for noun_phrase in tqdm(
                all_title_noun_phrases, desc="Expanding candidate phrases"
            )
        ]
    elif candidates == "abstract":
        all_abstract_noun_phrases = {
            noun_phrase
            for noun_phrase in tqdm(
                abstract_noun_phrases.keys(), desc="Computing all candidate phrases"
            )
        }
        candidate_expanded_noun_phrases = [
            get_all_candidate_strings_for_n_gram(noun_phrase, phrase_normalization)
            for noun_phrase in tqdm(
                all_abstract_noun_phrases, desc="Expanding candidate phrases"
            )
        ]
    else:
        raise Exception("Invalid candidate set")

    all_ids_in_corpus_canonical = {
        s2_id_to_canonical.get(id, "") for id in set(arxiv_to_s2_mapping.values())
    }
    score_results = []
    for noun_phrase_cluster in tqdm(
        candidate_expanded_noun_phrases, desc="Computing citation scores"
    ):
        if method == "ours":
            citation_scores = compute_citation_scores(
                noun_phrase_cluster,
                combined_noun_phrases,
                arxiv_to_s2_mapping,
                s2_id_to_citing_ids,
                s2_id_to_references,
                s2_id_to_canonical,
                s2_id_to_date_key,
                all_ids_in_corpus_canonical,
            )
            if citation_scores[1] != []:
                score_results.append(citation_scores)
        elif method == "cnlc":
            score = compute_cnlc_score(
                noun_phrase_cluster,
                combined_noun_phrases,
                s2_id_to_references,
                s2_id_to_canonical,
                arxiv_to_s2_mapping,
                all_ids_in_corpus_canonical,
            )
            score_results.append(score)
        elif method == "loor":
            score = compute_loor_score(
                noun_phrase_cluster,
                combined_noun_phrases,
                s2_id_to_references,
                arxiv_to_s2_mapping,
                s2_id_to_canonical,
                s2_id_to_citing_ids,
                all_ids_in_corpus_canonical,
            )
            score_results.append(score)

    if method == "ours":
        score_results = [
            (
                result[0],
                topic_score(
                    result[1][0][TERM_OCCURRENCES_INDEX],
                    result[1][0][TERM_CITATIONS_INDEX],
                ),
                result[1],
            )
            for result in score_results
        ]
        score_results = sorted(score_results, key=lambda x: x[1], reverse=True,)
    elif method == "cnlc":
        score_results = sorted(score_results, key=lambda x: x[1], reverse=True)
    elif method == "loor":
        score_results = sorted(score_results, key=lambda x: x[1], reverse=True)

    if candidates == "title":
        if method == "ours":
            output_file_path = NO_REFS_ARXIV_CS_TITLE_CANDIDATES_SCORES_PATH
        elif method == "cnlc":
            output_file_path = NO_REFS_ARXIV_CS_TITLE_CANDIDATES_CNLC_PATH
        elif method == "loor":
            output_file_path = NO_REFS_ARXIV_CS_TITLE_CANDIDATES_LOOR_PATH
    elif candidates == "abstract":
        if method == "ours":
            output_file_path = NO_REFS_ARXIV_CS_ABSTRACT_CANDIDATES_SCORES_PATH
        elif method == "cnlc":
            output_file_path = NO_REFS_ARXIV_CS_ABSTRACT_CANDIDATES_CNLC_PATH
        elif method == "loor":
            output_file_path = NO_REFS_ARXIV_CS_ABSTRACT_CANDIDATES_LOOR_PATH
    print("Dumping citation score output to {}".format(output_file_path))
    with open(output_file_path, "w") as _json_file:
        json.dump(score_results, _json_file)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Which dataset to run on")
    parser.add_argument("--method", help="Which method to use to score topics")
    parser.add_argument("--candidates", help="What candidate set to use")
    args = parser.parse_args()

    if args.dataset == "arxiv_no_refs":
        identify_topics_arxiv_no_refs(args.method, args.candidates)
    else:
        raise Exception(f"Dataset {args.dataset} not supported")
