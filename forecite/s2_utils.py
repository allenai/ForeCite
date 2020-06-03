from typing import Tuple, Optional, Dict, Set, List, Iterable, Any, Callable

import multiprocessing
import os
import psycopg2
import time
import json
import elasticsearch
import spacy
import functools
import re

from elasticsearch.exceptions import ConnectionTimeout, TransportError
from tqdm import tqdm
from multiprocessing import Manager
from spacy.tokens.token import Token
from spacy.language import Language
from forecite.consts import *

from nltk.corpus import stopwords

NLTK_STOPWORDS = set(stopwords.words("english")) | {"using", "with", "for"}

nlp_md = spacy.load('en_core_web_md')

def clip_references_from_text(body_text: str) -> str:
    """
    Function to heuristically remove the references section from the body text (if it is present)
    The idea is: find the last occurrences of a synonym of "references" and then make sure it is followed by
                 a new line or something that looks like a year
    """
    references_pattern = r"(REFERENCES)|(References)|(Bibliography)|(BIBLIOGRAPHY)|(WORKS CITED)|(Works Cited)|(Works cited)|(Verwysings)|(verwysings)"
    references_starting_pattern = r"(\n)|(\d\d\d\d)"
    references_found = [i.start() for i in re.finditer(references_pattern, body_text)]
    if references_found != []:
        last_reference_index = references_found[-1]
        if re.search(references_starting_pattern, body_text[last_reference_index:last_reference_index+300]) is not None:
            return body_text[:last_reference_index]
        else:
            return body_text
    else:
        return body_text


def _get_es_client(use_prod=False):
    """
    Returns an es client
    """
    if use_prod:
        client = elasticsearch.Elasticsearch(ELASTIC_SEARCH_URL_PORT_PROD)
    else:
        client = elasticsearch.Elasticsearch(ELASTIC_SEARCH_URL_PORT_DEV)
    return client


# ES = _get_es_client(use_prod=False)


def clean_arxiv_id(possible_arxiv_id: str):
    """
    Remove the version info from an arxiv id
    """
    possible_arxiv_id = possible_arxiv_id.split("v")[0]
    return possible_arxiv_id


def get_canonical_paperid_modified(paper_id: str, use_prod: bool = False) -> str:
    """
    Gets the canonical s2 id for a given paper id. This function was modified
    from what is in pys2, because the version in pys2 crashed if it was unable to canonicalize
    an id.
    """
    es_res = ES.search(
        index="paper", body={"query": {"term": {"clusterInfo.duplicateIds": paper_id}}}
    )
    if len(es_res["hits"]["hits"]) > 0:
        return es_res["hits"]["hits"][0]["_id"]
    else:
        print("WARNING: {} id cannot be canonicalized".format(paper_id))
        return ""


def get_paper_blob_modified(
    paper_id: str, use_canonical: bool = True, use_prod: bool = False
) -> Optional[Dict]:
    """
    Gets the paper blob for an s2 id. This function was modified from what is in pys2,
    because the version in pys2 crashed if it was unable to canonicalize an id.
    """
    es_res = ES.search(index="paper", body={"query": {"term": {"id": paper_id}}})
    if len(es_res["hits"]["hits"]) > 0:
        return es_res["hits"]["hits"][0]
    elif use_canonical:
        paper_id = get_canonical_paperid_modified(paper_id)
        if paper_id == "":
            return None
        else:
            return get_paper_blob_modified(paper_id, use_canonical=False)
    else:
        return None


def get_references_for_s2_id(s2_id: str) -> List[str]:
    """
    Gets the references for an s2 id, will keep retrying until the query succeeds.
    It is possible for there to be zero referring ids
    """
    while True:
        try:
            paper_blob = get_paper_blob_modified(s2_id)

            if paper_blob is None:
                return (s2_id, [])

            canonical_id = paper_blob["_id"]

            es_references = ES.search(
                body={"query": {"term": {"citingPaper.id": canonical_id}}}, size=100
            )

            referring_ids = []

            hits = es_references["hits"]["hits"]
            for reference in hits:
                source = reference["_source"]
                if (
                    "id" not in source["citedPaper"]
                    or "title" not in source["citedPaper"]
                ):
                    continue

                cited_paper = source["citedPaper"]["id"]
                referring_ids.append(cited_paper)

            return (s2_id, referring_ids)
        except ConnectionTimeout:
            print("Timeout error: Sleeping on s2 id {} for one second".format(s2_id))
            time.sleep(1)
        except TransportError:
            print("Transport error: Sleeping on s2 id {} for 5 seconds".format(s2_id))
            time.sleep(5)


def get_all_arxiv_cs_ids() -> Set[str]:
    """
    Query to get all arxiv ids in the corpus
    """
    arxiv_ids = set()
    with psycopg2.connect(**DB_S2_CRAWLER) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
            SELECT  metadata_id as arxiv_id
            FROM metadata
            WHERE content_hash IS NOT NULL
                AND (categories like 'cs.%' OR categories like 'stat.ML%')
            """
            )
            i = 0
            while True:
                i += 1
                rows = cursor.fetchmany(20000)
                if not rows:
                    break
                for (arxiv_id,) in rows:
                    arxiv_ids.add(arxiv_id)

    return arxiv_ids


def get_arxiv_to_s2_id_mapping(arxivs: List[str], eprints: List[str]) -> Dict[str, str]:
    """
    Gets the results of querying for mapping arxiv ids to s2 ids
    The arxivs and eprints arguments are just combined
    """
    arxivs = [clean_arxiv_id(arxiv) for arxiv in arxivs]
    eprints = [clean_arxiv_id(eprint) for eprint in eprints]
    arxiv_to_s2_mapping = {}
    with psycopg2.connect(**DB_S2_DEV) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                           SELECT pi.sha, sp.source_id
                           FROM papers p
                           INNER JOIN sourced_paper sp on sp.paper_id = p.id
                           INNER JOIN legacy_paper_ids pi on pi.paper_id = p.id
                           WHERE sp.source = 'ArXiv'
                           AND sp.source_id IN {arxiv_ids}
                           ;
                           """.format(
                    arxiv_ids="('" + "','".join(arxivs + eprints) + "')"
                )
            )
            i = 0
            while True:
                rows = cursor.fetchmany(5000)
                if not rows:
                    break
                for (s2_id, arxiv_id) in rows:
                    arxiv_to_s2_mapping[arxiv_id] = s2_id
                i += 1

    for arxiv_id in arxivs + eprints:
        if arxiv_id not in arxiv_to_s2_mapping:
            arxiv_to_s2_mapping[arxiv_id] = ""

    return arxiv_to_s2_mapping

def normalize_np(np: List[Token], nlp_md: Language) -> str:
    """
    Function to normalize a noun phrase
    """
    normalized_np = " ".join(
        [
            token.lemma_
            for token in nlp_md.tokenizer(np)
            if token.text not in NLTK_STOPWORDS
        ]
    )
    return normalized_np

def compute_normalized_nps(nps: List[str], normalization_dict: Dict[str, Set]) -> Set[str]:
    """
    Aggregates a list of nps into a set of normalized nps.
    Note that the normaliztion_dict argument is modified in this function, to keep track of a global normalization dict
    """
    normalized_nps = set()
    for np in nps:
        normalized_np = normalize_np(np, nlp_md)
        if normalized_np in normalization_dict:
            normalization_dict[normalized_np].add(np)
        else:
            normalization_dict[normalized_np] = {np}

        normalized_nps.add(normalized_np)
    return normalized_nps


def get_nps_from_text(title: str, abstract: str, body: str, body_percent: float = 1):
    """
    Function to extract noun phrases and phrase normalizations from the text of a given paper
    """
    title_nps = [span.text.lower() for span in nlp_md(title[:1000000]).noun_chunks]
    abstract_nps = [
        span.text.lower() for span in nlp_md(abstract[:1000000]).noun_chunks
    ]
    body_cutoff = min(int(len(body) * body_percent), 1000000)
    body_nps = [span.text.lower() for span in nlp_md(body[:body_cutoff]).noun_chunks]

    normalization_dict = {}
    normalized_title_nps = compute_normalized_nps(title_nps, normalization_dict)
    normalized_abstract_nps = compute_normalized_nps(abstract_nps, normalization_dict)
    normalized_body_nps = compute_normalized_nps(body_nps, normalization_dict)

    return (
        normalization_dict,
        normalized_title_nps,
        normalized_abstract_nps,
        normalized_body_nps,
    )


def full_data_collection_worker(s2_id: str):
    """
    Worker function to query for all of the data needed for topic extraction from an s2 id
    """
    while True:
        try:
            paper_blob = get_paper_blob_modified(s2_id)
            if paper_blob != None:
                source = paper_blob["_source"]
                title = source["title"] if "title" in source else ""
                abstract = source["paperAbstract"] if "paperAbstract" in source else ""
                body = source["bodyText"] if "bodyText" in source else ""
                body = clip_references_from_text(body)
                canonical = paper_blob["_id"]
                citing_ids = source["citedBy"] if "citedBy" in source else []
            else:
                canonical = ""
                title = ""
                abstract = ""
                body = ""
                citing_ids = []
            break
        except ConnectionTimeout:
            time.sleep(1)

    normalization_dict, title_nps, abstract_nps, body_nps = get_nps_from_text(
        title, abstract, body
    )
    _, references = get_references_for_s2_id(s2_id)

    return (
        s2_id,
        canonical,
        normalization_dict,
        title_nps,
        abstract_nps,
        body_nps,
        citing_ids,
        references,
    )


def full_data_collection_parallel(s2_ids: List[str], num_processes: int = 1):
    """
    Gets all data needed for topic extraction from a list of s2 ids in parallel
    """
    full_normalization_dict = {}
    title_inverted_index = {}
    abstract_inverted_index = {}
    body_inverted_index = {}
    s2_id_to_canonical = {}
    s2_id_to_references = {}
    s2_id_to_citing_ids = {}

    results = []
    with multiprocessing.Pool(processes=num_processes) as p:
        _max = len(s2_ids)
        with tqdm(total=_max) as pbar:
            for (
                i,
                (
                    s2_id,
                    canonical,
                    normalization_dict,
                    title_nps,
                    abstract_nps,
                    body_nps,
                    citing_ids,
                    references,
                ),
            ) in enumerate(p.imap(full_data_collection_worker, s2_ids)):

                if canonical == "":
                    continue

                for normalized_term, unnormalized_terms in normalization_dict.items():
                    if normalized_term == "":
                        continue
                    if normalized_term in full_normalization_dict:
                        full_normalization_dict[normalized_term].update(
                            unnormalized_terms
                        )
                    else:
                        full_normalization_dict[normalized_term] = unnormalized_terms

                for title_np in title_nps:
                    if title_np == "":
                        continue
                    if title_np in title_inverted_index:
                        title_inverted_index[title_np].add(s2_id)
                    else:
                        title_inverted_index[title_np] = {s2_id}

                for abstract_np in abstract_nps:
                    if abstract_np == "":
                        continue
                    if abstract_np in abstract_inverted_index:
                        abstract_inverted_index[abstract_np].add(s2_id)
                    else:
                        abstract_inverted_index[abstract_np] = {s2_id}

                for body_np in body_nps:
                    if body_np == "":
                        continue
                    if body_np in body_inverted_index:
                        body_inverted_index[body_np].add(s2_id)
                    else:
                        body_inverted_index[body_np] = {s2_id}

                s2_id_to_references[s2_id] = list(references)
                s2_id_to_citing_ids[s2_id] = list(citing_ids)
                s2_id_to_canonical[s2_id] = canonical
                pbar.update()

    # make them json serialiable
    title_inverted_index = {
        key: list(value) for key, value in title_inverted_index.items() if key != ""
    }
    abstract_inverted_index = {
        key: list(value) for key, value in abstract_inverted_index.items() if key != ""
    }
    body_inverted_index = {
        key: list(value) for key, value in body_inverted_index.items() if key != ""
    }
    full_normalization_dict = {
        key: list(value) for key, value in full_normalization_dict.items() if key != ""
    }

    return (
        title_inverted_index,
        abstract_inverted_index,
        body_inverted_index,
        full_normalization_dict,
        s2_id_to_citing_ids,
        s2_id_to_references,
        s2_id_to_canonical,
    )

