from typing import List, Dict

import json
import spacy
import pickle
import multiprocessing
import argparse

from scispacy.abbreviation import AbbreviationDetector
from tqdm import tqdm
from collections import defaultdict

from forecite.consts import *
from forecite import s2_utils

def compute_noun_phrases_worker(input_text: str) -> List:
    """Returns the noun phrases in a string"""
    doc = s2_utils.nlp_md(input_text[:1000000])
    return [span.text.lower() for span in doc.noun_chunks]


def get_date_key_from_arxiv_id(arxiv_id: str):
    """
    Return a date key of the form <year>_<month> from an arxiv id
    """
    if "/" in arxiv_id:
        arxiv_id = arxiv_id.split("/")[1]

    short_year = arxiv_id[0:2]
    month = arxiv_id[2:4]

    if (
        short_year.startswith("0")
        or short_year.startswith("1")
        or short_year.startswith("2")
    ):
        year = "20" + short_year
    else:
        year = "19" + short_year

    return year + "_" + month


def generate_dataset_no_refs_arxiv_cs(num_processes: int = 1):
    """
    Function to generate the full topic extraction dataset for arxiv cs with references clipped
    """
    if not os.path.exists(NO_REFS_ARXIV_CS_DATA_ROOT):
        print("Creating directory at {}".format(NO_REFS_ARXIV_CS_DATA_ROOT))
        os.mkdir(NO_REFS_ARXIV_CS_DATA_ROOT)

    if not os.path.exists(NO_REFS_ARXIV_CS_IDS_PATH):
        print("Querying for all arxiv cs ids...")
        arxiv_ids = s2_utils.get_all_arxiv_cs_ids()

        print("Writing arxiv cs ids to {}".format(NO_REFS_ARXIV_CS_IDS_PATH))
        with open(NO_REFS_ARXIV_CS_IDS_PATH, "w") as _arxiv_ids_json_file:
            json.dump(list(arxiv_ids), _arxiv_ids_json_file)
    else:
        print("Loading arxiv cs ids from {}".format(NO_REFS_ARXIV_CS_IDS_PATH))
        with open(NO_REFS_ARXIV_CS_IDS_PATH) as _arxiv_ids_json_file:
            arxiv_ids = json.load(_arxiv_ids_json_file)

    if not os.path.exists(NO_REFS_ARXIV_CS_TO_S2_MAPPING_PATH):
        print("Generating arxiv to s2 mapping...")
        arxiv_to_s2_mapping = s2_utils.get_arxiv_to_s2_id_mapping(arxiv_ids, [])

        print(
            "Writing arxiv to s2 mapping to {}".format(
                NO_REFS_ARXIV_CS_TO_S2_MAPPING_PATH
            )
        )
        with open(
            NO_REFS_ARXIV_CS_TO_S2_MAPPING_PATH, "w"
        ) as _arxiv_to_s2_mapping_json_file:
            json.dump(arxiv_to_s2_mapping, _arxiv_to_s2_mapping_json_file)
    else:
        print(
            "Loading arxiv to s2 mapping from {}".format(
                NO_REFS_ARXIV_CS_TO_S2_MAPPING_PATH
            )
        )
        with open(NO_REFS_ARXIV_CS_TO_S2_MAPPING_PATH) as _json_file:
            arxiv_to_s2_mapping = json.load(_json_file)

    s2_ids = [
        arxiv_to_s2_mapping[arxiv_id]
        for arxiv_id in arxiv_ids
        if arxiv_to_s2_mapping[arxiv_id] != ""
    ]

    if not os.path.exists(NO_REFS_ARXIV_CS_TITLE_NPS_PATH):
        print("Getting data")
        (
            title_inverted_index,
            abstract_inverted_index,
            body_inverted_index,
            normalization_dict,
            s2_id_to_citing_ids,
            s2_id_to_references,
            s2_id_to_canonical,
        ) = s2_utils.full_data_collection_parallel(s2_ids, num_processes)

        print("Dumping title nps to {}".format(NO_REFS_ARXIV_CS_TITLE_NPS_PATH))
        with open(NO_REFS_ARXIV_CS_TITLE_NPS_PATH, "w") as _json_file:
            json.dump(title_inverted_index, _json_file)

        print("Dumping abstract nps to {}".format(NO_REFS_ARXIV_CS_ABSTRACT_NPS_PATH))
        with open(NO_REFS_ARXIV_CS_ABSTRACT_NPS_PATH, "w") as _json_file:
            json.dump(abstract_inverted_index, _json_file)

        print("Dumping body nps to {}".format(NO_REFS_ARXIV_CS_BODY_NPS_PATH))
        with open(NO_REFS_ARXIV_CS_BODY_NPS_PATH, "w") as _json_file:
            json.dump(body_inverted_index, _json_file)

        print("Dumping normalization to {}".format(NO_REFS_ARXIV_CS_NORMALIZATION_PATH))
        with open(NO_REFS_ARXIV_CS_NORMALIZATION_PATH, "w") as _json_file:
            json.dump(normalization_dict, _json_file)

        print("Dumping citing ids to {}".format(NO_REFS_ARXIV_CS_CITING_IDS_PATH))
        with open(NO_REFS_ARXIV_CS_CITING_IDS_PATH, "w") as _json_file:
            json.dump(s2_id_to_citing_ids, _json_file)

        print("Dumping references to {}".format(NO_REFS_ARXIV_CS_REFERENCES_PATH))
        with open(NO_REFS_ARXIV_CS_REFERENCES_PATH, "w") as _json_file:
            json.dump(s2_id_to_references, _json_file)

        print(
            "Dumping canonicalization to {}".format(
                NO_REFS_ARXIV_CS_CANONICALIZATION_PATH
            )
        )
        with open(NO_REFS_ARXIV_CS_CANONICALIZATION_PATH, "w") as _json_file:
            json.dump(s2_id_to_canonical, _json_file)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Which dataset to run on")
    parser.add_argument("--num_processes", type=int, help="How many processes to use")
    args = parser.parse_args()

    if args.dataset == "no_refs_arxiv":
        generate_dataset_no_refs_arxiv_cs(num_processes=args.num_processes)
    else:
        raise Exception(f"Dataset {args.dataset} not supported")
