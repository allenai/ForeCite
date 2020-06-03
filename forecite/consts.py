import os

# Useful directories
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, os.pardir, "data")

TOPICID_DATA_ROOT = os.path.join(DATA_ROOT, "topic_identification")
NO_REFS_ARXIV_CS_DATA_ROOT = os.path.join(DATA_ROOT, "arxiv_no_refs")

# Processed data file paths
NO_REFS_ARXIV_CS_IDS_PATH = os.path.join(NO_REFS_ARXIV_CS_DATA_ROOT, "arxiv_ids.json")
NO_REFS_ARXIV_CS_TO_S2_MAPPING_PATH = os.path.join(
    NO_REFS_ARXIV_CS_DATA_ROOT, "arxiv_to_s2_mapping.json"
)
NO_REFS_ARXIV_CS_TITLE_NPS_PATH = os.path.join(
    NO_REFS_ARXIV_CS_DATA_ROOT, "title_nps.json"
)
NO_REFS_ARXIV_CS_ABSTRACT_NPS_PATH = os.path.join(
    NO_REFS_ARXIV_CS_DATA_ROOT, "abstract_nps.json"
)
NO_REFS_ARXIV_CS_BODY_NPS_PATH = os.path.join(
    NO_REFS_ARXIV_CS_DATA_ROOT, "body_nps.json"
)
NO_REFS_ARXIV_CS_NORMALIZATION_PATH = os.path.join(
    NO_REFS_ARXIV_CS_DATA_ROOT, "normalization.json"
)
NO_REFS_ARXIV_CS_REFERENCES_PATH = os.path.join(
    NO_REFS_ARXIV_CS_DATA_ROOT, "s2_id_to_references.json"
)
NO_REFS_ARXIV_CS_CITING_IDS_PATH = os.path.join(
    NO_REFS_ARXIV_CS_DATA_ROOT, "s2_id_to_citing_ids.json"
)
NO_REFS_ARXIV_CS_CANONICALIZATION_PATH = os.path.join(
    NO_REFS_ARXIV_CS_DATA_ROOT, "s2_id_to_canonical.json"
)
NO_REFS_ARXIV_CS_TITLE_CANDIDATES_SCORES_PATH = os.path.join(
    NO_REFS_ARXIV_CS_DATA_ROOT, "title_citation_scores.json"
)
NO_REFS_ARXIV_CS_TITLE_CANDIDATES_CNLC_PATH = os.path.join(
    NO_REFS_ARXIV_CS_DATA_ROOT, "title_cnlc_scores.json"
)
NO_REFS_ARXIV_CS_TITLE_CANDIDATES_LOOR_PATH = os.path.join(
    NO_REFS_ARXIV_CS_DATA_ROOT, "title_loor_scores.json"
)
NO_REFS_ARXIV_CS_ABSTRACT_CANDIDATES_SCORES_PATH = os.path.join(
    NO_REFS_ARXIV_CS_DATA_ROOT, "abstract_citation_scores.json"
)
NO_REFS_ARXIV_CS_ABSTRACT_CANDIDATES_CNLC_PATH = os.path.join(
    NO_REFS_ARXIV_CS_DATA_ROOT, "abstract_cnlc_scores.json"
)
NO_REFS_ARXIV_CS_ABSTRACT_CANDIDATES_LOOR_PATH = os.path.join(
    NO_REFS_ARXIV_CS_DATA_ROOT, "abstract_loor_scores.json"
)

# Evaluation related
RANDOM_SAMPLE_FOR_CALIBRATION_OUTPUT_PATH = os.path.join(
    TOPICID_DATA_ROOT, "random_noun_phrase_calibration_sample.csv"
)
RANDOM_SAMPLE_FOR_EVALUATION_OUTPUT_PATH = os.path.join(
    TOPICID_DATA_ROOT, "random_noun_phrase_sample.csv"
)

# Related to the output format of the citation scores
TERM_OCCURRENCES_INDEX = 2
TERM_CITATIONS_INDEX = 1
