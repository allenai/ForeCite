import unittest
import numpy as np

from new_topics.topic_identification.identify_topics import (
    compute_citation_scores,
    compute_cnlc_score,
    compute_loor_score,
)


class TestScoringFunctions(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # all numeric ids are dummies
        self.noun_phrase_cluster = ["bert"]
        self.combined_noun_phrases = {
            "bert": [
                "9b295d11eb60673311c7e1f3d85b357ce82126d4",  # bert
                "c24031d5cde53397d9d922feb7a5e52681281b86",  # mt-dnn, cites bert
                "6461d598670e08d76df83cfb30598f0f8488a5cc",  # xlnet, cites bert
                "73afd40f307ecfc1f35fba1d3da631f4f928518b",  # superglue, which cites bert and mt-dnn
                "ccd5efeb312121f182f7dcaa09e8a8088acc93cc",  # random paper containing "bert" without citing the bert paper, and published before the bert paper
            ]
        }
        self.s2_id_to_references = {
            "9b295d11eb60673311c7e1f3d85b357ce82126d4": [0, 1, 2, 3, 4],
            "c24031d5cde53397d9d922feb7a5e52681281b86": [
                0,
                1,
                2,
                3,
                "df2b0e26d0599ce3e70df8a9da02e51594e0e992",
            ],
            "6461d598670e08d76df83cfb30598f0f8488a5cc": [
                0,
                1,
                2,
                3,
                "df2b0e26d0599ce3e70df8a9da02e51594e0e992",
            ],
            "ccd5efeb312121f182f7dcaa09e8a8088acc93cc": [1, -2, -3, -4, -5],
            "73afd40f307ecfc1f35fba1d3da631f4f928518b": [
                0,
                1,
                2,
                "df2b0e26d0599ce3e70df8a9da02e51594e0e992",
                "658721bc13b0fa97366d38c05a96bf0a9f4bb0ac",
            ],
        }
        self.s2_id_to_citing_ids = {
            "9b295d11eb60673311c7e1f3d85b357ce82126d4": [
                "658721bc13b0fa97366d38c05a96bf0a9f4bb0ac",
                "e0c6abdbdecf04ffac65c440da77fb9d66bb474c",
                "d9f6ada77448664b71128bb19df15765336974a6",
            ],
            "c24031d5cde53397d9d922feb7a5e52681281b86": [
                "73afd40f307ecfc1f35fba1d3da631f4f928518b",
                0,
                1,
            ],
        }
        self.s2_id_to_canonical = {
            "9b295d11eb60673311c7e1f3d85b357ce82126d4": "df2b0e26d0599ce3e70df8a9da02e51594e0e992",
            "c24031d5cde53397d9d922feb7a5e52681281b86": "658721bc13b0fa97366d38c05a96bf0a9f4bb0ac",
            "6461d598670e08d76df83cfb30598f0f8488a5cc": "e0c6abdbdecf04ffac65c440da77fb9d66bb474c",
            "73afd40f307ecfc1f35fba1d3da631f4f928518b": "d9f6ada77448664b71128bb19df15765336974a6",
            "ccd5efeb312121f182f7dcaa09e8a8088acc93cc": "bac0e4820b2116a18a71110e0d4ce675b5e68932",
            "df2b0e26d0599ce3e70df8a9da02e51594e0e992": "df2b0e26d0599ce3e70df8a9da02e51594e0e992",
            "658721bc13b0fa97366d38c05a96bf0a9f4bb0ac": "658721bc13b0fa97366d38c05a96bf0a9f4bb0ac",
            "e0c6abdbdecf04ffac65c440da77fb9d66bb474c": "e0c6abdbdecf04ffac65c440da77fb9d66bb474c",
            "d9f6ada77448664b71128bb19df15765336974a6": "d9f6ada77448664b71128bb19df15765336974a6",
            "bac0e4820b2116a18a71110e0d4ce675b5e68932": "bac0e4820b2116a18a71110e0d4ce675b5e68932",
            1: 1,
            2: 2,
            3: 3,
            4: 4,
        }
        self.arxiv_to_s2_mapping = {
            "1810.04805": "9b295d11eb60673311c7e1f3d85b357ce82126d4",
            "1901.11504": "c24031d5cde53397d9d922feb7a5e52681281b86",
            "1906.08237": "6461d598670e08d76df83cfb30598f0f8488a5cc",
            "1905.00537": "73afd40f307ecfc1f35fba1d3da631f4f928518b",
            "cs/9904007": "ccd5efeb312121f182f7dcaa09e8a8088acc93cc",
        }
        self.arxiv_to_s2_mapping.update({i: i for i in range(1, 100000)})
        self.all_s2_ids_in_corpus_canonical = {i for i in range(1, 100000)}.union(
            set(self.s2_id_to_canonical.values())
        )
        self.s2_id_to_date_key = {
            "9b295d11eb60673311c7e1f3d85b357ce82126d4": "2018_10",
            "c24031d5cde53397d9d922feb7a5e52681281b86": "2019_01",
            "6461d598670e08d76df83cfb30598f0f8488a5cc": "2019_06",
            "73afd40f307ecfc1f35fba1d3da631f4f928518b": "2019_05",
            "ccd5efeb312121f182f7dcaa09e8a8088acc93cc": "1999_04",
        }

    def test_cnlc(self):
        computed_score = compute_cnlc_score(
            self.noun_phrase_cluster,
            self.combined_noun_phrases,
            self.s2_id_to_references,
            self.s2_id_to_canonical,
            self.arxiv_to_s2_mapping,
            self.all_s2_ids_in_corpus_canonical,
        )

        n_p = len(self.combined_noun_phrases["bert"])
        n = len(self.all_s2_ids_in_corpus_canonical)

        # sum of the number of references that each article has in the corpus
        sum_k_i = 4 + 4 + 4 + 1 + 4

        # 4 reference links within the subgraph
        sum_a_p_i = 4
        expected_score = (1 / n_p) * sum_a_p_i - (1 / n) * sum_k_i

        assert expected_score == computed_score[1]

    def test_loor(self):
        computed_score = compute_loor_score(
            self.noun_phrase_cluster,
            self.combined_noun_phrases,
            self.s2_id_to_references,
            self.arxiv_to_s2_mapping,
            self.s2_id_to_canonical,
            self.s2_id_to_citing_ids,
            self.all_s2_ids_in_corpus_canonical,
        )
        p_c = 0.9
        n_c_A = 4
        n_A = 5
        N = len(self.arxiv_to_s2_mapping)

        sum_1 = (
            np.log(1 - (1 - ((n_A - 1) / (N - 1))) ** 7)
            + np.log(1 - (1 - ((n_A - 1) / (N - 1))) ** 5)
            + np.log(1 - (1 - ((n_A - 1) / (N - 1))) ** 4)
            + np.log(1 - (1 - ((n_A - 1) / (N - 1))) ** 4)
        )
        sum_2 = 1 * np.log(1 - ((n_A - 1) / (N - 1)))

        h0 = sum_1 + sum_2
        h1 = n_c_A * np.log(p_c) + (n_A - n_c_A) * np.log(1 - p_c)

        expected_score = h1 - h0
        assert expected_score == computed_score[1]

    def test_citation(self):
        computed_scores = compute_citation_scores(
            self.noun_phrase_cluster,
            self.combined_noun_phrases,
            self.arxiv_to_s2_mapping,
            self.s2_id_to_citing_ids,
            self.s2_id_to_references,
            self.s2_id_to_canonical,
            self.s2_id_to_date_key,
        )

        expected_scores = [
            ["9b295d11eb60673311c7e1f3d85b357ce82126d4", 3, 3],
            ["c24031d5cde53397d9d922feb7a5e52681281b86", 1, 2],
        ]

        assert expected_scores == computed_scores[1]
