# ForeCite
This repository contains data and code for the paper "High-Precision Extraction of Emerging Concepts from Scientific Literature"

## Downloading the data
You will first want to download [this](https://s3.console.aws.amazon.com/s3/buckets/ai2-s2-forecite/?region=us-west-2) S3 bucket to `ForeCite/data/arxiv_no_refs`.

This directory will now contain the text and citations data needed to produce the output, the output itself, and the annotations used for evaluation. If you just want to look at the output, you can stop here. Please note that the full ranked concept lists differ slightly from the lists used for evaluation, because, for evaluation, we filtered the list to phrases that occur in titles in 2018 or earlier, while the full list contains phrases that occur in titles in 2019 as well.

## Data generation
The script to generate the underlying json files with all the text and citations data is [here](https://github.com/allenai/ForeCite/tree/master/forecite/topic_identification/generate_dataset.py). Please note, this script is present only for purposes of reproducibility and clarity. It will not actually run, as it contacts Semantic Scholar internal services.

## Concept scoring
To rerun concept scoring you need to:

1. [Optional] Create and activate new conda environment 
   ```
   conda create -n forecite python=3.7
   conda activate forecite
   ```
   
2. Setup package and install requirements from the root of this repository.
    ```
    pip install -r requirements.txt
    pip install -e .
    ```
    
3. Download spacy model.
   ```
   python -m spacy download en_core_web_md
   ```

4. Run concept scoring command. 
   ```
   python topic_identification/identify_topics.py --dataset arxiv_no_refs --method forecite --candidates title
   ```

Note: there is a small amount of unseeded randomness [here](https://github.com/allenai/ForeCite/tree/master/forecite/topic_identification/identify_topics.py:98) and so your output may differ slightly.

## Citing

If you use ForeCite in your research, please cite "High-Precision Extraction of Emerging Concepts from Scientific Literature".
```
@inproceedings{king-etal-2020-forecite,
    title = "{H}igh-{P}recision {E}xtraction of {E}merging {C}oncepts from {S}ientific {L}iterature",
    author = "King, Daniel  and
      Downey, Doug  and
      Weld, Daniel S.",
    booktitle = "Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’20)",
    month = Jul,
    year = "2020",
    address = "Virtual Event, China",
    publisher = "ACM",
    url = "https://doi.org/10.1145/3397271.3401235",
    doi = "10.1145/3397271.3401235",
}
```

ForeCite is an open-source project developed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
