from typing import List, Tuple

import argparse
import csv
import ast
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

from sklearn.metrics import auc


def make_curve(
    scores_and_labels: List[Tuple[int, int, List[str]]]
) -> List[Tuple[int, List[str], float, int]]:
    sorted_scores_and_labels = sorted(scores_and_labels, key=lambda x: x[0])
    points = []
    running_total = 0
    running_correct = 0
    running_tp_estimate = 0
    prev_index = -1
    for index, label, noun_phrases in sorted_scores_and_labels:
        running_total += 1
        running_correct += label
        running_precision = running_correct / running_total
        running_tp_estimate += running_precision * (index - prev_index)
        point = (
            index,
            noun_phrases,
            running_precision,
            int(running_precision * index),
            running_tp_estimate,
        )
        points.append(point)
        prev_index = index

    return points


def make_curve_incremental(scores_and_labels: List[Tuple[int, int, List[str]]]):
    sorted_scores_and_labels = sorted(scores_and_labels, key=lambda x: x[0])
    points = []
    running_total = 0
    running_correct = 0
    total_credit = 0
    num_labels_incremental = 0
    prev_index = -1
    for index, label, noun_phrases in sorted_scores_and_labels:
        running_total += 1
        running_correct += label
        num_labels_incremental += 1
        if label == 1:
            interval_precision = 1 / num_labels_incremental
            interval_length = index - prev_index
            interval_credit = interval_precision * interval_length
            total_credit += interval_credit
            point = (index, noun_phrases, running_correct / running_total, total_credit)
            points.append(point)
            prev_index = index
            num_labels_incremental = 0

    return points


def average_precision(curve: List[Tuple[int, List[str], float, int, float]]) -> float:
    total_credit = 0
    prev_index = -1
    for point in curve:
        index = point[0]
        running_precision = point[2]

        interval_credit = (index - prev_index) * running_precision
        total_credit += interval_credit
        prev_index = index

    x_coordinate_of_final_point = curve[-1][-1]
    return total_credit / x_coordinate_of_final_point


def main(
    annotation_file_path: str, output_plot_path: str, output_plot_path_incremental: str
):
    column_to_index = {
        "i": 0,
        "noun_phrases": 1,
        "citation_paper": 2,
        "label": 3,
        "citation_index": 4,
        "loor_index": 5,
        "cnlc_index": 6,
    }

    citation_scores_and_labels = []
    cnlc_scores_and_labels = []
    loor_scores_and_labels = []

    with open(annotation_file_path) as _csv_file:
        csv_reader = csv.reader(_csv_file, delimiter=",")

        for i, row in enumerate(csv_reader):
            if i == 0 or row[0] == "":
                continue

            label = row[column_to_index["label"]]
            citation_index = int(row[column_to_index["citation_index"]])
            cnlc_index = int(row[column_to_index["cnlc_index"]])
            loor_index = int(row[column_to_index["loor_index"]])
            noun_phrases = ast.literal_eval(row[column_to_index["noun_phrases"]])

            if citation_index >= 0:
                citation_scores_and_labels.append(
                    (citation_index, int(label), noun_phrases)
                )
            if cnlc_index >= 0:
                cnlc_scores_and_labels.append((cnlc_index, int(label), noun_phrases))
            if loor_index >= 0:
                loor_scores_and_labels.append((loor_index, int(label), noun_phrases))

    citation_curve = make_curve(citation_scores_and_labels)
    cnlc_curve = make_curve(cnlc_scores_and_labels)
    loor_curve = make_curve(loor_scores_and_labels)

    fig, ax1 = plt.subplots()
    ax1.set_ylim(0.7, 1.01)
    ax1.set_xscale("linear")

    ax1.plot(
        [running_tp for _, _, _, _, running_tp in citation_curve],
        [precision for _, _, precision, _, _ in citation_curve],
        "b-",
        label="citation",
    )
    ax1.plot(
        [running_tp for _, _, _, _, running_tp in cnlc_curve],
        [precision for _, _, precision, _, _ in cnlc_curve],
        "r--",
        label="cnlc",
    )
    ax1.plot(
        [running_tp for _, _, _, _, running_tp in loor_curve],
        [precision for _, _, precision, _, _ in loor_curve],
        "y-.",
        label="loor",
    )

    ax1.set_xlabel("Yield")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision vs Yield")
    ax1.legend()

    fig.savefig(output_plot_path)
    plt.close(fig)

    citation_tps = [running_tp for _, _, _, _, running_tp in citation_curve]
    citation_precisions = [precision for _, _, precision, _, _ in citation_curve]
    citation_auc = auc(citation_tps, citation_precisions)
    cnlc_tps = [running_tp for _, _, _, _, running_tp in cnlc_curve]
    cnlc_precisions = [precision for _, _, precision, _, _ in cnlc_curve]
    cnlc_auc = auc(cnlc_tps, cnlc_precisions)
    loor_tps = [running_tp for _, _, _, _, running_tp in loor_curve]
    loor_precisions = [precision for _, _, precision, _, _ in loor_curve]
    loor_auc = auc(loor_tps, loor_precisions)

    max_auc = 15000
    citation_aoc = max_auc - citation_auc
    cnlc_aoc = max_auc - cnlc_auc
    loor_aoc = max_auc - loor_auc

    print(f"Citation AUC {citation_auc}, AOC: {citation_aoc}")
    print(f"CNLC AUC {cnlc_auc}, AOC: {cnlc_aoc}")
    print(f"LOOR AUC {loor_auc}, AOC: {loor_aoc}")
    print(f"Citation vs cnlc reduction {(cnlc_aoc - citation_aoc)/cnlc_aoc}")
    print(f"Citation vs loor reduction {(loor_aoc - citation_aoc)/loor_aoc}")

    citation_ap = average_precision(citation_curve)
    cnlc_ap = average_precision(cnlc_curve)
    loor_ap = average_precision(loor_curve)

    print(f"Citation AP {citation_ap}")
    print(f"CNLC AP {cnlc_ap}")
    print(f"LOOR AP {loor_ap}")

    print("STARTING INCREMENTAL SECTION")

    citation_curve_incremental = make_curve_incremental(citation_scores_and_labels)
    cnlc_curve_incremental = make_curve_incremental(cnlc_scores_and_labels)
    loor_curve_incremental = make_curve_incremental(loor_scores_and_labels)

    fig, ax1 = plt.subplots()
    ax1.set_ylim(0.7, 1.01)
    ax1.set_xscale("linear")

    ax1.plot(
        [total_credit for _, _, _, total_credit in citation_curve_incremental],
        [precision for _, _, precision, _, in citation_curve_incremental],
        "b-",
        label="Ours",
    )
    ax1.plot(
        [total_credit for _, _, _, total_credit in cnlc_curve_incremental],
        [precision for _, _, precision, _ in cnlc_curve_incremental],
        "r--",
        label="CNLC",
    )
    ax1.plot(
        [total_credit for _, _, _, total_credit in loor_curve_incremental],
        [precision for _, _, precision, _ in loor_curve_incremental],
        "y-.",
        label="LoOR",
    )

    ax1.set_xlabel("Estimated Yield")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision vs Estimated Yield")
    ax1.legend()

    fig.savefig(output_plot_path_incremental)
    plt.close(fig)

    citation_x = [total_credit for _, _, _, total_credit in citation_curve_incremental]
    citation_y = [precision for _, _, precision, _ in citation_curve_incremental]
    citation_auc = auc(citation_x, citation_y)
    cnlc_x = [total_credit for _, _, _, total_credit in cnlc_curve_incremental]
    cnlc_y = [precision for _, _, precision, _ in cnlc_curve_incremental]
    cnlc_auc = auc(cnlc_x, cnlc_y)
    loor_x = [total_credit for _, _, _, total_credit in loor_curve_incremental]
    loor_y = [precision for _, _, precision, _, in loor_curve_incremental]
    loor_auc = auc(loor_x, loor_y)

    print(f"Citation AUC {citation_auc}")
    print(f"CNLC AUC {cnlc_auc}")
    print(f"LOOR AUC {loor_auc}")

    citation_aoc = citation_curve_incremental[-1][-1] - citation_auc
    cnlc_aoc = cnlc_curve_incremental[-1][-1] - cnlc_auc
    loor_aoc = loor_curve_incremental[-1][-1] - loor_auc

    print(f"Citation AOC {citation_aoc}")
    print(f"CNLC AOC {cnlc_aoc}")
    print(f"LOOR AOC {loor_aoc}")
    print(f"Citation vs cnlc reduction {(cnlc_aoc - citation_aoc)/cnlc_aoc}")
    print(f"Citation vs loor reduction {(loor_aoc - citation_aoc)/loor_aoc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file_path")
    parser.add_argument("--output_plot_path")
    parser.add_argument("--output_plot_path_incremental")

    args = parser.parse_args()
    main(
        args.annotation_file_path,
        args.output_plot_path,
        args.output_plot_path_incremental,
    )
