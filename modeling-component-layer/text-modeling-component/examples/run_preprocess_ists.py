import argparse

from clrcmd.evaluation.ists import AlignmentPair, load_alignment, save_alignment

parser = argparse.ArgumentParser()
parser.add_argument("--alignment-path", type=str, required=True)


def main():
    args = parser.parse_args()
    gold_alignments = load_alignment(args.alignment_path)
    for alignment in gold_alignments:
        words1 = alignment["sent1"].split()
        words2 = alignment["sent2"].split()
        if words1[-1] == "." and words2[-1] == ".":
            alignment["pairs"].append(
                AlignmentPair(
                    sent1_word_ids=[len(words1)],
                    sent2_word_ids=[len(words2)],
                    type="EQUI",
                    score=5.0,
                    comment=". == .",
                )
            )

    for alignment in gold_alignments:
        alignment["pairs"] = [
            x for x in alignment["pairs"] if x["type"] != "NOALI" and x["score"] >= 3.0
        ]

    outfile = f"{args.alignment_path}.equi"
    save_alignment(gold_alignments, outfile)


if __name__ == "__main__":
    main()
