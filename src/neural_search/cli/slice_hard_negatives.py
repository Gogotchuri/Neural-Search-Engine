from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter/slice mined BM25 candidates into a hard-negative training JSONL."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-negatives", type=int, default=2)
    parser.add_argument("--rank-start", type=int, default=0)
    parser.add_argument("--rank-end", type=int, default=None)
    parser.add_argument("--max-score", type=float, default=None)
    parser.add_argument("--max-score-ratio", type=float, default=None)
    args = parser.parse_args()

    if args.num_negatives <= 0:
        raise ValueError("num-negatives must be positive")

    if args.rank_start < 0:
        raise ValueError("rank-start must be non-negative")

    if args.rank_end is not None and args.rank_end <= args.rank_start:
        raise ValueError("rank-end must be greater than rank-start")

    if args.max_score_ratio is not None and args.max_score_ratio <= 0:
        raise ValueError("max-score-ratio must be positive")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as in_file, open(
        output_path, "w", encoding="utf-8"
    ) as out_file:
        for line in in_file:
            if not line.strip():
                continue

            row = json.loads(line)

            negatives = row.get("hard_negatives", [])
            scores = row.get("hard_negative_scores", [])

            if not negatives or not scores or len(negatives) != len(scores):
                skipped += 1
                continue

            top_score = scores[0]
            candidates = list(zip(negatives, scores))

            candidates = candidates[args.rank_start : args.rank_end]

            filtered = []
            for negative_text, score in candidates:
                if args.max_score is not None and score > args.max_score:
                    continue

                if (
                    args.max_score_ratio is not None
                    and top_score > 0
                    and score / top_score > args.max_score_ratio
                ):
                    continue

                filtered.append((negative_text, score))

                if len(filtered) >= args.num_negatives:
                    break

            if len(filtered) < args.num_negatives:
                skipped += 1
                continue

            selected_negatives = [text for text, _ in filtered]
            selected_scores = [score for _, score in filtered]

            output_row = {
                "query": row["query"],
                "positive_passage": row["positive_passage"],
                "hard_negatives": selected_negatives,
                "hard_negative_scores": selected_scores,
                "source_negative_file": str(input_path),
                "rank_start": args.rank_start,
                "rank_end": args.rank_end,
                "max_score": args.max_score,
                "max_score_ratio": args.max_score_ratio,
            }

            out_file.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} examples to {output_path}")
    print(f"Skipped {skipped} examples with too few valid negatives")


if __name__ == "__main__":
    main()