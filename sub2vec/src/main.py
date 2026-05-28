import argparse
import time

from structural import structural_embedding
from neighborhood import neighborhood_embedding


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="sub2vec.")

    parser.add_argument(
        "--input",
        nargs="?",
        required=True,
        help="Input directory"
    )

    parser.add_argument(
        "--property",
        default="n",
        choices=["n", "s"],
        required=True,
        help='Type of subgraph property to preserve. Use "n" for neighborhood and "s" for structural.'
    )

    parser.add_argument(
        "--walkLength",
        default=100000,
        type=int,
        help="Length of random walk on each subgraph"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output representation file"
    )

    parser.add_argument(
        "--d",
        default=128,
        type=int,
        help="Dimension of learned features for each subgraph"
    )

    parser.add_argument(
        "--iter",
        default=20,
        type=int,
        help="Training iterations"
    )

    parser.add_argument(
        "--windowSize",
        default=2,
        type=int,
        help="Window size of the model"
    )

    parser.add_argument(
        "--p",
        default=0.5,
        type=float,
        help="Meta parameter"
    )

    parser.add_argument(
        "--model",
        default="dm",
        choices=["dbon", "dm"],
        help="Model: SV-DM (dm) or SV-DBON (dbon)"
    )

    args = parser.parse_args()

    print("\n=== SUB2VEC CONFIG ===")
    print(args)

    if args.property == "s":
        structural_embedding(args)
    else:
        neighborhood_embedding(args)

    total_time = time.time() - start_time

    print("\n=== SUB2VEC DONE ===")
    print(f"Output file: {args.output}")
    print(f"Total runtime: {total_time:.2f}s")
    print(f"Total runtime: {total_time / 60:.2f}min")


if __name__ == "__main__":
    main()
