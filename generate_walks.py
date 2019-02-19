from tensorflow_node2vec.utils import generate_random_walks, generate_n2v_walks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("graph")
    parser.add_argument("sequences")
    parser.add_argument("vocab")
    parser.add_argument("-n2v",  action="store_true")
    parser.add_argument("-l", required=False, default=40, type=int)
    parser.add_argument("-n", required=False, default=5, type=int)
    parser.add_argument("-p", required=False, default=0.5, type=float)
    parser.add_argument("-q", required=False, default=0.5, type=float)
    parser.add_argument("-b", required=False, default=256, type=int)

    args = parser.parse_args()

    if args.n2v:
        walks, voc = generate_n2v_walks(
            args.graph, args.l, args.n, args.p, args.q, batchsize=args.b)
    else:
        walks, voc = generate_random_walks(
            args.graph, args.l, args.n, batchsize=args.b)
    print("The vocabulary contains {} entries".format(len(voc)))
    print("{} sequences generated ".format(sum([w.shape[0] for w in walks])))
    # with open(args.sequences, "w") as _:
    #     for walk_batch in walks:
    #         for i in range(walk_batch.shape[0]):
    #             _.write(" ".join([str(x) for x in walk_batch[i]]) + "\n")

    # with open(args.vocab, "w") as _:
    #     _.write("\n".join(voc))
