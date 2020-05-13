import argparse

import utils

def main(args):
    paths_file = args.files
    path_vocab = args.vocab

    utils.replace_oov_tokens(paths_in=paths_file, paths_out=paths_file, path_vocab=path_vocab)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--vocab", type=str, required=True)
    args = parser.parse_args()
    main(args)
