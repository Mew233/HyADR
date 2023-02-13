import torch
import argparse
from pipeline import *

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="data/")
    parser.add_argument('--output_path', type=str, default="results/")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=15)

    return parser.parse_args()


def main():
    args = arg_parse()
    data_path = args.data_path
    output_path = args.output_path

    learning_rate = args.learning_rate
    num_epochs = args.num_epochs

    train_test(data_path=data_path, output_path=output_path, learning_rate=learning_rate, number_epochs=num_epochs)



if __name__ == "__main__":
    main() 
