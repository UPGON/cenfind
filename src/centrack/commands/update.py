"""
Entry point to update the model using new data via labelbox.
"""
import argparse

from spotipy.spotipy.model import SpotNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('dataset')
    parser.add_argument('config')

    parser.parse_args()

    return parser


def cli(args=None):
    # if args is None:
    #     raise ValueError('Please provide args')
    model = SpotNet(config=None)

    print(model)


if __name__ == '__main__':
    # parsed_args = parse_args()
    cli()
