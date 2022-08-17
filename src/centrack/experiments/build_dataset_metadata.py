from centrack.experiments.constants import datasets, pattern_dataset
import re


def main():
    for ds in datasets:
        res = extract_from(pattern, ds)
        print(res)
    return 0


if __name__ == '__main__':
    main()
