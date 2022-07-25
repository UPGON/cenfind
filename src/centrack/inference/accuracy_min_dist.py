import pandas as pd
from matplotlib import pyplot as plt


def main():
    data = pd.read_csv('out/performances_base_model.csv')
    print(data)
    return 0


if __name__ == '__main__':
    main()
