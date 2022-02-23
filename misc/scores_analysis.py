from pathlib import Path
import pandas as pd


def main():
    dataset_path = Path('/Volumes/work/epfl/datasets/RPE1wt_CP110+GTU88+PCNT_2')
    results_path = dataset_path / 'results'

    assigned = pd.read_csv(results_path / 'results.csv')
    scores = assigned.groupby(['fov', 'nucleus']).count()['centriole']
    scores.to_csv(dataset_path / 'scores.csv')
    print(0)


if __name__ == '__main__':
    main()