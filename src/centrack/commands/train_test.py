import random
from pathlib import Path
from centrack.utils.status import fetch_files
from centrack.utils.constants import PREFIX, datasets


def create_splits(path: Path, p=.9, suffix='.ome.tif') -> None:
    path_dataset = Path(path)
    path_raw = path_dataset / 'raw'
    files = fetch_files(path_raw, suffix)

    file_stems = [f.name.removesuffix(suffix) for f in files]

    size = len(file_stems)
    split_idx = int(p * size)

    random.seed(1993)
    shuffled = random.sample(file_stems, k=size)

    with open(path_dataset / 'train.txt', 'w') as f:
        for fov in shuffled[:split_idx]:
            f.write(f"{fov}\n")

    with open(path_dataset / 'test.txt', 'w') as f:
        for fov in shuffled[split_idx:]:
            f.write(f"{fov}\n")


def main():
    for ds in datasets:
        create_splits(PREFIX / ds)


if __name__ == '__main__':
    main()
