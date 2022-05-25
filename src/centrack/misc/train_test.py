import random
from pathlib import Path
from src.centrack.commands.status import fetch_files


def main():
    p = .9
    suffix = '.ome.tif'
    path_dataset = Path('/data1/centrioles/rpe')
    path_raw = path_dataset / 'raw'
    files = fetch_files(path_raw, suffix)

    file_stems = [f.name.removesuffix(suffix) for f in files]

    size = len(file_stems)
    split_idx = int(p * size)

    shuffled = random.sample(file_stems, k=size)

    with open(path_dataset / 'train.txt', 'w') as f:
        for fov in shuffled[:split_idx]:
            f.write(f"{fov}\n")

    with open(path_dataset / 'test.txt', 'w') as f:
        for fov in shuffled[split_idx:]:
            f.write(f"{fov}\n")


if __name__ == '__main__':
    main()
