from centrack.layout.dataset import DataSet
from centrack.layout.constants import PREFIX, datasets


def main():
    for ds in datasets:
        path_dataset = PREFIX / ds
        dataset = DataSet(path_dataset)
        train_images, test_images = dataset.create_splits(p=.9, )
        with open(path_dataset / 'train.txt', 'w') as f:
            for fov in train_images:
                f.write(f"{fov}\n")

        with open(path_dataset / 'test.txt', 'w') as f:
            for fov in test_images:
                f.write(f"{fov}\n")


if __name__ == '__main__':
    main()
