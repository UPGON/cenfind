from datetime import datetime
from cenfind.experiments.constants import PREFIX_REMOTE, datasets
from centrosome_analysis.ml_foci_detect import train_model_fcn, FociDetector
import torch


def main():
    train_files = []
    for dataset in datasets:
        print(f'loading {dataset}')
        with open(PREFIX_REMOTE / dataset / 'train.txt', 'r') as f:
            file = f.readlines()
            for line in file:
                field_name, channel = line.rstrip().split(',')
                img_path = PREFIX_REMOTE / dataset / 'projections' / f"{field_name}_max.tif"
                annotation_path = PREFIX_REMOTE / dataset / 'annotations' / 'centrioles' / f"{field_name}_max_C{channel}.txt"
                train_files.append((img_path, annotation_path))

    model = FociDetector()
    print('Start learning')
    model = train_model_fcn(model, train_files=train_files, need_sigmoid=True, num_epochs=100)
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    torch.save(model.state_dict(), f'models/sankaran/dev/{time_stamp}')


if __name__ == '__main__':
    main()
