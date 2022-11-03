from datetime import datetime

import torch
from centrosome_analysis.ml_foci_detect import train_model_fcn, FociDetector

from cenfind.experiments.constants import PREFIX_REMOTE, datasets


def main():
    train_files = []
    fixed_channel = 1
    for dataset in datasets:
        print(f'loading {dataset}')
        with open(PREFIX_REMOTE / dataset / 'train.txt', 'r') as f:
            file = f.readlines()
            for line in file:
                field_name, channel = line.rstrip().split(',')
                channel = fixed_channel or channel
                img_path = PREFIX_REMOTE / dataset / 'projections' / f"{field_name}_max.tif"
                annotation_path = PREFIX_REMOTE / dataset / 'annotations' / 'centrioles' / f"{field_name}_max_C{channel}.txt"
                print(f'Appending {img_path.name} and {annotation_path.name}...')
                train_files.append((img_path, annotation_path))

    print('Start learning...')
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    model = FociDetector()
    fit = train_model_fcn(model, train_files=train_files, need_sigmoid=True, num_epochs=100)
    torch.save(fit.state_dict(), f'models/sankaran/dev/{time_stamp}')

    # model_multi = MultiChannelCombinedScorer()
    # fit_multi = train_model_fcn(model_multi, train_files=train_files, need_sigmoid=True, num_epochs=100)
    # torch.save(fit_multi.state_dict(), f'models/sankaran/dev/{time_stamp}_multi')


if __name__ == '__main__':
    main()
