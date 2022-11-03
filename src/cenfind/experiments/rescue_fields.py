from collections import defaultdict
from pathlib import Path

import labelbox
from dotenv import dotenv_values
from tqdm import tqdm

config = dotenv_values('.env')

dst = Path('/data1/centrioles/')
dst.mkdir(exist_ok=True)

def main():
    lb = labelbox.Client(api_key=config['LABELBOX_API_KEY'])
    project = lb.get_project(config['PROJECT_CENTRIOLES'])
    labels = project.label_generator()

    container = defaultdict(list)

    for label in tqdm(labels):
        ds = label.extra['Dataset Name']
        external_name = label.data.external_id
        # foci_in_label = [lab for lab in label.annotations if lab.name == 'Centriole']
        stem = external_name.split('.')[0]
        channel = stem.split('_')[-1]
        channel_id = channel[-1]
        field_name = '_'.join(stem.split('_')[:-2])
        container[ds].append(f"{field_name},{channel_id}")

    for ds in container.keys():
        with open(dst / ds / f'_fields_channel.txt', 'w') as f:
            for line in container[ds]:
                f.write(line + "\n")
        with open(dst / ds / f'_fields.txt', 'w') as f:
            for line in container[ds]:
                field, channel = line.split(',')
                f.write(field + "\n")


if __name__ == '__main__':
    main()
