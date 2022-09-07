import argparse
from pathlib import Path
import labelbox
from shapely.errors import TopologicalError
import pandas as pd
from dotenv import dotenv_values
from tqdm import tqdm

config = dotenv_values('.env')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--destination', type=str, default=None)
    args = parser.parse_args()

    project_id = args.project
    if args.project is None:
        project_id = config['PROJECT_CENTRIOLES']

    lb = labelbox.Client(api_key=config['LABELBOX_API_KEY'])
    project = lb.get_project(project_id)
    labels = project.label_generator(timeout_seconds=90 * 100)

    counts = []
    pad_lower = int(.1 * 2048)
    pad_upper = 2048 - pad_lower
    for label in labels:
        nuclei_in_label = [lab for lab in label.annotations if lab.name == 'Nucleus']
        dataset_name = label.extra['Dataset Name']
        label_name = label.data.external_id
        print(dataset_name)

        nuclei_n = len(nuclei_in_label)
        at_edge = 0
        for nuc in tqdm(nuclei_in_label):
            try:
                coords = nuc.value.shapely.centroid.centroid.coords
            except TopologicalError:
                print(f"problem with label {dataset_name} ({label_name})")
                continue
            if len(coords) != 2:
                print(f"problem with coords: {coords}")
                continue
            centre = [int(i) for i in coords[0]]

            if not all(pad_lower < p < pad_upper for p in centre):
                at_edge += 1

        record = (dataset_name, label_name, nuclei_n, at_edge)
        counts.append(record)
        with open(Path(args.destination), 'a+') as f:
            f.write(f"{record}\n")

    data_df = pd.DataFrame(counts)
    if args.destination is not None:
        data_df.to_csv(args.destination)
    else:
        print(data_df)


if __name__ == '__main__':
    main()
