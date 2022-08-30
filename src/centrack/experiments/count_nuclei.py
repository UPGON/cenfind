import argparse

import labelbox
import pandas as pd
from dotenv import dotenv_values

config = dotenv_values('../../../.env')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


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
    labels = project.label_generator()

    counts = []
    for label in labels:
        nuclei_in_label = [lab for lab in label.annotations if lab.name == 'Nucleus']
        dataset_name = label.extra['Dataset Name']
        nuclei_n = len(nuclei_in_label)
        counts.append((dataset_name, nuclei_n))

    data_df = pd.DataFrame(counts)
    if args.destination is not None:
        data_df.to_csv(args.destination)
    else:
        print(data_df)


if __name__ == '__main__':
    main()
