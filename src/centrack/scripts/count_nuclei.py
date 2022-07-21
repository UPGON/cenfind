import os
import labelbox
import pandas as pd
from dotenv import load_dotenv

load_dotenv('/Users/buergy/Dropbox/epfl/projects/centrack/.env')


def main():
    lb = labelbox.Client(api_key=os.environ['LABELBOX_API_KEY'])
    project = lb.get_project('cl5gnefndcjvi08wodvn05thr')
    labels = project.label_generator()
    counts = []
    for label in labels:
        external_name = label.data.external_id
        if external_name.endswith('C0.png'):
            nuclei_in_label = [lab for lab in label.annotations if lab.name == 'Nucleus']
            dataset_name = label.extra['Dataset Name']
            nuclei_n = len(nuclei_in_label)
            counts.append((dataset_name, nuclei_n))

    print(counts)
    pd.DataFrame(counts).to_csv('out/nuclei_counts.csv')


if __name__ == '__main__':
    main()
