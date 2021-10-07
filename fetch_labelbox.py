import labelbox
import pandas as pd

with open('configs/labelbox_api_key.txt', 'r') as apikey:
    LB_API_KEY = apikey.readline()
    LB_API_KEY = LB_API_KEY.rstrip('\n')


def main():
    lb = labelbox.Client(api_key=LB_API_KEY)
    project = lb.get_project('ckropx5l72brw0z8b3kq15l6r')
    labels = project.export_labels(download=True)

    data_export = []
    channel_id = 2

    for label_id, label in enumerate(labels):

        image_name = label['External ID']
        condition, markers, replicate, ds_row, ds_col, _, channel_code = image_name.split('.')[0].split('_')
        markers = markers.split('+')
        markers.insert(0, 'DAPI')
        marker = markers[channel_id]

        if f'C{channel_id}' not in image_name:
            continue
        if 'objects' not in label['Label'].keys():
            continue
        foci = label['Label']['objects']

        for focus_id, focus in enumerate(foci):
            point = focus['point']
            x = int(point['x'])
            y = int(point['y'])

            focus_dict = {'image_name': image_name,
                          'ds_row': int(ds_row),
                          'ds_col': int(ds_col),
                          'channel_id': channel_id,
                          'marker': marker,
                          'focus_id': focus_id,
                          'x': x,
                          'y': y}
            data_export.append(focus_dict)
    data_export_df = pd.DataFrame(data_export)
    data_export_df.to_csv('out/export.csv', index=False)
    print(0)


if __name__ == '__main__':
    main()
    print(0)
