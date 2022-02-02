def get_lb_api_key(path):
    with open(path, 'r') as apikey:
        lb_api_key = apikey.readline().rstrip('\n')
    return lb_api_key


def get_dataset_uid(client, name):
    """
    Retrieves the uid of a possibly existing dataset.
    :param client:
    :param name:
    :return:
    """
    datasets = client.get_datasets()
    dataset_ids = {ds.name: ds.uid for ds in datasets}
    if name in dataset_ids.keys():
        return dataset_ids[name]
    else:
        return None


def get_project_uid(client, name):
    """
    Retrieves the uid of a possibly existing project.
    :param client:
    :param name:
    :return:
    """
    projects = client.get_projects()
    project_ids = {proj.name: proj.uid for proj in projects}
    if name in project_ids.keys():
        return project_ids[name]
    else:
        return None
