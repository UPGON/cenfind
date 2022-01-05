def main():
    field = load_ome()
    projection_foci = project(field, channel=1)
    projection_nuclei = project(field, channel=0)

    foci = detect_foci(projection_foci)
    nuclei = segment_nuclei(projection_nuclei)

    res = assign(foci, nuclei)

    save_result('dst/res.json', res)


if __name__ == '__main__':
    main()
