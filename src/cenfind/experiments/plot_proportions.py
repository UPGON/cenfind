import pandas as pd
from cenfind.core.data import Dataset
from cenfind.core.analytics import reduce_data, generate_figure


def main():
    dataset = Dataset("/data1/centrioles/20230106/20230106_60X_Grenier_RPE_Plate2_951")

    data = pd.read_csv(
        dataset.statistics / "statistics.tsv", sep="\t", index_col=[0, 1], header=[0, 1]
    )
    data.columns = data.columns.droplevel(0)
    summed = reduce_data(data)
    fig = generate_figure(summed)
    fig.savefig(dataset.statistics / "layout_score.png", dpi=300)


if __name__ == "__main__":
    main()
