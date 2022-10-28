import pandas as pd

from cenfind.core.measure import field_score_frequency

if __name__ == '__main__':
    data = pd.read_csv('/data1/centrioles/CEP164_Test_1_INCell2_Comp_3D_25D_1_proj/statistics/scores_df.tsv', sep='\t')
    summary = field_score_frequency(data)
    summary.to_csv('/data1/centrioles/CEP164_Test_1_INCell2_Comp_3D_25D_1_proj/statistics/statistics.tsv', sep='\t', index=True)
    print(0)
