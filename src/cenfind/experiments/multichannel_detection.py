from cenfind.experiments.constants import datasets

main_channels = ['CEP63', 'CP110', 'CEP152', 'CEP63', 'CEP63']

def main():
    for ds, ch in zip(datasets, main_channels):
        print(ds, ch)

if __name__ == '__main__':
    main()
