from stardist.models import Config2D, StarDist2D, StarDistData2D
from glob import glob


def main():
    images = sorted(glob('/data1/centrioles/rpe/projections/*_max_C2.tif'))
    masks = sorted(glob('/data1/centrioles/rpe/projections/*_max_C2.tif'))

    print(images)
    print(masks)


if __name__ == '__main__':
    main()
