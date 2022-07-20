from keras.utils.vis_utils import plot_model
from centrack.inference.score import get_model


def main():
    model = get_model('../../../models/master')

    plot_model(model.keras_model,
               show_shapes=False,
               dpi=300,
               to_file='/home/buergy/projects/centrack/publication/data/spotnet_architecture.png')
    plot_model(model.keras_model,
               show_shapes=True,
               dpi=300,
               to_file='/home/buergy/projects/centrack/publication/data/spotnet_architecture_with_shapes.png')


if __name__ == '__main__':
    main()
