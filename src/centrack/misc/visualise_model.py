from keras.utils.vis_utils import plot_model
from spotipy.spotipy.model import SpotNet, Config
from keras.models import load_model
from centrack.commands.score import get_model


def main():
    config = Config(axes="YXC", n_channel_in=1, train_patch_size=(64, 64),
                    activation='elu', last_activation="sigmoid", backbone='unet')
    model = get_model('../../../models/leo3_multiscale_True_mae_aug_1_sigma_1.5_split_2_batch_2_n_300')

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
