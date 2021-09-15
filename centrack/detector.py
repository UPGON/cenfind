from functools import partial

import flash
from flash.core.utilities.imports import example_requires
from flash.image import InstanceSegmentation, InstanceSegmentationData

example_requires("image")

import icedata  # noqa: E402

if __name__ == '__main__':

    # 1. Create the DataModule
    data_dir = icedata.pets.load_data()

    datamodule = InstanceSegmentationData.from_folders(
        train_folder=data_dir,
        val_split=0.1,
        parser=partial(icedata.pets.parser, mask=True),
    )

    # 2. Build the task
    model = InstanceSegmentation(
        head="mask_rcnn",
        backbone="resnet18_fpn",
        num_classes=datamodule.num_classes,
    )

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=1)
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    # 4. Detect objects in a few images!
    predictions = model.predict(
        [
            str(data_dir / "images/yorkshire_terrier_9.jpg"),
            str(data_dir / "images/english_cocker_spaniel_1.jpg"),
            str(data_dir / "images/scottish_terrier_1.jpg"),
        ]
    )
    print(predictions)

    # 5. Save the model!
    trainer.save_checkpoint("instance_segmentation_model.pt")