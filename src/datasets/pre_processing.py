import torchvision
from torchvision import transforms
from torchvision.datasets.vision import StandardTransform

from src.datasets.road_segmentation_dataset import RoadSegmentationDataset
from src.datasets.transforms.random_crop import RandomCrop
from src.utils.template_utils import imshow


class PreProcessor:
    def __init__(self, root: str = "data/") -> None:
        self.root = root
        self.pre_process_erasing()

    def pre_process_erasing(self) -> None:
        transform_image = torchvision.transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        transform_mask = transforms.Compose([transforms.ToTensor()])

        standard_tranform = StandardTransform(transform_image, transform_mask)
        dataset = RoadSegmentationDataset(
            self.root, train=True, download=True, transforms=standard_tranform
        )
        image, mask = dataset.__getitem__(40)

        imshow(image)
        imshow(mask)

        cropper = RandomCrop(output_size=(200, 200))
        # flip = RandomFlip(1)
        images = cropper(image, mask)
        for image in images:
            imshow(image)

        print(image.size(), mask.size(), image.dtype, mask.dtype)


def main() -> None:
    PreProcessor()


if __name__ == "__main__":
    main()
