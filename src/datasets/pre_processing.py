import torchvision
from torchvision import transforms
from torchvision.datasets.vision import StandardTransform

from src.datasets.road_segmentation_dataset import RoadSegmentationDataset
from src.utils.template_utils import imshow


class PreProcessor:
    def __init__(self, root: str = "data/") -> None:
        self.root = root
        self.pre_process_erasing()

    def pre_process_erasing(self) -> None:
        transform_image = torchvision.transforms.Compose(
            [
                transforms.Resize(255),
                transforms.ToTensor(),
                transforms.RandomErasing()
                # transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        transform_mask = transforms.Compose(
            [transforms.Resize(255), transforms.ToTensor()]
        )

        standard_tranform = StandardTransform(transform_image, transform_mask)
        dataset = RoadSegmentationDataset(
            self.root, train=True, download=True, transforms=standard_tranform
        )
        image, mask = dataset.__getitem__(40)

        imshow(image)
        imshow(mask)

        print(image.size(), mask.size(), image.dtype, mask.dtype)

        # torchvision.transforms.ToPILImage()(image).save("test.png")


def main() -> None:
    PreProcessor()


if __name__ == "__main__":
    main()
