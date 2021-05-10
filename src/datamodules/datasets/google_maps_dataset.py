import io
import os
import re
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.io import imsave


class GoogleMapsDataset(torch.utils.data.Dataset):
    @property
    def folder_processed(self) -> str:
        return os.path.join(
            self.root, self._camel_to_snake(self.__class__.__name__), "processed"
        )

    @property
    def folder_train(self) -> str:
        return os.path.join(self.folder_processed, "train")

    def __init__(
        self,
        root: str = "data/",
        download: bool = False,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.transforms = transforms
        self.images: List[str] = []
        self.masks: List[str] = []
        self.indeces: torch.Tensor = torch.tensor([])

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        self.process()

    def randomize_image(self) -> Any:
        noise_coords = torch.rand(200, 2)

        # Some coordinates with good looking roads near Chicago
        init_lat, init_long = 41.776796856026245, -88.29816520687764

        # Generate 100 coordinates by adding random noise to previous init coordinates
        coords = torch.add(torch.Tensor([init_lat, init_long]), 0.3 * noise_coords)
        return coords

    def construct_image_mask(self, latitude: str, longitude: str) -> Tuple[Any, Any]:
        base_url = "https://maps.googleapis.com/maps/api/staticmap?"

        zoom_sat = "&zoom=17"
        zoom_road = "&zoom=18"
        size = "&size=512x512"
        key = "&key="
        remove_markers = "&style=feature:all|element:labels|visibility:off"

        YOUR_API_KEY = "AIzaSyA2h9DsyONlCFOo4w5p9RF3J96rV8mCVyQ"
        sat_type = "&maptype=satellite"
        roadmap_type = "&maptype=roadmap"

        # Construct the url for the satellite and the roadmap type images
        sat_url = (
            base_url
            + "center="
            + latitude
            + ","
            + longitude
            + zoom_sat
            + sat_type
            + size
            + remove_markers
            + key
            + YOUR_API_KEY
        )
        roadmap_url = (
            base_url
            + "center="
            + latitude
            + ","
            + longitude
            + zoom_road
            + roadmap_type
            + size
            + remove_markers
            + key
            + YOUR_API_KEY
        )

        # Fetch maps images from these URLs
        sat_img = np.array(
            Image.open(io.BytesIO(requests.get(sat_url).content)).convert("RGB")
        )
        roadmap_img = np.array(
            Image.open(io.BytesIO(requests.get(roadmap_url).content)).convert("RGB")
        )

        # Construct mask from the roadmap image, easy as roads are in full white
        mask = np.floor(rgb2gray(np.floor(roadmap_img >= 254))).astype(np.float32)

        return sat_img, mask

    def download(self) -> None:

        # remove the not once the dataset is constructed
        if not self._check_exists():
            return

        os.makedirs(self.folder_processed, exist_ok=True)

        print("Downloading google maps dataset.")

        coords = self.randomize_image()

        for i, coord in enumerate(coords):
            lat = str(float(coord[0]))
            lon = str(float(coord[1]))
            img, mask = self.construct_image_mask(lat, lon)
            self.images.append(img)
            self.masks.append(mask)
            imsave(
                os.path.join(self.folder_train, "{}_maps_image.png".format(i)),
                img_as_ubyte(img),
            )
            imsave(
                os.path.join(self.folder_train, "{}_maps_mask.png".format(i)),
                img_as_ubyte(mask),
            )

    def process(self) -> None:

        images, masks = [], []

        folder_train_files = [
            f
            for f in os.listdir(self.folder_train)
            if os.path.isfile(os.path.join(self.folder_train, f))
        ]

        images = [
            os.path.join(self.folder_train, f)
            for f in folder_train_files
            if "image" in f
        ]

        masks = [
            os.path.join(self.folder_train, f)
            for f in folder_train_files
            if "mask" in f
        ]

        self.images = sorted(images)
        self.masks = sorted(masks)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask

    def __len__(self) -> int:
        return len(self.images)

    def _check_exists(self) -> bool:
        print(self.folder_train)

        return os.path.exists(self.folder_train)

    def _camel_to_snake(self, name: str) -> str:
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

    def _img_number_from_name(self, name: str) -> int:
        return int(re.compile(r"\d+").findall(name)[0])
