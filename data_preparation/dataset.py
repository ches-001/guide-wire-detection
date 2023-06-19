import torch, os, cv2
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Iterable, Optional, Union
from torchvision.transforms import Compose, Resize


class FrameDataset(Dataset):
    def __init__(
            self, 
            data_path: str, 
            bbox_loc: str="BBox", 
            img_loc: str="Image",  
            grayscale: bool=True, 
            size: Optional[Tuple[int, int]] = (224, 224),
            transforms: Optional[Compose]=None):
        
        self.data_path = data_path
        self.bbox_loc = bbox_loc
        self.img_loc = img_loc
        self.grayscale = grayscale
        self.size = size
        self.transforms = transforms
        self.img_bbox_pairs = self._get_imgs_and_bboxes()
        
        if size is not None:
            self._resize_module = Resize(self.size, antialias=True)


    def __len__(self) -> int:
        return len(self.img_bbox_pairs)
    

    def __getitem__(self, idx: int)->Tuple[torch.Tensor, torch.Tensor]:
        img_path, bbox_path = self.img_bbox_pairs[idx]
        img = self._load_img(img_path).astype(np.float32)
        bbox = self._load_bbox(bbox_path)
        img = self._normalize_img(img)
        img = torch.from_numpy(img)
        
        if img.ndim == 2:
            img = img.unsqueeze(dim=0)
        img = self._resize(img)

        bbox = torch.Tensor(bbox)
        bbox = torch.cat([torch.ones(1), bbox], axis=0)     # format: [confidence, x, y, w, h]

        if self.transforms:
            img = self.transforms(img)
        return img, bbox


    def _get_imgs_and_bboxes(self) -> Iterable[Tuple[str, str]]:
        path_tuples = []

        for sample_feed_folder in os.listdir(self.data_path):
            sample_feed_folder = os.path.join(self.data_path, sample_feed_folder)
            sample_frames_folder = os.path.join(sample_feed_folder, self.img_loc)
            sample_bboxes_folder = os.path.join(sample_feed_folder, self.bbox_loc)
            sample_frames = os.listdir(sample_frames_folder)
            sample_bboxes = os.listdir(sample_bboxes_folder)

            for img, bbox in zip(sample_frames, sample_bboxes):
                img = os.path.join(sample_frames_folder, img)
                bbox = os.path.join(sample_bboxes_folder, bbox)
                path_tuples.append((img, bbox))
        return path_tuples


    def _load_img(self, img_path: str) -> np.ndarray:
        color_flag = cv2.COLOR_BGR2RGB if not self.grayscale else cv2.IMREAD_GRAYSCALE
        img = cv2.imread(img_path, color_flag)
        return img


    def _resize(self, img: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "_resize_module"):
            return img
        return self._resize_module(img)


    def _load_bbox(self, bbox_path: str) -> Tuple[float]:
        assert bbox_path.split(".")[-1] == "txt", "bound boxes must be stored in .txt files"
        with open(bbox_path, "r") as f:
            bbox = tuple(map(lambda x : float(x), f.read().split()[1:]))
        f.close()
        return bbox     # (x, y, w, h)


    def _normalize_img(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        img = (img - img.min()) / (img.max() - img.min())
        return img