import torch, cv2
import numpy as np
from torchvision.transforms import Resize
from typing import Iterable, Union, Tuple, Optional


def prepare_bbox(
        bbox: Union[np.ndarray, torch.Tensor], 
        size: Optional[Tuple[int, int]]=None, 
        xy_centered: bool=True) -> np.ndarray:
    
    if bbox.shape[0] == 4:
        x, y, w, h = bbox
    else:
        _, x, y, w, h = bbox
    if size is not None:
        H, W = size
        x = x * W
        y = y * H
        w = w * W
        h = h * H
    if xy_centered:
        x = x-(w/2)
        y = y-(h/2)
    return np.array([x, y, w, h]).astype(int)


def prepare_img_and_bbox(
        img: torch.Tensor, 
        bbox: torch.Tensor, 
        scaled_bbox: bool=True,
        xy_centered: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    
    img = (img * 255).type(torch.uint8).numpy()
    _, H, W = img.shape
    size = (H, W) if scaled_bbox else None
    bbox = prepare_bbox(bbox, size, xy_centered=xy_centered)
    img = np.transpose(img, axes=(1, 2, 0))
    if img.shape[-1] == 1:
        np.squeeze(img, axis=-1)
    return img, bbox


def draw_bbox_on_img(
        img :np.ndarray, 
        bbox: np.ndarray, 
        grayscale: bool=True) -> np.ndarray:
    
    if bbox.shape[0] == 4:
        x, y, w, h = bbox
    else:
        _, x, y, w, h = bbox
    if not grayscale:
        ch = (0, 200, 90) 
    else: 
        ch = (255,)
    return cv2.rectangle(img.copy(), (x, y), (x+w, y+h), ch, 2)


def resize_img(
        img: Union[np.ndarray, torch.Tensor], 
        size: Tuple[int, int]) -> Union[np.ndarray, torch.Tensor]:
    
    if isinstance(img, np.ndarray):
        return cv2.resize(img, size)
    
    elif isinstance(img, torch.Tensor):
        resize_module = Resize(size=size)
        return resize_module(img)
    else:
        raise ValueError("img should either be 'ndarray' or 'Tensor'")


def normalize_img(img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        img = (img - img.min()) / (img.max() - img.min())
        return img


def to_tensor(img: np.ndarray, device: str="cpu") -> torch.Tensor:
    return torch.from_numpy(img).to(device)