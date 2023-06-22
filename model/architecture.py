import torch, torchvision, json
import torch.nn as nn
from typing import Optional, Callable, Iterable, Tuple
from torchvision.models.resnet import ResNet, BasicBlock


class DetectNET(nn.Module):
    def __init__(
            self, 
            n_anchors: int, 
            n_classes: int, 
            last_fmap_ch: int=512):
        super(DetectNET, self).__init__()

        self.n_anchors = n_anchors
        self.n_classes = n_classes
        self.last_fmap_ch = last_fmap_ch
        self.out_channels = self.n_anchors * (self.n_classes + 5)
        self.projectors = self._build_conv_projector()

    def forward(
            self, 
            fmaps: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_fmaps = []
        for i, fmap in enumerate(fmaps):
            fmap = self.projectors[i](fmap)
            new_fmaps.append(fmap)
        return tuple(new_fmaps)
    
    def _build_conv_projector(self) -> nn.ModuleList:
        layers = []
        ch = self.last_fmap_ch // 4
        for i in range(3):
            layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(ch),
                    nn.Conv2d(ch, self.out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                )
            )
            ch *= 2
        return nn.ModuleList(layers)


class BackBoneNET(ResNet):
    def __init__(
        self, 
        input_channels: int, 
        block: Callable=BasicBlock, 
        block_layers: Iterable[int]=[2, 2, 2, 2], 
        pretrained_resnet: Optional[str]="resnet18"):

        super(BackBoneNET, self).__init__(block, block_layers)
        self.block = block
        self.block_layers = block_layers
        self.input_channels = input_channels
        self.pretrained_resnet = pretrained_resnet

        #init resnet weights
        if pretrained_resnet:
            self.load_state_dict(
                getattr(torchvision.models, pretrained_resnet)(weights="DEFAULT").state_dict())
        
        if self.input_channels != 3:
            self.rgb_channel_projector = nn.Conv2d(
                self.input_channels, 3, kernel_size=(3, 3), stride=1, padding=1)

        #delete unwanted layers
        del self.maxpool, self.fc, self.avgpool
        
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.input_channels != 3:
            x = self.rgb_channel_projector(x)
        fmap1 = self.conv1(x)
        _res = self.bn1(fmap1)
        _res = self.relu(_res)
        _res = self.layer1(_res)
        fmap1 = self.layer2(_res)
        fmap2 = self.layer3(fmap1)
        fmap3 = self.layer4(fmap2)
        
        return fmap1, fmap2, fmap3
    

class BBoxCompiler(nn.Module):
    def __init__(self, anchors: torch.Tensor, img_size: torch.Tensor, n_classes: int):
        super(BBoxCompiler, self).__init__()
        self.register_buffer("img_size", img_size)
        self.register_buffer("anchors", anchors * img_size, persistent=True)
        self.n_classes = n_classes
        self.n_anchors = len(self.anchors)
    
    def forward(self, feature_map: torch.Tensor):
        #1. feature_map size: (N, B*(5+C), h, w) --> (N, h, w, B*(5+C))
        #2. obj_score size: (N, h, w, n_anchor, cofidence)
        #3. pred_coordinates size: (N, h, w, n_anchor, bb_coord)
        #4. class_scores size: (N, h, w, n_anchor, class_scores)

        if self.anchors.device != feature_map.device:
            self.anchors = self.anchors.to(feature_map.device)

        if self.img_size.device != feature_map.device:
            self.img_size = self.img_size.to(feature_map.device)

        feature_map = torch.permute(feature_map, (0, 2, 3, 1))
        N, H, W, _ = feature_map.shape
        feature_map = feature_map.reshape(N, H, W, self.n_anchors, (5+self.n_classes))

        grid = self._make_grid(W, H)
        if grid.device != feature_map.device:
            grid = grid.to(feature_map.device)

        stride = self.img_size // torch.Tensor((W, H))       # downsample scale (width_stride, height_stride)
        confidence = feature_map[..., 0].unsqueeze(dim=-1)
        boxlocs = feature_map[..., 1:5]
        boxXY = (boxlocs[..., 0:2].sigmoid() + grid) * stride
        boxWH = torch.exp(boxlocs[..., 2:4]) * self.anchors
        boxXY = boxXY / self.img_size
        boxWH = boxWH / self.img_size                        # scale final output by original image dimensions
        confidence = confidence.sigmoid()
        class_scores = feature_map[..., 5:]
        bboxes = torch.cat((boxXY, boxWH), dim=-1)

        if len(class_scores) == 0:
            return (
                torch
                .cat((confidence, bboxes), dim=-1)
                .reshape(N, -1, 5)
            )
        
        class_scores = torch.sigmoid(class_scores)
        bboxes = (
            torch
            .cat((confidence, bboxes, class_scores), dim=-1)
            .reshape(N, -1, self.n_classes+5)
        )
        return bboxes
        
    def _make_grid(self, nx: int, ny: int):
        xindex = torch.arange(nx)
        yindex = torch.arange(ny)
        ygrid, xgrid = torch.meshgrid([yindex, xindex], indexing='ij')
        return torch.stack((xgrid, ygrid), 2).view((1, ny, nx, 1, 2)).float()


class GWDetectionNET(nn.Module):
    def __init__(
            self, 
            input_channels: int,
            img_size: Tuple[int, int],
            *,
            anchors_path: str,
            n_classes: int=0,
            block: Callable=BasicBlock, 
            block_layers: Iterable[int]=[2, 2, 2, 2], 
            pretrained_resnet_backbone: Optional[str]=None,
            last_fmap_ch: int=512):
        
        super(GWDetectionNET, self).__init__()
        
        H, W = img_size
        self.img_size = torch.Tensor((W, H))
        self._load_and_set_anchors(anchors_path)
        anchors = [self.sm_anchors, self.md_anchors, self.lg_anchors]
        n_anchors = len(anchors[0])
        self.backbone = BackBoneNET(input_channels, block, block_layers, pretrained_resnet_backbone)
        self.detect_net = DetectNET(n_anchors, n_classes, last_fmap_ch)
        ## bbox compiler for small, mid and large anchors
        self.bbox_compiler = nn.ModuleList([
            BBoxCompiler(anchors[i], self.img_size, n_classes) for i in range(3)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap1, fmap2, fmap3 = self.backbone(x)

        # detection maps for small, mid and large scale objects
        # each with shape of (N, B*(5+C), h, w)
        sm_dmap, md_dmap, lg_dmap = self.detect_net((fmap1, fmap2, fmap3))
        
        bboxes = []
        for i, dmap in enumerate([sm_dmap, md_dmap, lg_dmap]):
            bbox = self.bbox_compiler[i](dmap)
            bboxes.append(bbox)

        bboxes = torch.cat(bboxes, dim=1)
        return bboxes
    
    def _load_and_set_anchors(self, anchors_path: str):
        with open(anchors_path, "r") as f:
            anchors = json.load(f)
        f.close()

        self.sm_anchors = torch.Tensor(anchors["small anchors"])
        self.md_anchors = torch.Tensor(anchors["mid anchors"])
        self.lg_anchors = torch.Tensor(anchors["large anchors"])