import math
import os
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution matching Chollet's Xception.
    Combines a depthwise 3x3 conv with a pointwise 1x1 conv.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    """
    Residual block of Xception with repeated separable convolutions.
    """
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        reps: int,
        strides: int = 1,
        start_with_relu: bool = True,
        grow_first: bool = True,
    ):
        super().__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(kernel_size=3, stride=strides, padding=1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skipbn(self.skip(inp))
        else:
            skip = inp
        return x + skip


class XceptionBackbone(nn.Module):
    """
    Modified Xception backbone tailored for F3-Net.
    
    Exposes partitioned intermediate stages to facilitate cross-branch MixBlock fusion:
      - Stage 1 (Entry Flow): conv1, conv2, blocks 1..3  -> [N, 728, 19, 19]
      - Stage 2 (Middle Flow Part 1): blocks 4..7       -> [N, 728, 19, 19] (MixBlock 1 insertion point)
      - Stage 3 (Middle Flow Part 2): blocks 8..11      -> [N, 728, 19, 19]
      - Stage 4 (Exit Flow): block 12, conv3, conv4     -> [N, 2048, 10, 10]
      - Pooling: AdaptiveAvgPool2d                      -> [N, 2048]
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        is_lfs_branch: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.is_lfs_branch = is_lfs_branch

        # Initial convolutions
        # For LFS branch, input is already 149x149; stride 1 with padding 1 maintains 149x149.
        # For FAD/RGB branch, input is 299x299; stride 2 with padding 0 produces 149x149.
        if is_lfs_branch:
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=0, bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Entry Flow Blocks
        self.block1 = XceptionBlock(64, 128, reps=2, strides=2, start_with_relu=False, grow_first=True)
        self.block2 = XceptionBlock(128, 256, reps=2, strides=2, start_with_relu=True, grow_first=True)
        self.block3 = XceptionBlock(256, 728, reps=2, strides=2, start_with_relu=True, grow_first=True)

        # Middle Flow Blocks (Part 1: Blocks 4 to 7)
        self.block4 = XceptionBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block5 = XceptionBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block6 = XceptionBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block7 = XceptionBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)

        # Middle Flow Blocks (Part 2: Blocks 8 to 11)
        self.block8 = XceptionBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block9 = XceptionBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block10 = XceptionBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block11 = XceptionBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)

        # Exit Flow Block & Convolutions
        self.block12 = XceptionBlock(728, 1024, reps=2, strides=2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(2048)

    def forward_stage_0_7(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes through Entry Flow and Middle Flow Blocks 4..7.
        Input: (N, C, 299, 299) or (N, C, 149, 149)
        Output: (N, 728, 19, 19)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        return x

    def forward_stage_8_12(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes through Middle Flow Blocks 8..11, Exit Block 12, and Convs 3 & 4.
        Input: (N, 728, 19, 19)
        Output: (N, 2048, 10, 10)
        """
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        x = self.block12(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        return x

    def pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Global pooling to 2048-dim embedding.
        Input: (N, 2048, 10, 10)
        Output: (N, 2048)
        """
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full backbone forward pass without intermediate fusion."""
        fea_mid = self.forward_stage_0_7(x)
        fea_exit = self.forward_stage_8_12(fea_mid)
        return self.pool_features(fea_exit)

    def load_imagenet_state_dict(
        self,
        state_dict_or_path: Union[str, os.PathLike, Dict[str, torch.Tensor]],
    ) -> Dict[str, int]:
        """
        Loads ImageNet-pretrained Xception weights, adapting conv1 for multi-channel input.
        
        Args:
            state_dict_or_path: Filepath (str/PathLike) or pre-loaded state_dict.
            
        Returns:
            Dictionary summarizing loaded, missing, and unexpected parameter keys.
            
        Raises:
            FileNotFoundError: If state_dict_or_path is a path and the file does not exist.
            ValueError: If state_dict is not a dictionary or contains incompatible architecture weights.
        """
        if isinstance(state_dict_or_path, (str, os.PathLike)):
            path_str = os.fspath(state_dict_or_path)
            if not os.path.isfile(path_str):
                raise FileNotFoundError(f"Pretrained Xception checkpoint not found at: {path_str}")
            try:
                raw_dict = torch.load(path_str, map_location="cpu", weights_only=True)
            except Exception:
                raw_dict = torch.load(path_str, map_location="cpu")
        elif isinstance(state_dict_or_path, dict):
            raw_dict = state_dict_or_path
        else:
            raise ValueError(
                f"Expected weights to be a filepath (str/PathLike) or state_dict (dict), got {type(state_dict_or_path).__name__}"
            )

        if not isinstance(raw_dict, dict):
            raise ValueError(f"Loaded checkpoint is not a dictionary. Got type: {type(raw_dict).__name__}")

        # Unwrap common model state dict wrappers
        for wrapper_key in ("state_dict", "model_state_dict", "model"):
            if wrapper_key in raw_dict and isinstance(raw_dict[wrapper_key], dict):
                raw_dict = raw_dict[wrapper_key]
                break

        # Strip prefixes (e.g., 'module.', 'backbone.')
        unwrapped = {}
        for k, v in raw_dict.items():
            clean_k = k
            if clean_k.startswith("module."):
                clean_k = clean_k[7:]
            if clean_k.startswith("backbone."):
                clean_k = clean_k[9:]
            unwrapped[clean_k] = v

        # Validate architecture compatibility: conv1.weight is the foundational entry conv
        if "conv1.weight" not in unwrapped:
            raise ValueError("Incompatible checkpoint: missing required Xception backbone key 'conv1.weight'.")

        conv1_tensor = unwrapped["conv1.weight"]
        if not isinstance(conv1_tensor, torch.Tensor) or conv1_tensor.ndim != 4 or conv1_tensor.shape[0] != 32:
            raise ValueError(
                f"Incompatible checkpoint: 'conv1.weight' must be a 4D tensor with 32 output channels, "
                f"got shape {getattr(conv1_tensor, 'shape', None)}"
            )

        adapted_dict = {}
        for key, val in unwrapped.items():
            if isinstance(val, torch.Tensor):
                if "pointwise" in key and val.ndim == 2:
                    val = val.unsqueeze(-1).unsqueeze(-1)
                adapted_dict[key] = val
            else:
                adapted_dict[key] = val

        # Handle conv1 input channel mismatch (e.g., 3 -> 12 for FAD, 3 -> 6 for LFS)
        if adapted_dict["conv1.weight"].shape[1] != self.in_channels:
            orig_w = adapted_dict["conv1.weight"]
            orig_channels = orig_w.shape[1]
            scale = float(orig_channels) / float(self.in_channels)
            repeats = math.ceil(self.in_channels / orig_channels)
            expanded_w = orig_w.repeat(1, repeats, 1, 1)[:, :self.in_channels, :, :] * scale
            adapted_dict["conv1.weight"] = expanded_w

        # Filter out incompatible classifier weights (fc, classifier, last_linear)
        clean_dict = {
            k: v for k, v in adapted_dict.items()
            if not (k.startswith("fc") or k.startswith("classifier") or k.startswith("last_linear"))
        }

        # Validate that at least one model key matches
        model_keys = set(self.state_dict().keys())
        matched_keys = model_keys.intersection(clean_dict.keys())
        if not matched_keys:
            raise ValueError("Incompatible checkpoint: no parameter keys matched the Xception backbone architecture.")

        incompat = self.load_state_dict(clean_dict, strict=False)
        return {
            "matched_keys": len(matched_keys),
            "missing_keys": len(incompat.missing_keys),
            "unexpected_keys": len(incompat.unexpected_keys),
        }
