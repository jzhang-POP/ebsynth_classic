"""ComfyUI nodes for ebsynth classic."""

import torch
import numpy as np

from .ebsynth_wrapper import get_wrapper, EbSynthError


class EbSynthTransfer:
    """Transfer style using guide images (texture-by-numbers)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_image": ("IMAGE",),
                "source_guide": ("IMAGE",),
                "target_guide": ("IMAGE",),
            },
            "optional": {
                "guide_weight": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.1,
                    },
                ),
                "uniformity": (
                    "FLOAT",
                    {
                        "default": 3500.0,
                        "min": 0.0,
                        "max": 50000.0,
                        "step": 100.0,
                        "tooltip": "Higher = smoother/more uniform results",
                    },
                ),
                "patchsize": (
                    "INT",
                    {
                        "default": 5,
                        "min": 3,
                        "max": 99,
                        "step": 2,
                        "tooltip": "Patch size (odd numbers only)",
                    },
                ),
                "pyramidlevels": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 12,
                    },
                ),
                "searchvoteiters": (
                    "INT",
                    {
                        "default": 12,
                        "min": 1,
                        "max": 30,
                    },
                ),
                "patchmatchiters": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 20,
                    },
                ),
                "extrapass3x3": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Extra refinement pass for finer details",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transfer"
    CATEGORY = "image/ebsynth"

    def transfer(
        self,
        style_image: torch.Tensor,
        source_guide: torch.Tensor,
        target_guide: torch.Tensor,
        guide_weight: float = 1.0,
        uniformity: float = 3500.0,
        patchsize: int = 5,
        pyramidlevels: int = 6,
        searchvoteiters: int = 12,
        patchmatchiters: int = 6,
        extrapass3x3: bool = False,
    ) -> tuple[torch.Tensor]:
        """Run ebsynth style transfer."""

        # Convert from ComfyUI format (B, H, W, C) to numpy
        style_np = style_image[0].cpu().numpy()
        src_np = source_guide[0].cpu().numpy()
        tgt_np = target_guide[0].cpu().numpy()

        # Ensure patchsize is odd
        if patchsize % 2 == 0:
            patchsize += 1

        wrapper = get_wrapper()

        print(f"style = {style_np}")
        print(f"guide = {[(src_np, tgt_np, guide_weight)]}")
        print(f"uniformity = {uniformity}")
        print(f"patchsize = {patchsize}")
        print(f"pyramidlevels = {pyramidlevels}")
        print(f"searchvoteiters = {searchvoteiters}")
        print(f"extrapass3x3 = {extrapass3x3}")
        print("backend = cpu")

        result = wrapper.run(
            style=style_np,
            guides=[(src_np, tgt_np, guide_weight)],
            uniformity=uniformity,
            patchsize=patchsize,
            pyramidlevels=pyramidlevels,
            searchvoteiters=searchvoteiters,
            patchmatchiters=patchmatchiters,
            extrapass3x3=extrapass3x3,
            backend="cpu",  # Mac only supports CPU
        )

        # Convert back to ComfyUI format
        result_tensor = torch.from_numpy(result).unsqueeze(0)

        return (result_tensor,)


class EbSynthTransferBatch:
    """Transfer style to a batch of target frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_image": ("IMAGE",),
                "source_guide": ("IMAGE",),
                "target_guides": ("IMAGE",),  # Batch of targets
            },
            "optional": {
                "guide_weight": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.1,
                    },
                ),
                "uniformity": (
                    "FLOAT",
                    {
                        "default": 3500.0,
                        "min": 0.0,
                        "max": 50000.0,
                        "step": 100.0,
                    },
                ),
                "patchsize": (
                    "INT",
                    {
                        "default": 5,
                        "min": 3,
                        "max": 99,
                        "step": 2,
                    },
                ),
                "pyramidlevels": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 12,
                    },
                ),
                "searchvoteiters": (
                    "INT",
                    {
                        "default": 12,
                        "min": 1,
                        "max": 30,
                    },
                ),
                "patchmatchiters": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 20,
                    },
                ),
                "extrapass3x3": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transfer_batch"
    CATEGORY = "image/ebsynth"

    def transfer_batch(
        self,
        style_image: torch.Tensor,
        source_guide: torch.Tensor,
        target_guides: torch.Tensor,
        guide_weight: float = 1.0,
        uniformity: float = 3500.0,
        patchsize: int = 5,
        pyramidlevels: int = 6,
        searchvoteiters: int = 12,
        patchmatchiters: int = 6,
        extrapass3x3: bool = False,
    ) -> tuple[torch.Tensor]:
        """Run ebsynth on each target frame."""

        style_np = style_image[0].cpu().numpy()
        src_np = source_guide[0].cpu().numpy()

        if patchsize % 2 == 0:
            patchsize += 1

        wrapper = get_wrapper()
        results = []

        batch_size = target_guides.shape[0]
        for i in range(batch_size):
            tgt_np = target_guides[i].cpu().numpy()

            result = wrapper.run(
                style=style_np,
                guides=[(src_np, tgt_np, guide_weight)],
                uniformity=uniformity,
                patchsize=patchsize,
                pyramidlevels=pyramidlevels,
                searchvoteiters=searchvoteiters,
                patchmatchiters=patchmatchiters,
                extrapass3x3=extrapass3x3,
                backend="cpu",
            )
            results.append(result)

        # Stack results into batch
        result_tensor = torch.from_numpy(np.stack(results, axis=0))

        return (result_tensor,)


class EbSynthMultiGuide:
    """Transfer style with multiple guide channels."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_image": ("IMAGE",),
                "source_guide_1": ("IMAGE",),
                "target_guide_1": ("IMAGE",),
            },
            "optional": {
                "weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "source_guide_2": ("IMAGE",),
                "target_guide_2": ("IMAGE",),
                "weight_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "source_guide_3": ("IMAGE",),
                "target_guide_3": ("IMAGE",),
                "weight_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "uniformity": ("FLOAT", {"default": 3500.0, "min": 0.0, "max": 50000.0, "step": 100.0}),
                "patchsize": ("INT", {"default": 5, "min": 3, "max": 99, "step": 2}),
                "pyramidlevels": ("INT", {"default": 6, "min": 1, "max": 12}),
                "searchvoteiters": ("INT", {"default": 12, "min": 1, "max": 30}),
                "patchmatchiters": ("INT", {"default": 6, "min": 1, "max": 20}),
                "extrapass3x3": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transfer"
    CATEGORY = "image/ebsynth"

    def transfer(
        self,
        style_image: torch.Tensor,
        source_guide_1: torch.Tensor,
        target_guide_1: torch.Tensor,
        weight_1: float = 1.0,
        source_guide_2: torch.Tensor | None = None,
        target_guide_2: torch.Tensor | None = None,
        weight_2: float = 1.0,
        source_guide_3: torch.Tensor | None = None,
        target_guide_3: torch.Tensor | None = None,
        weight_3: float = 1.0,
        uniformity: float = 3500.0,
        patchsize: int = 5,
        pyramidlevels: int = 6,
        searchvoteiters: int = 12,
        patchmatchiters: int = 6,
        extrapass3x3: bool = False,
    ) -> tuple[torch.Tensor]:
        """Run ebsynth with multiple guides."""

        style_np = style_image[0].cpu().numpy()

        # Build guide list
        guides = [(source_guide_1[0].cpu().numpy(), target_guide_1[0].cpu().numpy(), weight_1)]

        if source_guide_2 is not None and target_guide_2 is not None:
            guides.append((source_guide_2[0].cpu().numpy(), target_guide_2[0].cpu().numpy(), weight_2))

        if source_guide_3 is not None and target_guide_3 is not None:
            guides.append((source_guide_3[0].cpu().numpy(), target_guide_3[0].cpu().numpy(), weight_3))

        if patchsize % 2 == 0:
            patchsize += 1

        wrapper = get_wrapper()

        result = wrapper.run(
            style=style_np,
            guides=guides,
            uniformity=uniformity,
            patchsize=patchsize,
            pyramidlevels=pyramidlevels,
            searchvoteiters=searchvoteiters,
            patchmatchiters=patchmatchiters,
            extrapass3x3=extrapass3x3,
            backend="cpu",
        )

        result_tensor = torch.from_numpy(result).unsqueeze(0)

        return (result_tensor,)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "EbSynthTransfer": EbSynthTransfer,
    "EbSynthTransferBatch": EbSynthTransferBatch,
    "EbSynthMultiGuide": EbSynthMultiGuide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EbSynthTransfer": "EbSynth Transfer",
    "EbSynthTransferBatch": "EbSynth Transfer (Batch)",
    "EbSynthMultiGuide": "EbSynth Multi-Guide",
}
