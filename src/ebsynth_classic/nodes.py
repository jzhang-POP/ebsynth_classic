"""ComfyUI nodes for ebsynth classic."""

import torch
import numpy as np

from .ebsynth_wrapper import get_wrapper, EbSynthError


class EbSynthGuide:
    """Create a single guide pair for ebsynth."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "target": ("IMAGE",),
            },
            "optional": {
                "weight": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.1,
                    },
                ),
                "previous_guides": ("EBSYNTH_GUIDES",),
            },
        }

    RETURN_TYPES = ("EBSYNTH_GUIDES",)
    RETURN_NAMES = ("guides",)
    FUNCTION = "create_guide"
    CATEGORY = "image/ebsynth"

    def create_guide(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        weight: float = 1.0,
        previous_guides: list | None = None,
    ) -> tuple[list]:
        """Create a guide and optionally chain with previous guides."""

        guide = {
            "source": source[0].cpu().numpy(),
            "target": target[0].cpu().numpy(),
            "weight": weight,
        }

        if previous_guides is None:
            guides = [guide]
        else:
            guides = previous_guides + [guide]

        return (guides,)


class EbSynthTransfer:
    """Transfer style using guide images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_image": ("IMAGE",),
                "guides": ("EBSYNTH_GUIDES",),
            },
            "optional": {
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
        guides: list,
        uniformity: float = 3500.0,
        patchsize: int = 5,
        pyramidlevels: int = 6,
        searchvoteiters: int = 12,
        patchmatchiters: int = 6,
        extrapass3x3: bool = False,
    ) -> tuple[torch.Tensor]:
        """Run ebsynth style transfer."""

        style_np = style_image[0].cpu().numpy()

        # Convert guide dicts to tuples
        guide_tuples = [(g["source"], g["target"], g["weight"]) for g in guides]

        # Ensure patchsize is odd
        if patchsize % 2 == 0:
            patchsize += 1

        wrapper = get_wrapper()

        result = wrapper.run(
            style=style_np,
            guides=guide_tuples,
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


class EbSynthTransferSimple:
    """Simple single-guide transfer (convenience node)."""

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
        """Run ebsynth style transfer with a single guide."""

        style_np = style_image[0].cpu().numpy()
        src_np = source_guide[0].cpu().numpy()
        tgt_np = target_guide[0].cpu().numpy()

        if patchsize % 2 == 0:
            patchsize += 1

        wrapper = get_wrapper()

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
                "target_guides": ("IMAGE",),
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

        result_tensor = torch.from_numpy(np.stack(results, axis=0))

        return (result_tensor,)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "EbSynthGuide": EbSynthGuide,
    "EbSynthTransfer": EbSynthTransfer,
    "EbSynthTransferSimple": EbSynthTransferSimple,
    "EbSynthTransferBatch": EbSynthTransferBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EbSynthGuide": "EbSynth Guide",
    "EbSynthTransfer": "EbSynth Transfer",
    "EbSynthTransferSimple": "EbSynth Transfer (Simple)",
    "EbSynthTransferBatch": "EbSynth Transfer (Batch)",
}
