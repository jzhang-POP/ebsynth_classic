class EbSynthTransfer:
    """Basic ebsynth style transfer node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_image": ("IMAGE",),
                "source_guide": ("IMAGE",),
                "target_guide": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transfer"
    CATEGORY = "image/ebsynth"

    def transfer(self, style_image, source_guide, target_guide):
        # For now, just return the style image as a test
        return (style_image,)


NODE_CLASS_MAPPINGS = {
    "EbSynthTransfer": EbSynthTransfer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EbSynthTransfer": "EbSynth Transfer (Classic)",
}
