"""Wrapper for the ebsynth command-line tool."""

import subprocess
import tempfile
import platform
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


class EbSynthError(Exception):
    """Raised when ebsynth fails."""

    pass


class EbSynthWrapper:
    def __init__(self, bin_path: str | None = None):
        self.bin_path = bin_path or self._find_binary()

    def _find_binary(self) -> str:
        """Locate the ebsynth binary."""
        # Check in the bin/ folder relative to this file
        module_dir = Path(__file__).parent.parent.parent  # up to ebsynth_classic/
        system = platform.system().lower()

        if system == "windows":
            names = ["ebsynth.exe", "EbSynth.exe"]
        else:
            names = ["ebsynth"]

        # Check local bin/ folder
        for name in names:
            candidate = module_dir / "bin" / name
            if candidate.exists():
                return str(candidate)

        # Check PATH
        found = shutil.which("ebsynth")
        if found:
            return found

        raise FileNotFoundError(
            f"ebsynth binary not found.\n"
            f"Please either:\n"
            f"  1. Place it in: {module_dir / 'bin'}\n"
            f"  2. Add it to your PATH\n"
            f"Build instructions: https://github.com/jamriska/ebsynth"
        )

    def run(
        self,
        style: np.ndarray,
        guides: list[tuple[np.ndarray, np.ndarray, float]],
        uniformity: float = 3500.0,
        patchsize: int = 5,
        pyramidlevels: int = 6,
        searchvoteiters: int = 12,
        patchmatchiters: int = 6,
        extrapass3x3: bool = False,
        backend: str = "cpu",
    ) -> np.ndarray:
        """
        Run ebsynth synthesis.

        Args:
            style: Style image (H, W, C), float32 [0,1] or uint8 [0,255]
            guides: List of (source_guide, target_guide, weight) tuples
            uniformity: Smoothness weight (500-15000 typical)
            patchsize: Patch size, odd number >= 3
            pyramidlevels: Number of pyramid levels
            searchvoteiters: Search vote iterations
            patchmatchiters: PatchMatch iterations
            extrapass3x3: Extra refinement pass
            backend: "cpu" or "cuda"

        Returns:
            Synthesized image (H, W, C), float32 [0,1]
        """
        if not guides:
            raise ValueError("At least one guide pair is required")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save style
            style_path = tmpdir / "style.png"
            self._save_image(style, style_path)

            # Build command
            cmd = [self.bin_path, "-style", str(style_path)]

            # Add guides
            for i, (src, tgt, weight) in enumerate(guides):
                src_path = tmpdir / f"src_{i}.png"
                tgt_path = tmpdir / f"tgt_{i}.png"
                self._save_image(src, src_path)
                self._save_image(tgt, tgt_path)

                cmd.extend(["-guide", str(src_path), str(tgt_path)])
                if weight != 1.0:
                    cmd.extend(["-weight", str(weight)])

            # Output
            output_path = tmpdir / "output.png"
            cmd.extend(["-output", str(output_path)])

            # Parameters
            cmd.extend(
                [
                    "-uniformity",
                    str(uniformity),
                    "-patchsize",
                    str(int(patchsize)),
                    "-pyramidlevels",
                    str(int(pyramidlevels)),
                    "-searchvoteiters",
                    str(int(searchvoteiters)),
                    "-patchmatchiters",
                    str(int(patchmatchiters)),
                    "-backend",
                    backend,
                ]
            )

            if extrapass3x3:
                cmd.append("-extrapass3x3")

            # Run
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise EbSynthError(f"ebsynth failed (code {result.returncode}):\n{result.stderr or result.stdout}")

            if not output_path.exists():
                raise EbSynthError(f"ebsynth produced no output:\n{result.stdout}\n{result.stderr}")

            return self._load_image(output_path)

    def _save_image(self, img: np.ndarray, path: Path) -> None:
        """Save numpy array as PNG."""
        if img.dtype in (np.float32, np.float64):
            img = (img * 255).clip(0, 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # Handle grayscale
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.concatenate([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]  # Drop alpha

        Image.fromarray(img).save(path, "PNG")

    def _load_image(self, path: Path) -> np.ndarray:
        """Load PNG as float32 [0,1]."""
        img = np.array(Image.open(path).convert("RGB"))
        return img.astype(np.float32) / 255.0


# Singleton
_wrapper: EbSynthWrapper | None = None


def get_wrapper(bin_path: str | None = None) -> EbSynthWrapper:
    global _wrapper
    if _wrapper is None or bin_path is not None:
        _wrapper = EbSynthWrapper(bin_path)
    return _wrapper
