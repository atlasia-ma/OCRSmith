# src/ocrsmith/core/OCRSmithEngine.py

import json
import os
import random

from PIL import Image

# Import your managers
from ocrsmith.config.schema import AppConfig
from ocrsmith.core.AugmentationManager import AugmentationManager
from ocrsmith.core.BackgroundManager import BackgroundManager
from ocrsmith.core.FontManager import FontManager
from ocrsmith.core.TextPlacementManager import TextPlacementManager
from ocrsmith.core.TextRenderingManager import TextRenderingManager
from ocrsmith.datasets import TextDataManager


def _worker_generate_and_save(args):
    """Top-level worker for multiprocessing. Reconstructs engine, seeds RNGs, generates, saves image, returns annotation dict.
    Args: (idx, conf_dict, text_value, output_dir)
    """
    idx, conf_dict, text_value, output_dir = args
    # Derive per-sample seed
    base_seed = conf_dict.get("seed")
    if base_seed is not None:
        conf_dict = dict(conf_dict)
        conf_dict["seed"] = int(base_seed) + int(idx)
    # Rebuild config and engine
    config_obj = AppConfig.model_validate(conf_dict)
    engine = OCRSmithEngine(config_obj)
    # Generate
    image, text, bbox, metadata = engine.generate_sample(text=text_value)
    # Save image in worker
    image_filename = f"sample_{idx:04d}.png"
    image_path = os.path.join(output_dir, image_filename)
    image.save(image_path)
    return {
        "image_path": image_path,
        "text": text,
        "bbox": bbox,
        "placement_metadata": metadata,
        "index": idx,
    }


class OCRSmithEngine:
    """Test engine using the new manager classes."""

    def __init__(self, config: AppConfig):
        self.config = config

        # Seed RNGs for reproducibility
        if getattr(self.config, "seed", None) is not None:
            try:
                import numpy as np

                np.random.seed(self.config.seed)
            except Exception:
                pass
            random.seed(self.config.seed)

        # Initialize managers
        self.background_manager = BackgroundManager(self.config)
        self.text_rendering_manager = TextRenderingManager(self.config)
        self.text_placement_manager = TextPlacementManager(self.config)
        self.augmentation_manager = AugmentationManager(self.config)
        self.font_manager = FontManager(self.config)
        self.text_data_manager = TextDataManager()

    def generate_sample(self, text: str, output_size: tuple = (800, 600)):
        """
        Generate a single OCR training sample.
        This method now requires text to be provided.
        """
        if not text:
            raise ValueError("Text must be provided to generate a sample.")

        # Get random font
        font = self.font_manager.get_random_font()

        # Step 1: Render text with random text rendering strategy
        text_context = self.text_rendering_manager.get_random_text_renderer()
        # Pass layout constraints to the renderer if available
        max_w = getattr(getattr(self.config, "layout", None), "max_width", None)
        max_h = getattr(getattr(self.config, "layout", None), "max_height", None)

        text_image, mask, (width, height) = text_context.render_text(
            font, text, max_width=max_w, max_height=max_h
        )

        # Step 2: Create background with random background strategy
        bg_context = self.background_manager.get_random_background()
        # Add some padding to the background size
        background_image = bg_context.render(width + 100, height + 100)

        # Ensure both images are in RGBA mode for proper pasting
        if text_image.mode != "RGBA":
            text_image = text_image.convert("RGBA")
        if background_image.mode != "RGBA":
            background_image = background_image.convert("RGBA")

        # Step 3: Place text on background with random placement strategy
        placement_context = self.text_placement_manager.get_random_placement()
        placement_result = placement_context.place_text(text_image, background_image)

        composed_image = placement_result.composed_image

        # Step 4: Apply augmentation pipeline (if configured), track applied params
        applied_augmentations = []
        if self.config.augmentations:
            try:
                # Reproduce pipeline with parameter capture
                pipeline = [
                    cfg for cfg in self.augmentation_manager.augmentation_configs if cfg.get("enabled", True)
                ]
                if getattr(self.config, "augmentation_order", "random") == "random":
                    random.shuffle(pipeline)
                out_img = composed_image
                for cfg in pipeline:
                    prob = cfg.get("probability", 1.0)
                    if random.random() > prob:
                        continue
                    params = dict(cfg["params"])
                    # Sample ranges
                    for key, val in list(params.items()):
                        if isinstance(val, (list, tuple)):
                            lo, hi = (
                                (float(val[0]), float(val[1]))
                                if len(val) == 2
                                else (float(val[0]), float(val[0]))
                            )
                            if hi < lo:
                                lo, hi = hi, lo
                            params[key] = random.uniform(lo, hi)
                    strategy = cfg["strategy_class"](**params)
                    out_img = strategy.apply(out_img)
                    applied_augmentations.append({"type": cfg["type"], "params": params, "probability": prob})
                composed_image = out_img
            except Exception as e:
                print(f"Augmentation pipeline failed: {e}, skipping...")

        # Convert back to RGB for saving, handling transparency
        if composed_image.mode == "RGBA":
            # Create a white background and composite the image onto it
            final_image = Image.new("RGB", composed_image.size, (255, 255, 255))
            final_image.paste(composed_image, mask=composed_image.split()[-1])
        else:
            final_image = composed_image

        # Enrich metadata
        metadata = dict(placement_result.metadata)
        metadata.update(
            {
                "seed": getattr(self.config, "seed", None),
                "font_size": getattr(font, "size", None),
                "renderer": type(text_context._strategy).__name__,
                "background": type(bg_context._strategy).__name__,
                "augmentations": applied_augmentations,
            }
        )
        # Include trimmed label text if renderer provided it
        label_text = getattr(getattr(text_context, "_strategy", None), "last_rendered_text", None)
        if label_text:
            metadata["text"] = label_text

        return final_image, text, placement_result.bbox, metadata

    def generate_test_dataset(self, num_samples: int = 10, output_dir: str = "test_output", workers: int = 1):

        if not self.config.text_data:
            raise ValueError("`text_data` configuration is missing from the config file.")

        text_config = self.config.text_data.model_dump()
        source_type = text_config.pop("source_type")
        source_path = text_config.pop("source_path")

        # Load texts (TextDataManager.load_from_source may return texts or set current_loader)
        self.text_data_manager.load_from_source(
            source_type=source_type,
            source_path=source_path,
            **text_config,  # Pass the rest of the config as kwargs
        )

        os.makedirs(output_dir, exist_ok=True)

        annotations = []

        texts = []
        for _ in range(num_samples):
            t = self.text_data_manager.get_random_text()
            if t:
                texts.append(t)
        tasks = [(i, self.config.model_dump(), texts[i], output_dir) for i in range(len(texts))]

        if workers and workers > 1:
            try:
                from concurrent.futures import ProcessPoolExecutor, as_completed

                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(_worker_generate_and_save, t) for t in tasks]
                    for fut in as_completed(futures):
                        try:
                            ann = fut.result()
                            annotations.append({k: v for k, v in ann.items() if k != "index"})
                            print(f"Generated sample {ann['index'] + 1}/{num_samples}")
                        except Exception as e:
                            print(f"Worker failed: {e}")
            except Exception as e:
                print(f"Falling back to single-process due to error: {e}")
                workers = 1

        if not workers or workers == 1:
            for i, text_value in enumerate(texts):
                try:
                    image, text, bbox, metadata = self.generate_sample(text=text_value)
                    image_filename = f"sample_{i:04d}.png"
                    image_path = os.path.join(output_dir, image_filename)
                    image.save(image_path)
                    annotations.append(
                        {"image_path": image_path, "text": text, "bbox": bbox, "placement_metadata": metadata}
                    )
                    print(f"Generated sample {i + 1}/{num_samples}")
                except Exception as e:
                    print(f"Error generating sample {i + 1}: {e}")
                    continue

        # Save annotations
        annotations_path = os.path.join(output_dir, "annotations.json")
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)

        print(f"\nGenerated {len(annotations)} samples successfully!")
        print(f"Images saved to: {output_dir}")
        print(f"Annotations saved to: {annotations_path}")

        return annotations
