import argparse
from ocrsmith.config.loader import load_config
from ocrsmith.core.OCRSmithEngine import OCRSmithEngine


def parse_overrides(pairs):
    """Parse CLI key=value pairs into a flat dict. Supports simple keys like text_data.source_type=csv"""
    overrides = {}
    for pair in pairs or []:
        if '=' not in pair:
            continue
        key, value = pair.split('=', 1)
        overrides[key.strip()] = value.strip()
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Generate OCR dataset")
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--set', dest='overrides', action='append', help='Override config, e.g. text_data.source_path=assets/texts.csv', default=[])
    parser.add_argument('--set-json', dest='overrides_json', type=str, default=None, help='JSON overrides for config')
    parser.add_argument('--seed', type=int, default=None, help='Random seed to use')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes for generation')
    args = parser.parse_args()

    import json as _json
    config = load_config(args.config)

    # Apply simple dot-notation overrides for common fields
    for k, v in parse_overrides(args.overrides).items():
        try:
            # Minimal support: update output paths and text_data
            if k == 'output.images_dir':
                config.output.images_dir = v
            elif k == 'output.metadata_file':
                config.output.metadata_file = v
            elif k == 'text_data.source_type' and config.text_data:
                config.text_data.source_type = v  # type: ignore
            elif k == 'text_data.source_path' and config.text_data:
                config.text_data.source_path = v  # type: ignore
            elif k == 'text_data.text_column' and config.text_data:
                config.text_data.text_column = v  # type: ignore
            elif k == 'seed':
                config.seed = int(v)
        except Exception:
            pass

    if args.overrides_json:
        try:
            obj = _json.loads(args.overrides_json)
            # Minimal deep-merge for text_data and output
            if 'output' in obj:
                if 'images_dir' in obj['output']:
                    config.output.images_dir = obj['output']['images_dir']
                if 'metadata_file' in obj['output']:
                    config.output.metadata_file = obj['output']['metadata_file']
            if 'text_data' in obj and config.text_data:
                for k, v in obj['text_data'].items():
                    setattr(config.text_data, k, v)
        except Exception:
            pass

    if args.seed is not None:
        config.seed = args.seed

    engine = OCRSmithEngine(config)
    engine.generate_test_dataset(num_samples=args.num_samples, output_dir=args.output_dir, workers=args.workers)


if __name__ == '__main__':
    main()