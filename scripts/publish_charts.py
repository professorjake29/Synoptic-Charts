"""Validate exactly 27 new PNGs and stage them to replace repository-root charts.

Usage: python scripts/publish_charts.py --output output
Exits nonzero before modifying repository charts if files are missing/broken.
The caller is responsible for git add/commit/push after successful validation.
"""
import argparse
from pathlib import Path
import shutil
import sys

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output.resolve()
    manifest = root / 'expected_chart_filenames.txt'
    expected = [name.strip() for name in manifest.read_text().splitlines() if name.strip()]
    if len(expected) != 27 or len(set(expected)) != 27:
        raise RuntimeError('Expected manifest must contain exactly 27 unique PNG filenames')
    if any(Path(name).name != name or not name.endswith('.png') for name in expected):
        raise RuntimeError('Manifest includes an invalid filename')

    actual = {p.name for p in output_dir.glob('*.png') if p.is_file()}
    missing = set(expected) - actual
    unexpected = actual - set(expected)
    if missing or unexpected:
        raise RuntimeError(f'Chart set mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}')

    # Validate EVERY image BEFORE replacing a single existing repository file.
    for name in expected:
        path = output_dir / name
        if path.stat().st_size < 1024:
            raise RuntimeError(f'PNG is unexpectedly small: {name}')
        try:
            with Image.open(path) as im:
                if im.format != 'PNG' or im.width < 100 or im.height < 100:
                    raise ValueError('Unexpected PNG format or dimensions')
                im.verify()
        except Exception as exc:
            raise RuntimeError(f'PNG failed image validation: {name}: {exc}') from exc
        print(f'Validated {name} ({path.stat().st_size:,} bytes)', flush=True)

    print('All 27 charts validated. Copying to repository root...', flush=True)
    for name in expected:
        shutil.copy2(output_dir / name, root / name)
    print('All 27 repository chart paths updated locally; ready for git commit.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
