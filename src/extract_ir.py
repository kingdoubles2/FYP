import argparse
import json
import sys
from pathlib import Path

from spec_parser.parser import parse_openapi

SPEC_EXTENSIONS = {".json", ".yaml", ".yml"}


def collect_spec_files(paths: list[str]) -> list[Path]:
    """Gather spec files from a mix of file paths and directories."""
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(
                f for f in sorted(p.iterdir()) if f.suffix.lower() in SPEC_EXTENSIONS
            )
        elif p.is_file():
            files.append(p)
        else:
            print(f"WARNING: skipping '{p}' (not found)", file=sys.stderr)
    return files


def process_spec(spec_path: Path, output_dir: Path | None, pretty: bool) -> bool:
    """Parse one spec file and write its IR JSON. Returns True on success."""
    try:
        spec_text = spec_path.read_text(encoding="utf-8")
        parsed = parse_openapi(spec_text)
    except Exception as exc:
        print(f"ERROR: failed to parse '{spec_path}': {exc}", file=sys.stderr)
        return False

    data = parsed.to_dict()
    indent = 2 if pretty else None

    dest_dir = output_dir if output_dir else spec_path.parent
    out_path = dest_dir / f"{spec_path.stem}_ir.json"

    out_path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
    print(f"  {spec_path} -> {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract IR JSON from OpenAPI 3 specs (files or folders).",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more spec files (.json/.yaml/.yml) or directories containing them",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument(
        "-o", "--output-dir",
        help="Directory to write IR files into (default: same directory as each spec)",
    )
    args = parser.parse_args()

    output_dir: Path | None = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_spec_files(args.paths)
    if not files:
        print("No spec files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} spec file(s):")
    ok = sum(process_spec(f, output_dir, args.pretty) for f in files)
    failed = len(files) - ok
    print(f"\nDone: {ok} succeeded, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()