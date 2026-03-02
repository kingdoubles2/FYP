import argparse
import json
import sys
from pathlib import Path

from test_generator import generate_test_cases


def collect_ir_files(paths: list[str]) -> list[Path]:
    """Gather IR JSON files from a mix of file paths and directories (recursive)."""
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(
                f for f in sorted(p.rglob("*_ir.json"))
                if f.is_file()
            )
        elif p.is_file():
            files.append(p)
        else:
            print(f"WARNING: skipping '{p}' (not found)", file=sys.stderr)
    return files


def process_ir(
    ir_path: Path, output_dir: Path | None, base_dir: Path | None, pretty: bool
) -> bool:
    """Read one IR file, generate test cases, write output JSON."""
    try:
        ir_data = json.loads(ir_path.read_text(encoding="utf-8"))
        suite = generate_test_cases(ir_data)
    except Exception as exc:
        print(f"ERROR: failed to process '{ir_path}': {exc}", file=sys.stderr)
        return False

    data = suite.to_dict()
    indent = 2 if pretty else None

    if output_dir and base_dir:
        rel = ir_path.parent.relative_to(base_dir)
        dest_dir = output_dir / rel
    elif output_dir:
        dest_dir = output_dir
    else:
        dest_dir = ir_path.parent

    dest_dir.mkdir(parents=True, exist_ok=True)

    # petstore_ir.json -> petstore_tests.json
    stem = ir_path.stem
    if stem.endswith("_ir"):
        stem = stem[:-3]
    out_path = dest_dir / f"{stem}_tests.json"

    out_path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
    print(f"  {ir_path} -> {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate test cases from IR JSON files (files or folders).",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more IR JSON files (*_ir.json) or directories containing them",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument(
        "-o", "--output-dir",
        help="Directory to write test case files into (default: same directory as each IR file)",
    )
    args = parser.parse_args()

    output_dir: Path | None = None
    base_dir: Path | None = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(args.paths) == 1 and Path(args.paths[0]).is_dir():
            base_dir = Path(args.paths[0])

    files = collect_ir_files(args.paths)
    if not files:
        print("No IR files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} IR file(s):")
    ok = sum(process_ir(f, output_dir, base_dir, args.pretty) for f in files)
    failed = len(files) - ok
    print(f"\nDone: {ok} succeeded, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
