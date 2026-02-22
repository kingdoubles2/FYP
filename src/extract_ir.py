import argparse
import json
from pathlib import Path

from spec_parser.parser import parse_openapi


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract IR JSON from an OpenAPI 3 spec.")
    parser.add_argument("spec_path", help="Path to OpenAPI file (.json/.yaml/.yml)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    spec_path = Path(args.spec_path)
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    spec_text = spec_path.read_text(encoding="utf-8")
    parsed = parse_openapi(spec_text)

    data = parsed.to_dict()
    if args.pretty:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    main()