from spec_parser.parser import parse_openapi


if __name__ == "__main__":
    with open("petstore.json", "r") as f:
        spec_text = f.read()

    parsed = parse_openapi(spec_text)

    print("Title:", parsed.title)
    print("Version:", parsed.version)
    print("Total Endpoints:", len(parsed.endpoints))

    first = parsed.endpoints[0]
    print("Method:", first.method)
    print("Path:", first.path)
    print("Operation ID:", first.operation_id)
    print("Request Schema:", first.request_schema)
    print("Response Schemas:", first.response_schemas.keys())