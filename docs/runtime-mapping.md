# Runtime Mapping Injection

This project supports runtime path-parameter injection before live API execution.

## Why

Generated tests can include placeholder path values such as owner, repo, or username. In real environments, these values must come from runtime configuration.

## How it works

1. Generate tests into `build/tests`.
2. Run `src/test_runner/prepare_runtime_tests.py`.
3. The script injects values from mapping and environment variables.
4. Cases still missing required path values are removed from runnable suites and reported as skipped.
5. Runner executes only `build/runtime-tests`.

## Mapping file

Default path: `config/runtime_mapping.json`

Copy from example:

```bash
cp config/runtime_mapping.example.json config/runtime_mapping.json
```

The file supports three levels:

- `globals`: fallback values for all suites/tests
- `suites`: values scoped by `api_title` or suite file name
- `tests`: values scoped by `test_id`

## Environment override

Environment variables override mapping file values.

Prefix: `CONTRACTGUARD_MAP_`

Examples:

- `CONTRACTGUARD_MAP_OWNER=my-org`
- `CONTRACTGUARD_MAP_REPO=my-repo`
- `CONTRACTGUARD_MAP_USERNAME=octocat`

You can also provide the full mapping object as JSON using:

- `CONTRACTGUARD_RUNTIME_MAPPING_JSON`

In GitHub Actions this is typically stored as a repository secret and mapped to the workflow environment.

Priority order is:

1. `CONTRACTGUARD_MAP_*` variables
2. `CONTRACTGUARD_RUNTIME_MAPPING_JSON`
3. `config/runtime_mapping.json`

## Output artifacts

- Runtime-ready tests: `build/runtime-tests/*.json`
- Skip report: `build/results/runtime-skip-report.json`

## Skip reasons

- `MISSING_MAPPING`: required path placeholder could not be resolved
- `MISSING_INPUT_DATA`: test case structure is missing first-step input data
