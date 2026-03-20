# CA400 template repo

This is a template for CA400 projects.

## 1. Create your repo

One person from your project team should fork this repo, then add other teammates as project members on GitLab.

## 2. Name your repo appropriately

The name of your project must be of the form `2024-ca400-XXXXXXX`, where "`XXXXXXX`"
should be replaced with your usernames (e.g. `2024-ca400-sblott-pclarke`).
**Note** that the year should be set as appropriate to your year of study. For example, in the
2022/2023 academic year this would change to '2023-ca400-sblott-pclarke'), 
in the 2023/2024 academic year this would change to '2024-ca400-sblott-pclarke'), etc. 

It is the *name of your repo* which matters (not the name of your project).

You can change the name of your repo on GitLab under:

- Settings / General / Advanced / Change path

It looks like this:

![change-repo-path](./res/repo-change-path.png "Change repo path.")

You should replace all of this file with a README describing your own project.

## Additional resources

## CLI auth test workflow (GitHub sample)

Use this flow to reproduce auth-required behavior with `api.github.com.2026-03-10.yaml`.

```powershell
$env:PYTHONPATH='src'
python src/extract_ir.py src/test_uploads/user1/api.github.com.2026-03-10.yaml --pretty -o build/ir
python src/generate_tests.py build/ir/api.github.com.2026-03-10_ir.json --pretty -o build/tests
python src/test_runner/run_test.py build/tests/api.github.com.2026-03-10_tests.json --timeout 12
```

Expected without token: the suite hits an auth wall after initial `401` responses (for this sample: `3 failed, 8 skipped`).

Run with bearer token:

```powershell
$env:PYTHONPATH='src'
python src/test_runner/run_test.py build/tests/api.github.com.2026-03-10_tests.json --timeout 12 --bearer-token <GITHUB_TOKEN>
```

Or with env var:

```powershell
$env:PYTHONPATH='src'
$env:CONTRACTGUARD_BEARER_TOKEN='<GITHUB_TOKEN>'
python src/test_runner/run_test.py build/tests/api.github.com.2026-03-10_tests.json --timeout 12
```

Note: global auth headers are now ignored for tests with `category="auth"` so auth-negative cases still validate correctly.

