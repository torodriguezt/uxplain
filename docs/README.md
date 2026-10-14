# Building the documentation

Run the following commands from the repository root with your virtual
environment activated. Install the package and the documentation tools:

```sh
python -m pip install -e ".[docs]"
```

The site uses MkDocs Material and mkdocstrings. Edit the guides in `docs/docs/`
and the navigation and plugin settings in `docs/mkdocs.yml`. The API reference
is generated from the NumPy-style docstrings in `uxplain/`.

Build the site with warnings treated as errors:

```sh
python -m mkdocs build --strict -f docs/mkdocs.yml
```

The generated HTML is written to `docs/site/`. Preview the site while editing:

```sh
python -m mkdocs serve -f docs/mkdocs.yml
```

Open <http://127.0.0.1:8000/uxplain/>. MkDocs rebuilds the preview when a source file
changes.

## Validate the extension example

The [Extending guide](docs/extending.md) includes a MAPIE adapter covered by
`tests/test_extensibility.py`. Use the same MAPIE version as the documentation
workflow:

```sh
python -m pip install pytest "mapie==1.5.0"
python -c "import mapie; print(mapie.__version__)"
python -m pytest tests/test_extensibility.py -q
```

Both tests must pass, including the MAPIE test. The explicit import ensures
that a missing MAPIE dependency fails validation instead of silently skipping
the adapter test. A strict MkDocs build validates the rendered documentation;
it does not execute Python code blocks.

## Publish to GitHub Pages

The [documentation workflow](../.github/workflows/docs.yml) runs the extension
tests, builds the site in strict mode, and uploads the generated site as an
artifact for pull requests, pushes to `main`, and manual runs. It publishes to
GitHub Pages only for a push to `main` or a manual run on `main`.

To enable publishing, open the repository's **Settings > Pages > Build and
deployment**, then select **GitHub Actions** as the **Source**. See GitHub's
[custom workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

After these changes reach `main`, check that the **Documentation** workflow
finishes successfully. The expected public URL is
<https://torodriguezt.github.io/uxplain/>. Verify that the home page, Getting
started, Extending, and API reference load before citing this URL or setting
it as the `Documentation` URL in `pyproject.toml`.
