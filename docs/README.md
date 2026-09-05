<div align="center">

**[📗 Documentation](https://rathaumons.github.io/pyppbox/) | [🚀 Getting started](https://rathaumons.github.io/pyppbox/getstarted.html) | [💡 Examples](https://rathaumons.github.io/pyppbox/examples.html)**

</div>

Build from the repository root using Python 3.11 and the package's runtime dependencies:

```sh
python -m pip install -r requirements/docs.txt
python -m sphinx -b html -n -W --keep-going docs docs/_build/html
```

The build imports the actual API for its signatures and docstrings and includes the
scripts from `examples/` directly. Model weights and classifier retraining are not
required for the documentation build. Unresolved API references and other warnings
fail the build in pull requests; publishing remains restricted to pushes to `main`.
