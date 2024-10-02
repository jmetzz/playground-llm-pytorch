# playground-llm

My playground to implement LLM related code and components.

> NOTE: this project uses `uv` to manage the python environment and dependencies.

## Project Organization

Differently from other data science template projects out there, I deliberately don't use `data`, `models`, `reports`  directories under version control.

I don't think these artifacts should be in the code repository, but rather in dedicated object stores.

## Preparing the environment

To create the python environment and install the project dependencies,
enter the directory you created the project and run

```bash
make deps
```

Apart from the creation of the python environment, `make deps` will also initialize this project as a `git` repository and setup `pre-commit` hooks.

Run `make help` to see other make tasks available.

## Executing the code

Make sure you activate the correct environment before executing any script/notebook.

```bash
uv run python <your-cli.py> [PARAMS]
```
