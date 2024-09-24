# playground-llm

My playground to implement LLM related code and components

## Project Organization

Differently from other data science template projects, I deliberately removed `data`, `models`, `reports`  directories.
I don't think these artifacts should be in the code repository, but rather in dedicated object stores.

## Preparing the environment

> NOTE: `poetry` is used to manage the dependencies for this project.

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
poetry shell
```

Alternatively, run everything prefixing with `poetry run`.
