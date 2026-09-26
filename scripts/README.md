# Scripts

Each project lives in its own subfolder, for example `scripts/project_name/`, containing the code and a README based on the [project README template](../docs/script-template.md).

A project README states the language and version, the dependencies, the run command from the repository root, and the expected inputs and outputs. For multi-file projects, keep the dependency file next to the code it refers to.

Projects that combine Python with C++ keep the C++ sources in a `cpp/` subfolder and expose them to Python through [pybind11](https://github.com/pybind/pybind11); they are built with `python -m pip install -e scripts/<project_name>`.

Add every new project to the catalogue in the [main README](../README.md).
