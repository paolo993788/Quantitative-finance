# Project README template

Copy the template below into `scripts/<project_name>/README.md` and replace every placeholder. Remove any section that does not apply.

````markdown
# Project title

One or two sentences describing what the project does and why it is useful.

## Requirements

- Language and version: for example, Python 3.12.
- Dependencies: `scripts/<project_name>/requirements.txt`, or "none".

## Usage

Run from the repository root:

```bash
python scripts/<project_name>/main.py --input data/examples/<file>
```

## Inputs

| Name | Format | Description |
| --- | --- | --- |
| `<input>` | CSV | Content of the input and units of measure. |

## Outputs

| File | Description |
| --- | --- |
| `outputs/<project_name>/<file>` | Content of the output. |

## Method

Model, assumptions, parameter values, day-count and compounding conventions, and numerical method.

## Verification

Closed-form solutions, published benchmarks or limiting cases used to validate the implementation, the numerical tolerances adopted, Monte Carlo seeds and standard errors, and the exact command to reproduce the checks.

## References

Sources for models, data (with license) and any reused code.
````
