from typing import Union
import click
import yaml
import importlib

import steps

functions: dict[str, callable] = {
    name: getattr(steps, name)
    for name in filter(lambda x: not x.startswith("_"), dir(steps))
}


def load_yaml(file_path) -> dict[str, Union[dict, str]]:
    with open(file_path, "r") as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            click.echo(f"Error loading YAML file: {e}")

    return data["steps"]

# This could be set up differently by parsing each block into a Pydantic class that defines the
# arguments, types, etc. Similar results, but each step is more structured and uses Pydantic
# validators to validate data
@click.command()
@click.argument("file_path", type=click.Path(exists=True))
def main(file_path):
    for step in load_yaml(file_path):
        name = step.pop('name')
        click.echo(f"Running step: {name}")
        functions[name](**step)

if __name__ == "__main__":
    main()
