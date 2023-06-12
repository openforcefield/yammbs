import click


@click.group()
def cli():
    pass


@cli.group()
def run():
    pass


@run.command()
@click.option("--force-field", required=True, help="An identifier of a force field.")
@click.option("--molecules", required=True, help="A directory containing SDF files.")
@click.option("--outfile", default="rmsd.csv", help="The file to write RMSDs to.")
def rmsd(force_field, molecules, outfile):
    from ib.compute.rmsd import _run_rmsd

    _run_rmsd(force_field, molecules, outfile)


if __name__ == "__main__":
    cli()
