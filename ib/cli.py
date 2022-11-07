import click


@click.group()
def cli():
    pass


@cli.group()
def run():
    pass


@run.command()
@click.option("--offxml", required=True, help="A SMIRNOFF force field file.")
@click.option("--molecules", required=True, help="A directory containing SDF files.")
@click.option("--outfile", default="rmsd.csv", help="The file to write RMSDs to.")
def rmsd(offxml, molecules, outfile):
    from ib.compute.rmsd import _run_rmsd

    _run_rmsd(offxml, molecules, outfile)


if __name__ == "__main__":
    cli()
