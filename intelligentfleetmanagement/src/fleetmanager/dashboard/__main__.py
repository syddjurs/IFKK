import click

from .view import app


@click.command()
@click.option("--debug", is_flag=True, default=False)
def cli(debug):
    app.run_server(debug=debug)


if __name__ == "__main__":
    cli()
