import yaml
from rich.console import Console
from rich.table import Table

from tiamat.mandelbrot import Mandelbrot


def process():
    # TODO DOC
    with open('params.yml', 'r') as file:
        simulations = yaml.safe_load_all(file)

        console = Console()

        for i,params in enumerate(simulations):

            ymin = params['ymin']
            ymax = params['ymax']
            xmin = params['xmin']
            xmax = params['xmax']
            res = params['res']
            attribute = params['attribute']

            table = Table(title=f"Mandelbrot Object #{i+1}")

            table.add_column("Parameter", justify="left", style="magenta", no_wrap=True)
            table.add_column("Value", justify="right",style="cyan")

            table.add_row("Bottom left corner", f"{xmin},{ymin}")
            table.add_row("Top right corner", f"{xmax}, {ymax}")
            table.add_row("Resolution", str(res))

            print(f"Constructing Mandelbrot object...\n")
            M = Mandelbrot((xmin, xmax),(ymin,ymax),res)

            console.print(table)

            for N in params['max_step']:
                if attribute == 'escape_time':
                    M.compute_escape_time(N)
                elif attribute == 'lyapunov':
                    pass # TODO Faire l'exposant de lyapunov
                print("Saving data...")
                M.save(attribute, path="./data")

if __name__ == '__main__':
    console = Console()
    console.print("\nStart of program.", style="green")
    process()
    console.print("\nEnd of program.", style='green')