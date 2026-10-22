from rich.console import Console
try:
    from pyfiglet import Figlet
except ImportError:
    Figlet = None

console = Console()
err_style = "bold red"
warning_style = "bold yellow"
success_style = "green"
dim_style = "dim"

# benchmark dataset
f = Figlet(font='digital') if Figlet is not None else None
head_print = lambda x : f.renderText(x) if f is not None else x

head_style = "bold white on blue"
subhead_style = "bold black on bright_blue"
row_style = "black on bright_white"


# Generator styles
head_style_2 = "bold white on magenta"
subhead_style_2 = "white"
