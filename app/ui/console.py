from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


def print_welcome():
    console.print("[bold cyan]🤖 Welcome to DocMind[/bold cyan]")
    console.print("Type 'exit' to quit.\n")


def ask_user() -> str:
    return console.input("[bold green]You >[/bold green] ")


def print_goodbye():
    console.print("Goodbye 👋")


def print_error(message: str):
    console.print(f"[bold red]Unable to reach Gemini:[/bold red] {message}")


def stream_response(chunks) -> str:
    """Render a streamed Gemini response inside a live-updating panel.

    `chunks` is the generator from ChatService.stream(). We accumulate the
    text and redraw the panel on every chunk, so the box fills in as the
    response arrives instead of appearing all at once.
    """
    text = ""
    with Live(console=console, refresh_per_second=12) as live:
        for chunk in chunks:
            text += chunk
            live.update(Panel(Markdown(text), title="Gemini", border_style="cyan"))
    return text
