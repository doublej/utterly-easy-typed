from rich import box
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from collections import deque
from datetime import datetime
import sys


class ConsoleTable:
    def __init__(self, total_cost_decimals: int = 6, table_width: int = 120):
        # Replace standard console output with our controlled console
        self.console = Console(file=sys.stdout, highlight=False)
        self.table = Table(show_footer=False, width=table_width)
        self.total_cost = 0
        self.total_cost_decimals = total_cost_decimals
        self.bottom_section_height = 25
        self.logs = deque(maxlen=self.bottom_section_height)  # Rolling window of 5 latest logs
        self.layout = Layout()
        self.vad_status = Text("●", style="dim")  # Default VAD status indicator
        
        # Ensure standard output is flushed
        sys.stdout.flush()

    def _update_cost(self, cost: float):
        self.total_cost += cost
        self.table.columns[4].footer = (
            f"${round(self.total_cost, self.total_cost_decimals)}"
        )

    def _setup_table(self):
        # Only clear on initial setup, not during updates
        self.table.add_column("Date", no_wrap=True)
        self.table.add_column(
            "Transcription", Text.from_markup("[b]Total:", justify="right")
        )
        self.table.add_column("Mode", no_wrap=True)
        self.table.add_column("Filtered Text")
        self.table.add_column(
            "Cost", Text.from_markup("[b]$0", justify="right"), no_wrap=True
        )
        self.table.show_footer = True

        self.table.columns[0].header_style = "bold green"
        self.table.columns[0].style = "green"
        self.table.columns[1].header_style = "bold blue"
        self.table.columns[1].style = "blue"
        self.table.columns[1].footer = "Total"
        self.table.columns[2].header_style = "bold magenta"
        self.table.columns[2].style = "magenta"
        self.table.columns[3].header_style = "bold yellow"
        self.table.columns[3].style = "yellow"
        self.table.columns[4].header_style = "bold cyan"
        self.table.columns[4].style = "cyan"
        self.table.row_styles = ["none", "dim"]
        self.table.box = box.SIMPLE_HEAD
        
        # Create a fixed layout
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="table"),
            Layout(name="bottom_section", size=self.bottom_section_height)  # Combined section for status and logs
        )
        
        # Split the bottom section horizontally for status and logs
        self.layout["bottom_section"].split_row(
            Layout(name="status", size=60),  # Increase size for VAD indicator to show more info
            Layout(name="logs")               # Logs on the right
        )
        
        # Add table with centering
        table_centered = Align.center(self.table)
        self.layout["table"].update(table_centered)
        
        # Initialize VAD status panel
        self._update_vad_panel()
        
        # Initialize empty logs panel
        self._update_logs_panel()

    def _update_vad_panel(self):
        """Update the VAD status panel with the current indicator"""
        vad_panel = Panel(
            Align.left(self.vad_status),  # Left align for better readability of multiple lines
            title="Voice Activity Monitor",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        self.layout["status"].update(vad_panel)
        
    def update_vad_status(self, status: Text):
        """Update the VAD status indicator"""
        self.vad_status = status
        self._update_vad_panel()
        # Refresh the display if we're live rendering
        if hasattr(self, 'live_rendering') and self.live_rendering._started:
            self.live_rendering.refresh()

    def _update_logs_panel(self):
        if not self.logs:
            logs_content = Text("No logs yet", style="dim")
        else:
            logs_content = Text("\n".join(self.logs))
        
        logs_panel = Panel(
            Align.left(logs_content),
            title="Recent Logs",
            border_style="blue",
            box=box.ROUNDED,
            padding=(0, 1)
        )
        self.layout["logs"].update(logs_panel)

    def add_log(self, message: str):
        """Add a log message to the rolling log window."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        self._update_logs_panel()
        # Ensure the log is visible immediately, even if we're not rendering yet
        if not hasattr(self, 'live_rendering') or not self.live_rendering._started:
            return
        self.live_rendering.refresh()

    def __enter__(self):
        # Ensure we have a clean console state
        print("\033[2J\033[H", end="")  # ANSI clear screen and move to home position
        sys.stdout.flush()
        
        # Clear console once at the beginning
        self.console.clear()
        self._setup_table()
        self.live_rendering = Live(
            self.layout,
            console=self.console,
            screen=True,  # Use alternate screen buffer
            refresh_per_second=2,  # Reduce refresh rate for stability
            auto_refresh=True,
            transient=False,  # Don't leave output when done
        )
        self.live_rendering.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        self.live_rendering.__exit__(*args, **kwargs)

    def insert(self, transcription: str, mode: str, filtered_text: str, cost: float):
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%dth %B, %I:%M%p")
        self.table.add_row(formatted_datetime, transcription, mode, filtered_text, f"${cost}")
        self._update_cost(cost)
        self.add_log(f"Added: {transcription[:30]}... (Mode: {mode}, Cost: ${cost})")
        # Text("API Error", style="bold red")
