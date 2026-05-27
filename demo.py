"""
demo.py — ALAS interactive terminal demo

Runs a complete ALAS session in the terminal without needing a frontend.
Shows the LangGraph agent, memory system, and evaluation engine in action.

Usage:
    python demo.py                                    # default: job interview
    python demo.py --scenario difficult_conversation_v1
    python demo.py --user alice --scenario job_interview_v1
    python demo.py --list                             # list available scenarios

Requires:  OPENAI_API_KEY in .env or environment.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text
from rich.rule import Rule

from agent_service.session import startup, orchestrator
from agent_service.scenarios.registry import list_scenarios

console = Console()

EMOTION_COLORS = {
    "neutral": "white",
    "curious": "cyan",
    "concerned": "yellow",
    "encouraging": "green",
    "challenging": "red",
    "warm": "magenta",
    "disappointed": "dark_orange",
}

SCORE_COLOR = lambda v: "green" if v >= 0.7 else ("yellow" if v >= 0.5 else "red")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_header():
    console.print()
    console.print(Panel.fit(
        "[bold blue]ALAS — Adaptive Learning Agent System[/bold blue]\n"
        "[dim]Production-inspired demo · LangGraph · Multi-layer Memory · Real-time Evaluation[/dim]",
        border_style="blue",
    ))
    console.print()


def print_avatar(name: str, text: str, emotion: str, phase: str):
    color = EMOTION_COLORS.get(emotion, "white")
    header = f"[bold {color}]{name}[/bold {color}]  [dim][{emotion}  ·  phase: {phase}][/dim]"
    console.print(Panel(
        f"[{color}]{text}[/{color}]",
        title=header,
        border_style=color,
        padding=(0, 1),
    ))


def print_score(score: dict, turn_index: int):
    table = Table(
        title=f"[dim]Evaluation · Turn {turn_index}[/dim]",
        box=box.SIMPLE,
        show_header=True,
        header_style="dim",
        padding=(0, 1),
    )
    table.add_column("Dimension", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Bar", min_width=20)

    dims = ["clarity", "empathy", "structure", "relevance", "confidence", "composite"]
    for dim in dims:
        val = score.get(dim, 0.0)
        color = SCORE_COLOR(val)
        bar_filled = int(val * 20)
        bar = f"[{color}]{'█' * bar_filled}[/{color}][dim]{'░' * (20 - bar_filled)}[/dim]"
        style = "bold" if dim == "composite" else ""
        table.add_row(
            f"[{style}]{dim}[/{style}]",
            f"[{color}{style}]{val:.2f}[/{color}{style}]",
            bar,
        )

    rationale = score.get("rationale", "")
    console.print(table)
    if rationale:
        console.print(f"  [dim italic]↳ {rationale}[/dim italic]")
    console.print()


def print_session_summary(summary: dict):
    console.print(Rule("[bold green]Session Complete[/bold green]"))
    avg = summary.get("averages", {})
    trend = summary.get("trend", "stable")
    trend_icon = "📈" if trend == "improving" else ("📉" if trend == "declining" else "➡️")

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Turns Evaluated", str(summary.get("turns_evaluated", 0)))
    table.add_row("Trend", f"{trend_icon} {trend}")
    table.add_row("", "")
    for dim, val in avg.items():
        color = SCORE_COLOR(val)
        table.add_row(dim.title(), f"[{color}]{val:.3f}[/{color}]")

    console.print(table)

    strongest = summary.get("strongest_dimension", "")
    weakest = summary.get("weakest_dimension", "")
    if strongest:
        console.print(f"\n  [green]✓ Strongest: {strongest}[/green]")
    if weakest:
        console.print(f"  [yellow]⚠ Growth area: {weakest}[/yellow]")

    rationales = summary.get("rationales", [])
    if rationales:
        console.print("\n[dim]Turn-by-turn notes:[/dim]")
        for i, r in enumerate(rationales):
            if r:
                console.print(f"  [dim]Turn {i}: {r}[/dim]")
    console.print()


def print_memory_context(context: dict):
    """Show what the memory system retrieved for this turn (debug view)."""
    chunks = context.get("scenario_chunks", [])
    summaries = context.get("user_summaries", [])
    notes = context.get("behavioral_notes", [])

    if not (chunks or summaries or notes):
        return

    lines = ["[dim][Memory context retrieved this turn][/dim]"]
    if summaries:
        lines.append(f"  [dim]Episodic ({len(summaries)} prior session summary/ies):[/dim]")
        for s in summaries[:2]:
            lines.append(f"    [dim]• {s[:80]}{'…' if len(s) > 80 else ''}[/dim]")
    if notes:
        lines.append(f"  [dim]Behavioral notes:[/dim]")
        for n in notes[:2]:
            lines.append(f"    [dim]• {n[:80]}[/dim]")
    if chunks:
        lines.append(f"  [dim]Scenario KB ({len(chunks)} chunk(s) retrieved)[/dim]")

    console.print("\n".join(lines))
    console.print()


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------

async def run_demo(user_id: str, scenario_id: str, show_memory: bool = False):
    print_header()

    console.print(f"[dim]Initialising session for user [bold]{user_id}[/bold] · scenario [bold]{scenario_id}[/bold]...[/dim]")
    startup()

    info = await orchestrator.create_session(user_id=user_id, scenario_id=scenario_id)

    console.print(f"[dim]Session ID: {info.session_id}[/dim]")
    console.print(f"[dim]Scenario: {info.scenario_title}[/dim]")
    console.print()

    print_avatar(info.persona_name, info.opening_line, "neutral", "setup")
    console.print("[dim]Type your response below. Enter [bold]quit[/bold] or [bold]exit[/bold] to end.[/dim]\n")

    turn = 0
    while True:
        try:
            student_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Interrupted.[/dim]")
            break

        if student_input.lower() in {"quit", "exit", "q"}:
            break

        if not student_input:
            console.print("[dim]Please enter a response.[/dim]")
            continue

        turn += 1
        console.print()

        # Stream the response token by token
        avatar_text_parts = []
        result_data = None

        console.print(f"[dim]{info.persona_name} is responding...[/dim]")

        async for chunk in orchestrator.stream_message(
            session_id=info.session_id,
            user_id=user_id,
            student_message=student_input,
        ):
            if chunk["type"] == "token":
                avatar_text_parts.append(chunk["content"])
            elif chunk["type"] == "result":
                result_data = chunk["data"]
            elif chunk["type"] == "error":
                console.print(f"[red]Error: {chunk['message']}[/red]")
                break

        if result_data:
            full_text = "".join(avatar_text_parts)
            emotion = result_data.get("emotion", "neutral")
            phase = result_data.get("scenario_phase", "core")

            print_avatar(info.persona_name, full_text, emotion, phase)

            if show_memory:
                # Re-fetch retrieved context from STM for display
                stm = orchestrator._stm_debug(info.session_id)
                if stm:
                    pass  # future: expose retrieved_context from result

            score = result_data.get("turn_score")
            if score:
                print_score(score, turn)

            if result_data.get("session_ended"):
                summary = result_data.get("session_summary")
                if summary:
                    print_session_summary(summary)
                break

    # End session if not already ended
    final = await orchestrator.end_session(info.session_id)
    summary = final.get("summary")
    if summary and summary.get("turns_evaluated", 0) > 0:
        print_session_summary(summary)

    console.print("[dim]Session ended. Thank you for using ALAS.[/dim]")


def list_available_scenarios():
    console.print("\n[bold]Available Scenarios[/bold]\n")
    for s in list_scenarios():
        console.print(f"  [cyan]{s['id']}[/cyan]")
        console.print(f"    {s['title']}")
        console.print(f"    [dim]{s['description']}[/dim]\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ALAS interactive terminal demo")
    parser.add_argument("--user", default="demo-user", help="User ID")
    parser.add_argument("--scenario", default="job_interview_v1", help="Scenario ID")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit")
    parser.add_argument("--memory", action="store_true", help="Show memory context each turn")
    args = parser.parse_args()

    if args.list:
        list_available_scenarios()
        return

    asyncio.run(run_demo(args.user, args.scenario, show_memory=args.memory))


if __name__ == "__main__":
    main()
