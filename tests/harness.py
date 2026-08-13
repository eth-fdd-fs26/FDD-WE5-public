"""A tiny, dependency-light test harness with friendly console output.

Tests are grouped by the *function* they exercise, so the final report tells a
student exactly which function is problematic — not just that "a test failed".

Usage in a test file:

    from tests.harness import TestSuite

    suite = TestSuite("Retrieval Core")

    @suite.case("keyword_search", "ranks matching chunks first")
    def _():
        assert ...

    if __name__ == "__main__":
        raise SystemExit(0 if suite.run() else 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class _Result:
    name: str
    passed: bool
    error: str = ""


class TestSuite:
    def __init__(self, title: str):
        self.title = title
        # Insertion-ordered: target function name -> list of (test name, fn).
        self._groups: dict[str, list[tuple[str, Callable[[], None]]]] = {}

    def case(self, target: str, name: str):
        """Register a test under the function it is meant to validate."""

        def decorator(fn: Callable[[], None]) -> Callable[[], None]:
            self._groups.setdefault(target, []).append((name, fn))
            return fn

        return decorator

    def run(self) -> bool:
        """Run every test, print a report, and return True if all passed."""
        console.print()
        console.print(Panel(self.title, style="bold cyan", box=box.ROUNDED, expand=False))

        results: dict[str, list[_Result]] = {}
        for target, tests in self._groups.items():
            results[target] = [self._run_one(name, fn) for name, fn in tests]

        self._print_details(results)
        return self._print_summary(results)

    @staticmethod
    def _run_one(name: str, fn: Callable[[], None]) -> _Result:
        try:
            fn()
            return _Result(name, True)
        except Exception as exc:  # noqa: BLE001 - we want to catch everything
            return _Result(name, False, f"{type(exc).__name__}: {exc}")

    def _print_details(self, results: dict[str, list[_Result]]) -> None:
        for target, group in results.items():
            passed = sum(r.passed for r in group)
            color = "green" if passed == len(group) else "red"
            console.print(f"\n[bold {color}]{target}[/]  [{color}]{passed}/{len(group)}[/]")
            for r in group:
                if r.passed:
                    console.print(f"  [green]✓[/] {r.name}")
                else:
                    console.print(f"  [red]✗[/] {r.name}  [dim]→ {r.error}[/]")

    def _print_summary(self, results: dict[str, list[_Result]]) -> bool:
        table = Table(box=box.SIMPLE_HEAVY, title="Summary", title_style="bold")
        table.add_column("Function", style="bold")
        table.add_column("Passed", justify="center")
        table.add_column("Status", justify="center")

        problematic: list[str] = []
        total_passed = total = 0
        for target, group in results.items():
            passed = sum(r.passed for r in group)
            total_passed += passed
            total += len(group)
            ok = passed == len(group)
            if not ok:
                problematic.append(target)
            status = "[green]✓ PASS[/]" if ok else "[red]✗ FAIL[/]"
            table.add_row(target, f"{passed}/{len(group)}", status)

        console.print()
        console.print(table)

        if problematic:
            body = "Functions needing attention:\n" + "\n".join(f"  • {p}" for p in problematic)
            console.print(
                Panel(body, title=f"{total_passed}/{total} checks passed", style="yellow", box=box.ROUNDED, expand=False)
            )
        else:
            console.print(
                Panel(f"All {total} checks passed — nice work! 🎉", style="bold green", box=box.ROUNDED, expand=False)
            )
        return not problematic
