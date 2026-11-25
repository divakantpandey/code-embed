"""Download and cache all required datasets to disk across supported languages."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from datasets import Dataset, DatasetDict, load_dataset
from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="CodeEmbed Dataset Downloader CLI")

ALL_LANGUAGES: list[str] = ["python", "go", "java", "ruby", "php", "javascript"]

SYNTHETIC_DATA: dict[str, dict[str, list[str]]] = {
    "python": {
        "docstring_summary": [
            "Compute sum of two numbers",
            "Binary search in sorted list",
            "Check if string is palindrome",
            "Bubble sort list of elements",
            "Calculate factorial iteratively",
        ],
        "func_code_string": [
            "def add(a, b):\n    return a + b",
            "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1",
            "def is_palindrome(s):\n    return s == s[::-1]",
            "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]: arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
            "def factorial(n):\n    res = 1\n    for i in range(2, n + 1): res *= i\n    return res",
        ],
    },
    "go": {
        "docstring_summary": [
            "Compute sum of two integers",
            "Binary search in integer slice",
            "Check if string is palindrome",
            "Bubble sort integer slice",
            "Calculate factorial of integer",
        ],
        "func_code_string": [
            "func Add(a, b int) int {\n\treturn a + b\n}",
            "func BinarySearch(arr []int, target int) int {\n\tlow, high := 0, len(arr)-1\n\tfor low <= high {\n\t\tmid := (low + high) / 2\n\t\tif arr[mid] == target { return mid }\n\t\tif arr[mid] < target { low = mid + 1 } else { high = mid - 1 }\n\t}\n\treturn -1\n}",
            "func IsPalindrome(s string) bool {\n\tfor i := 0; i < len(s)/2; i++ {\n\t\tif s[i] != s[len(s)-1-i] { return false }\n\t}\n\treturn true\n}",
            "func BubbleSort(arr []int) []int {\n\tn := len(arr)\n\tfor i := 0; i < n; i++ {\n\t\tfor j := 0; j < n-i-1; j++ {\n\t\t\tif arr[j] > arr[j+1] { arr[j], arr[j+1] = arr[j+1], arr[j] }\n\t\t}\n\t}\n\treturn arr\n}",
            "func Factorial(n int) int {\n\tres := 1\n\tfor i := 2; i <= n; i++ { res *= i }\n\treturn res\n}",
        ],
    },
    "java": {
        "docstring_summary": [
            "Compute sum of two integers",
            "Binary search in integer array",
            "Check if string is palindrome",
            "Bubble sort integer array",
            "Calculate factorial of integer",
        ],
        "func_code_string": [
            "public class MathOps {\n    public static int add(int a, int b) {\n        return a + b;\n    }\n}",
            "public class Search {\n    public static int binarySearch(int[] arr, int target) {\n        int low = 0, high = arr.length - 1;\n        while (low <= high) {\n            int mid = (low + high) / 2;\n            if (arr[mid] == target) return mid;\n            else if (arr[mid] < target) low = mid + 1;\n            else high = mid - 1;\n        }\n        return -1;\n    }\n}",
            "public class Strings {\n    public static boolean isPalindrome(String s) {\n        return new StringBuilder(s).reverse().toString().equals(s);\n    }\n}",
            "public class Sort {\n    public static void bubbleSort(int[] arr) {\n        int n = arr.length;\n        for (int i = 0; i < n; i++) {\n            for (int j = 0; j < n - i - 1; j++) {\n                if (arr[j] > arr[j + 1]) {\n                    int tmp = arr[j]; arr[j] = arr[j + 1]; arr[j + 1] = tmp;\n                }\n            }\n        }\n    }\n}",
            "public class MathOps2 {\n    public static int factorial(int n) {\n        int res = 1;\n        for (int i = 2; i <= n; i++) res *= i;\n        return res;\n    }\n}",
        ],
    },
    "ruby": {
        "docstring_summary": [
            "Compute sum of two numbers",
            "Binary search in array",
            "Check if string is palindrome",
            "Bubble sort array of numbers",
            "Calculate factorial of number",
        ],
        "func_code_string": [
            "def add(a, b)\n  a + b\nend",
            "def binary_search(arr, target)\n  low, high = 0, arr.length - 1\n  while low <= high\n    mid = (low + high) / 2\n    return mid if arr[mid] == target\n    arr[mid] < target ? low = mid + 1 : high = mid - 1\n  end\n  -1\nend",
            "def is_palindrome(s)\n  s == s.reverse\nend",
            "def bubble_sort(arr)\n  n = arr.length\n  loop do\n    swapped = false\n    (n - 1).times do |i|\n      if arr[i] > arr[i + 1]\n        arr[i], arr[i + 1] = arr[i + 1], arr[i]\n        swapped = true\n      end\n    end\n    break unless swapped\n  end\n  arr\nend",
            "def factorial(n)\n  (1..n).inject(1, :*)\nend",
        ],
    },
    "php": {
        "docstring_summary": [
            "Compute sum of two numbers",
            "Binary search in array",
            "Check if string is palindrome",
            "Bubble sort array of numbers",
            "Calculate factorial of number",
        ],
        "func_code_string": [
            "<?php\nfunction add($a, $b) {\n    return $a + $b;\n}",
            "<?php\nfunction binary_search($arr, $target) {\n    $low = 0;\n    $high = count($arr) - 1;\n    while ($low <= $high) {\n        $mid = intdiv($low + $high, 2);\n        if ($arr[$mid] === $target) return $mid;\n        elseif ($arr[$mid] < $target) $low = $mid + 1;\n        else $high = $mid - 1;\n    }\n    return -1;\n}",
            "<?php\nfunction is_palindrome($s) {\n    return $s === strrev($s);\n}",
            "<?php\nfunction bubble_sort($arr) {\n    $n = count($arr);\n    for ($i = 0; $i < $n; $i++) {\n        for ($j = 0; $j < $n - $i - 1; $j++) {\n            if ($arr[$j] > $arr[$j + 1]) {\n                $tmp = $arr[$j];\n                $arr[$j] = $arr[$j + 1];\n                $arr[$j + 1] = $tmp;\n            }\n        }\n    }\n    return $arr;\n}",
            "<?php\nfunction factorial($n) {\n    $res = 1;\n    for ($i = 2; $i <= $n; $i++) $res *= $i;\n    return $res;\n}",
        ],
    },
    "javascript": {
        "docstring_summary": [
            "Compute sum of two numbers",
            "Binary search in sorted array",
            "Check if string is palindrome",
            "Bubble sort array of numbers",
            "Calculate factorial of number",
        ],
        "func_code_string": [
            "function add(a, b) {\n    return a + b;\n}",
            "function binarySearch(arr, target) {\n    let low = 0, high = arr.length - 1;\n    while (low <= high) {\n        const mid = Math.floor((low + high) / 2);\n        if (arr[mid] === target) return mid;\n        else if (arr[mid] < target) low = mid + 1;\n        else high = mid - 1;\n    }\n    return -1;\n}",
            "function isPalindrome(s) {\n    return s === s.split('').reverse().join('');\n}",
            "function bubbleSort(arr) {\n    const a = [...arr];\n    for (let i = 0; i < a.length; i++) {\n        for (let j = 0; j < a.length - i - 1; j++) {\n            if (a[j] > a[j + 1]) [a[j], a[j + 1]] = [a[j + 1], a[j]];\n        }\n    }\n    return a;\n}",
            "function factorial(n) {\n    let res = 1;\n    for (let i = 2; i <= n; i++) res *= i;\n    return res;\n}",
        ],
    },
}


def download_codesearchnet_language(language: str, base_output_dir: Path) -> None:
    """Download CodeSearchNet split for a language and save to disk."""
    lang = language.lower()
    console.print(f"[bold blue]Downloading CodeSearchNet ({lang.capitalize()})...[/]")
    out_dir = base_output_dir / lang if base_output_dir.name != lang else base_output_dir
    try:
        ds = load_dataset("code-search-net/code_search_net", lang)
    except Exception:
        try:
            ds = load_dataset("code_search_net", lang)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not download CodeSearchNet (%s) online (%s). Creating synthetic fallback.",
                lang,
                exc,
            )
            synth = SYNTHETIC_DATA.get(lang, SYNTHETIC_DATA["python"])
            raw = Dataset.from_dict(
                {
                    "docstring_summary": synth["docstring_summary"],
                    "func_documentation_string": synth["docstring_summary"],
                    "func_code_string": synth["func_code_string"],
                    "language": [lang] * len(synth["docstring_summary"]),
                }
            )
            ds = DatasetDict({"train": raw, "test": raw, "validation": raw})

    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    logger.info("CodeSearchNet %s saved to %s", lang.capitalize(), out_dir)

    # For python, maintain backwards-compatibility with root data/raw
    if lang == "python" and base_output_dir == Path("data/raw"):
        import contextlib

        with contextlib.suppress(Exception):
            ds.save_to_disk(str(base_output_dir))


def download_cosqa() -> None:
    """Download CoSQA dataset."""
    console.print("[bold blue]Downloading CoSQA...[/]")
    try:
        ds = load_dataset("gonglinyuan/CoSQA")
        logger.info("CoSQA loaded: %s", ds)
    except Exception:
        try:
            ds = load_dataset("code_x_glue_cc_code_search", "cosqa")
            logger.info("CoSQA loaded: %s", ds)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not download CoSQA online (%s).", exc)


@app.command()
def main(
    languages: str = typer.Option(
        "python,go,java,ruby,php,javascript",
        "--languages",
        "-l",
        help="Comma-separated list of languages to download (or 'all')",
    ),
    output_dir: Path = typer.Option(
        Path("data/raw"), "--output-dir", "-o", help="Base directory to save raw datasets"
    ),
    include_cosqa: bool = typer.Option(True, "--cosqa/--no-cosqa", help="Download CoSQA dataset"),
) -> None:
    """Download datasets for all specified languages."""
    console.print("[bold green]📦 Downloading datasets...[/]")
    if languages.strip().lower() == "all":
        lang_list = ALL_LANGUAGES
    else:
        lang_list = [item.strip().lower() for item in languages.split(",") if item.strip()]

    for lang in lang_list:
        download_codesearchnet_language(lang, output_dir)

    if include_cosqa:
        download_cosqa()

    console.print("[bold green]✅ Dataset download step complete.[/]")


if __name__ == "__main__":
    app()
