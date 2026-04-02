# ============================================================
# math_server.py
# LOCAL MCP SERVER — Safe math calculator
# This runs as a subprocess launched by your Jupyter notebook
# ============================================================

import math as mathlib
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("math")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together and return the result."""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a and return the result."""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together and return the result."""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b and return the result."""
    if b == 0:
        return "Error: Cannot divide by zero"
    return round(a / b, 6)


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent. Example: power(2, 10) = 1024."""
    return round(base ** exponent, 6)


@mcp.tool()
def square_root(number: float) -> str:
    """Calculate the square root of a number."""
    if number < 0:
        return "Error: Cannot take square root of a negative number"
    return str(round(mathlib.sqrt(number), 6))


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a full mathematical expression safely.
    Supports: +, -, *, /, **, sqrt, log, sin, cos, pi, e, abs, round.
    Examples: '2025 - 1991', 'sqrt(144)', '15 * 8 + 20'"""
    try:
        safe_globals = {
            "__builtins__": {},
            "sqrt":  mathlib.sqrt,
            "log":   mathlib.log,
            "log2":  mathlib.log2,
            "log10": mathlib.log10,
            "sin":   mathlib.sin,
            "cos":   mathlib.cos,
            "tan":   mathlib.tan,
            "ceil":  mathlib.ceil,
            "floor": mathlib.floor,
            "pi":    mathlib.pi,
            "e":     mathlib.e,
            "abs":   abs,
            "round": round,
            "pow":   pow,
        }
        result = eval(expression, safe_globals)
        return f"{expression} = {round(float(result), 6)}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except NameError as err:
        return f"Error: Unknown function — {err}"
    except SyntaxError:
        return f"Error: Invalid syntax in '{expression}'"
    except Exception as err:
        return f"Error evaluating '{expression}': {err}"


# ============================================================
# START THE SERVER
# stdio = runs as subprocess from notebook
# ============================================================
if __name__ == "__main__":
    mcp.run(transport="stdio")
