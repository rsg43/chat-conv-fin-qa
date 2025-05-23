"""
MCP server for basic mathemtical operations, with tools provided as decribed in
the ConvFinQA paper, implemeted using FastMCP.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool()
def add(a: float, b: float) -> float:
    """
    Add two numbers.

    :param a: First number
    :type a: float
    :param b: Second number
    :type b: float
    :return: Sum of a and b
    :rtype: float
    """
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """
    Subtract two numbers.

    :param a: First number
    :type a: float
    :param b: Second number
    :type b: float
    :return: Difference of a and b
    :rtype: float
    """
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.

    :param a: First number
    :type a: float
    :param b: Second number
    :type b: float
    :return: Product of a and b
    :rtype: float
    """
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """
    Divide two numbers.

    :param a: First number
    :type a: float
    :param b: Second number
    :type b: float
    :raises ValueError: If b is zero
    :return: Quotient of a and b
    :rtype: float
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@mcp.tool()
def exp(a: float, b: float) -> float:
    """
    Raise a to the power of b.

    :param a: Base
    :type a: float
    :param b: Exponent
    :type b: float
    :return: a raised to the power of b
    :rtype: float
    """
    res: float = a**b
    return res


@mcp.tool()
def greater(a: float, b: float) -> bool:
    """
    Check if a is greater than b.

    :param a: First number
    :type a: float
    :param b: Second number
    :type b: float
    :return: True if a is greater than b, False otherwise
    :rtype: bool
    """
    return a > b


if __name__ == "__main__":
    mcp.run(transport="stdio")
