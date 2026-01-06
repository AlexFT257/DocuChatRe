from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


@tool
def search(query: str) -> str:
    """Busca informacion actualizada en internet, utilizando DuckDuckGo.
    Util para noticias, verificar hechos y datos actuales. Entrega los resultados en texto plano."""

    try:
        search = DuckDuckGoSearchResults(num_results=5)
        results = search.run(query)

        # formatted = "Resultados de búsqueda:\n\n"
        # formatted += results

        return results
    except Exception as e:
        return f"Error al buscar: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """Calcula expresiones matemáticas. Soporta: +, -, *, /, **, sqrt, sin, cos, tan, log, etc.
    Ejemplos: '2 + 2', 'sqrt(16)', 'sin(pi/2)', '2^3', 'log(100)'"""
    try:
        # Transformaciones para hacer la sintaxis más flexible
        transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )

        # Parsear y evaluar de forma segura
        result = parse_expr(expression, transformations=transformations)

        # Evaluar numéricamente
        numeric_result = result.evalf()

        return f"Resultado: {numeric_result}"
    except Exception as e:
        return f"Error al calcular: {str(e)}"
