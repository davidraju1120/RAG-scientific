import re
from typing import List

LATEX_PATTERN = r'\$.*?\$|\\\[.*?\\\]'  # Matches $...$ or \[...\]


def extract_latex_equations(text: str) -> List[str]:
    """
    Extract LaTeX equations from text.
    """
    return re.findall(LATEX_PATTERN, text, re.DOTALL)


def render_latex_equations(equations: List[str]) -> str:
    """
    Format LaTeX equations for Streamlit display.
    """
    return '\n'.join([f'$$ {eq.strip("$").strip()} $$' for eq in equations])
