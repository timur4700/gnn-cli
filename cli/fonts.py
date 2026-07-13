from typing import Literal

def color_font(text: str, c: Literal['red', 'green', 'yellow', 'blue']):

    colors = {
        'red': "\033[31m{}\033[0m",
        'green': "\033[32m{}\033[0m",
        'yellow': "\033[33m{}\033[0m",
        'blue': "\033[34m{}\033[0m"
    }

    return f"{colors[c].format(text)}"
