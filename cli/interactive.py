import inquirer
from inquirer.themes import Default, term

class MyTheme(Default):
    def __init__(self):
        super().__init__()
        self.Question.mark_color = term.yellow
        self.Question.brackets_color = term.cyan
        self.Question.default_color = term.green

        self.List.selection_color = term.green
        self.List.selection_cursor = "➤"
        self.List.unselected_color = term.white


def query(msg: str, choices: list, use_index=True) -> int:

    question = [inquirer.List("choice", message=msg, choices=choices)]
    answer = inquirer.prompt(question, theme=MyTheme())

    if use_index:
        return choices.index(answer['choice'])
    
    else:
        return answer['choice']
