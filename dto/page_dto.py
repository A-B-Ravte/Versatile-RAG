from dataclasses import dataclass

@dataclass
class Page():
    page_no : int
    text : str
    source : str