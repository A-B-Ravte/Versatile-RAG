from dataclasses import dataclass
from typing import Optional

@dataclass
class Metadata():
    page_no : int
    source : str
    parent_id : Optional[str]

@dataclass
class Chunk():
    chunk_id : str
    chunk_text : str
    metadata : Metadata
