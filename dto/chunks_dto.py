from dataclasses import dataclass, field
from typing import Optional, List

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
    embedding: Optional[List[float]] = field(default=None)
