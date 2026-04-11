from dataclasses import dataclass

@dataclass
class Metadata():
    page_no : int
    source : str
    parent_id : int

@dataclass
class Chunk():
    chunk_id : int
    chunk_text : str
    metadata : Metadata
