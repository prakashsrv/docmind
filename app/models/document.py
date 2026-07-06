from dataclasses import dataclass


@dataclass
class Document:
    id: str
    name: str
    text: str
