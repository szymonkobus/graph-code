from __future__ import annotations


class TNode:
    def __init__(self, 
                id: int, 
                parent: TNode | None = None, 
                children: list[TNode] = [], 
                name: str = ''):
        self.id = id
        self.parent = parent
        self.children = children
        self.name = name
        if parent is not None:
            self.link_parent()
        if len(children)!=0:
            self.link_children()
        
    def link_parent(self) -> None:
        self.parent.children.append(self)

    def link_children(self) -> None:
        for child in self.children:
            child.parent = self