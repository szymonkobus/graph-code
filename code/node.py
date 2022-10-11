from __future__ import annotations


class TNode:
    def __init__(self, 
                idx: int, 
                parent: TNode | None = None, 
                children: list[TNode] | None = None, 
                name: str = ''):
        self.idx = idx
        self.parent = parent
        self.children = children if children is not None else []
        self.name = name
        if parent is not None:
            self.link_parent()
        if len(self.children)!=0:
            self.link_children()
      
    def link_parent(self) -> None:
        if self.parent is not None:
            self.parent.children.append(self)

    def link_children(self) -> None:
        for child in self.children:
            child.parent = self

    def draw(self):
        lines = self.rec_draw([], 0)
        return "\n".join(lines)
        
    def rec_draw(self, lines, depth):
        desc = '  '*depth + \
               f'idx: {self.idx}, n_child: {len(self.children)}'
        lines.append(desc)
        for child in self.children:
            lines = child.rec_draw(lines, depth+1)
        return lines
