from enum import Enum


# used for nodes (node level)
class CVNodeLevel(Enum):
    ROOT = 1
    SECTION = 2
    SUBSECTION = 3


# Represents CV structure
class CVNode:
    def __init__(self):
        self.id: str | None = None
        self.title: str = ""
        self.text: str = ""
        self.parent: CVNode | None = None
        self.children: list[CVNode] = []
        self.level: CVNodeLevel | None = None

    def get_path(self) -> str:
        parts = []

        temp_node = self
        while temp_node is not None:
            # strip heading markers and whitespace
            title = temp_node.title.lstrip('#').strip()
            parts.append(title)
            temp_node = temp_node.parent

        # reverse to get root -> leaf
        parts.reverse()

        # join into path and leave out the root node as it is always the same
        return "/".join(parts[1:])
