import hashlib

from ai_chat.models import CVNode, CVNodeLevel


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class CVParser:
    cv_content: str

    def __init__(self, cv: str):
        self.cv_content = cv.strip()

    def build_tree(self) -> CVNode:
        lines = self.cv_content.splitlines()
        root_node = CVNode()
        last_2nd_level_node = CVNode()

        current_node = None
        for line in lines:
            clean_line = line.strip()
            if clean_line.startswith('###'):
                current_node = CVNode()
                current_node.title = clean_line
                current_node.parent = last_2nd_level_node
                last_2nd_level_node.children.append(current_node)
                current_node.id = stable_id(current_node.get_path())
                current_node.level = CVNodeLevel.SUBSECTION
            elif clean_line.startswith('##'):
                current_node = CVNode()
                current_node.title = clean_line
                current_node.parent = root_node
                root_node.children.append(current_node)
                last_2nd_level_node = current_node
                current_node.id = stable_id(current_node.get_path())
                current_node.level = CVNodeLevel.SECTION
            elif clean_line.startswith('#'):
                current_node = CVNode()
                current_node.title = clean_line
                current_node.parent = None
                root_node = current_node
                current_node.id = stable_id(current_node.get_path())
                current_node.level = CVNodeLevel.ROOT
            else:
                current_node.text += clean_line + ' '

        return root_node
