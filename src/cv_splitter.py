from src.models import CVNode


class CvSplitter:

    cv_content: str

    def __init__(self, cv: str):
        self.cv_content = cv.strip()


    def build_doc_tree(self) -> CVNode:
        lines = self.cv_content.splitlines()
        root_node = CVNode()
        last_2nd_level_node = CVNode()

        current_node = None
        for line in lines:
            if line.strip().startswith('###'):
                current_node = CVNode()
                current_node.title = line.strip()
                current_node.parent = last_2nd_level_node
                last_2nd_level_node.children.append(current_node)
            elif line.strip().startswith('##'):
                current_node = CVNode()
                current_node.title = line.strip()
                current_node.parent = root_node
                root_node.children.append(current_node)
                last_2nd_level_node = current_node
            elif line.strip().startswith('#'):
                current_node = CVNode()
                current_node.title = line.strip()
                current_node.parent = None
                root_node = current_node
            else:
                current_node.text += line.strip() + ' '

        return root_node
