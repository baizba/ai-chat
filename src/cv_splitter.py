from src.models import CVNode


class CvSplitter:

    cv_content: str

    def __init__(self, cv: str):
        self.cv_content = cv.strip()


    def build_doc_tree(self) -> CVNode:
        lines = self.cv_content.splitlines()
        root_node = CVNode()
        last_2nd_level_node = CVNode()

        for line in lines:
            current_node = CVNode()
            if line.strip().startswith('###'):
                current_node.title = line.strip('###').strip()
                current_node.parent = last_2nd_level_node
                last_2nd_level_node.children.append(current_node)
            elif line.strip().startswith('##'):
                current_node.title = line.strip('##').strip()
                current_node.parent = root_node
                root_node.children.append(current_node)
                last_2nd_level_node = current_node
            elif line.strip().startswith('#'):
                current_node.title = line.strip('#').strip()
                current_node.parent = None
                root_node = current_node

            current_node.text += line.strip() + ' '

        return root_node


with open("../cv/Extended_CV.md", "r", encoding="utf-8") as f:
    content = f.read()

splitter = CvSplitter(content)
tree = splitter.build_doc_tree()
print(tree)