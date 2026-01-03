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
            print(repr(line))
            if line.strip().startswith('###'):
                current_node = CVNode()
                current_node.text = ''
                current_node.title = line.strip('###').strip()
                current_node.parent = last_2nd_level_node
            elif line.strip().startswith('##'):
                current_node = CVNode()
                current_node.text = ''
                current_node.title = line.strip('##').strip()
                current_node.parent = root_node
                last_2nd_level_node = current_node
            elif line.strip().startswith('#'):
                current_node = CVNode()
                current_node.text = ''
                current_node.title = line.strip('#').strip()
                current_node.parent = None
                root_node = current_node

            current_node.text += line.strip() + ' '

        return root_node


with open("../cv/Extended_CV.md", "r", encoding="utf-8") as f:
    content = f.read()

splitter = CvSplitter(content)
print(splitter.build_doc_tree())