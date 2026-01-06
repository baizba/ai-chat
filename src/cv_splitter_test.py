from cv_splitter import CvSplitter

def test_simple_content():
    # prepare
    content = """
    # My CV
    
    ## Education
    
    ### School
    Went to school
    
    ### University
    University in Zrenjanin
    
    ## Experience
    Company 1 - good projects
    Company 2 - bad projects
    
    ## Highlights
    Communicates with Cats.
    Talks to cows.
    """

    # tests
    splitter = CvSplitter(content)
    tree = splitter.build_doc_tree()

    #verify
    assert len(tree.children) == 3
    assert tree.title == "# My CV"

    assert tree.children[0].title == "## Education"

    assert tree.children[0].children[0].title == "### School"
    assert tree.children[0].children[0].text.strip() == "Went to school"

    assert tree.children[0].children[1].title == "### University"
    assert tree.children[0].children[1].text.strip() == "University in Zrenjanin"

    assert tree.children[1].title == "## Experience"
    assert tree.children[1].text.strip() == 'Company 1 - good projects Company 2 - bad projects'

    assert tree.children[2].title == "## Highlights"
    assert tree.children[2].text.strip() == 'Communicates with Cats. Talks to cows.'

test_simple_content()