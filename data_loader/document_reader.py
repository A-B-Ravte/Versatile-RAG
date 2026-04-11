import fitz
import os

class SingleDocument():
    def __init__(self, doc_path : str):
        self.doc_path = doc_path

    def read_document(self)-> list:
        if not os.path.isfile(self.doc_path):
            raise FileNotFoundError(f"Error : file {self.doc_path} not exists")

        doc = fitz.open(self.doc_path)

        pages = []

        for page_no, page in enumerate(doc, start=1):
            text = page.get_text('text')
            pages.append({
                'page_no':page_no,
                'text':text,
                'source': self.doc_path
            })   

        return pages

class MultipleDocuments():
    def __init__(self, directory_path: str):
        self_directory = directory_path

    def read_documents_from_directory(self)-> list :
        pass  