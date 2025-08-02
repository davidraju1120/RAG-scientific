import re
from typing import List, Dict

class CitationNetwork:
    def __init__(self):
        self.network = {}

    def add_paper(self, paper_id: str, references: List[str]):
        self.network[paper_id] = references

    def get_citations(self, paper_id: str) -> List[str]:
        return self.network.get(paper_id, [])

    def build_from_context(self, context: List[Dict]):
        for doc in context:
            pid = doc.get('doc_id') or doc.get('source_id')
            refs = self.extract_references(doc.get('content', ''))
            if pid:
                self.add_paper(str(pid), refs)

    @staticmethod
    def extract_references(text: str) -> List[str]:
        # Simple regex for [n] style citations
        return re.findall(r'\[(\d+)\]', text)
