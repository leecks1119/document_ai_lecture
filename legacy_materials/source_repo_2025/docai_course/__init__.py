"""
Document AI 강의용 Python 패키지
"""

__version__ = "1.0.0"

from docai_course.ocr.benchmark import OCRBenchmark
from docai_course.preprocessing.preprocessor import DocumentPreprocessor
from docai_course.ner.unified_ner import UnifiedNERSystem
from docai_course.hybrid.system import HybridDocumentAI

__all__ = [
    "OCRBenchmark",
    "DocumentPreprocessor",
    "UnifiedNERSystem",
    "HybridDocumentAI",
]

