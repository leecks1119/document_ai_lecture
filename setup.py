from setuptools import setup

setup(
    name="docai_course",
    version="2.0.0",
    description="초보자용 Document AI 8교시 실습 함수",
    url="https://github.com/leecks1119/document_ai_lecture",
    packages=["src"],
    install_requires=[
        "gradio==6.20.0",
        "jsonschema>=4.23,<5",
        "pandas>=2.0,<4",
        "Pillow>=10,<13",
        "PyMuPDF==1.28.0",
    ],
    extras_require={
        "ocr": ["paddleocr==3.7.0", "paddlepaddle>=3.2.1,<3.3"],
        "vlm": [
            "paddleocr[doc-parser]==3.7.0",
            "transformers>=5.8,<6",
        ],
    },
    python_requires=">=3.12,<3.13",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
)
