from setuptools import setup, find_packages

setup(
    name="docai_course",
    version="1.0.0",
    description="Document AI 강의용 Python 패키지",
    author="Chanhee Lee",
    author_email="leecks1119@gmail.com",
    url="https://github.com/leecks1119/document_ai_lecture",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pytesseract>=0.3.10",
        "paddlepaddle>=2.5.0",
        "paddleocr>=2.7.0",
        "easyocr>=1.7.0",
        "python-Levenshtein>=0.21.0",
        "matplotlib>=3.7.0",
        "scikit-image>=0.21.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

