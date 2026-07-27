"""이미지 전처리 도구"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any


class DocumentPreprocessor:
    """문서 이미지 전처리 클래스"""
    
    def __init__(self, debug_mode: bool = False):
        """
        Args:
            debug_mode: 디버그 모드 (중간 결과 출력)
        """
        self.debug_mode = debug_mode
    
    def adaptive_preprocessing_pipeline(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        적응형 전처리 파이프라인
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            (전처리된 이미지, 이진화 이미지, 품질 지표)
        """
        # 이미지 로드
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 적응형 이진화
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 기울기 보정
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.5:
                (h, w) = binary.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                binary = cv2.warpAffine(binary, M, (w, h), 
                                       flags=cv2.INTER_CUBIC, 
                                       borderMode=cv2.BORDER_REPLICATE)
                denoised = cv2.warpAffine(denoised, M, (w, h),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
        
        # 품질 지표 계산
        quality_metrics = {
            'mean_intensity': np.mean(gray),
            'std_intensity': np.std(gray),
            'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var()
        }
        
        return denoised, binary, quality_metrics

