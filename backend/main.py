from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
from typing import Dict, Any
import os
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

app = FastAPI(title="Vegetation Analysis API")

# CORS設定
origins = [
    "http://localhost:3000",
    "https://Gucchyon.github.io",
    "https://Gucchyon.github.io/VegetationPython"
]

# main.py のCORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://gucchyon.github.io",
        "https://gucchyon.github.io/VegetationPython"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VegetationAnalysis:
    def __init__(self):
        self.algorithms = {
            'INT': lambda r, g, b: (r + g + b) / 3,
            'NRI': lambda r, g, b: np.divide(r, r + g + b, out=np.zeros_like(r), where=(r + g + b) != 0),
            'NGI': lambda r, g, b: np.divide(g, r + g + b, out=np.zeros_like(g), where=(r + g + b) != 0),
            'NBI': lambda r, g, b: np.divide(b, r + g + b, out=np.zeros_like(b), where=(r + g + b) != 0),
            'RGRI': lambda r, g, b: np.divide(r, g, out=np.zeros_like(r), where=g != 0),
            'ExR': lambda r, g, b: 1.4 * r - g,
            'ExG': lambda r, g, b: 2 * g - r - b,
            'ExB': lambda r, g, b: 1.4 * b - g,
            'ExGR': lambda r, g, b: (2 * g - r - b) - (1.4 * r - g),
            'GRVI': lambda r, g, b: np.divide(g - r, g + r, out=np.zeros_like(g), where=(g + r) != 0),
            'VARI': lambda r, g, b: np.divide(g - r, g + r - b, out=np.zeros_like(g), where=(g + r - b) != 0),
            'GLI': lambda r, g, b: np.divide(2 * g - r - b, 2 * g + r + b, out=np.zeros_like(g), where=(2 * g + r + b) != 0),
            'MGRVI': lambda r, g, b: np.divide(g**2 - r**2, g**2 + r**2, out=np.zeros_like(g), where=(g**2 + r**2) != 0),
            'RGBVI': lambda r, g, b: np.divide(g**2 - r*b, g**2 + r*b, out=np.zeros_like(g), where=(g**2 + r*b) != 0),
            'VEG': lambda r, g, b: np.divide(g, np.power(r, 0.667) * np.power(b, 0.333), out=np.zeros_like(g), 
                                           where=(r != 0) & (b != 0))
        }

    def normalize_rgb(self, img: np.ndarray) -> np.ndarray:
        """RGB値を正規化する"""
        float_img = img.astype(float)
        sum_rgb = np.sum(float_img, axis=2)
        normalized = np.zeros_like(float_img)
        for i in range(3):
            normalized[:, :, i] = np.divide(float_img[:, :, i], sum_rgb,
                                          out=np.zeros_like(float_img[:, :, i]),
                                          where=sum_rgb != 0)
        return normalized

    def calculate_otsu_threshold(self, exg: np.ndarray) -> float:
        """大津の方法で閾値を計算"""
        normalized_exg = ((exg + 2) * 127.5).astype(np.uint8)
        threshold, _ = cv2.threshold(normalized_exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (threshold / 127.5) - 2

    def process_image(self, img: np.ndarray, threshold_method: str = 'otsu',
                     manual_threshold: float = 0.0) -> Dict[str, Any]:
        """画像を解析して結果を返す"""
        try:
            # BGR to RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # RGB正規化
            normalized = self.normalize_rgb(img)
            r, g, b = normalized[:, :, 0], normalized[:, :, 1], normalized[:, :, 2]

            # ExG計算
            exg = self.algorithms['ExG'](r, g, b)

            # 閾値計算
            threshold = self.calculate_otsu_threshold(exg) if threshold_method == 'otsu' else manual_threshold

            # 植生マスク作成
            vegetation_mask = exg > threshold

            # 結果の集計
            result = {
                'vegetation_coverage': float(np.mean(vegetation_mask) * 100),
                'vegetation_pixels': int(np.sum(vegetation_mask)),
                'total_pixels': int(vegetation_mask.size),
                'threshold_method': threshold_method,
                'threshold_value': float(threshold),
                'indices': {
                    'vegetation': {},
                    'whole': {}
                }
            }

            # 各指数の計算
            for name, func in self.algorithms.items():
                index_values = func(r, g, b)
                result['indices']['vegetation'][name] = float(np.mean(index_values[vegetation_mask]))
                result['indices']['whole'][name] = float(np.mean(index_values))

            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

# APIエンドポイント
# main.py の analyze_image エンドポイントを修正
@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """画像を解析するエンドポイント"""
    try:
        print(f"Received file: {file.filename}")  # ファイル受信確認
        # 画像データの読み込み
        contents = await file.read()
        print(f"File size: {len(contents)} bytes")  # ファイルサイズ確認
        
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Failed to decode image")  # デコードエラーの確認
            raise HTTPException(status_code=400, detail="Invalid image file")

        print(f"Image shape: {img.shape}")  # 画像サイズ確認
        
        # 解析実行
        analyzer = VegetationAnalysis()
        result = analyzer.process_image(img)
        print(f"Analysis result: {result}")  # 結果確認
        
        return JSONResponse(content=result)

    except Exception as e:
        print(f"Error details: {str(e)}")  # エラー詳細の出力
        raise HTTPException(status_code=500, detail=str(e))
# ヘルスチェックエンドポイント
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)