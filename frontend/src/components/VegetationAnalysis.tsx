import React, { useState } from 'react';
import axios, { AxiosResponse } from 'axios';

// 型定義の部分を修正・追加
interface AnalysisResult {
  vegetation_coverage: number;
  vegetation_pixels: number;
  total_pixels: number;
  threshold_method: string;
  threshold_value: number;
  indices: {
    vegetation: Record<string, number>;
    whole: Record<string, number>;
  };
}

// 解析実行部分を修正
const handleAnalyze = async () => {
    if (!file) return;

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response: AxiosResponse<AnalysisResult> = await axios.post(
        `${process.env.REACT_APP_API_URL}/api/analyze`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      setResult(response.data);
    } catch (error) {
      console.error('Analysis error:', error);
      alert('Error analyzing image');
    } finally {
      setIsProcessing(false);
    }
};

const VegetationAnalysis: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [language, setLanguage] = useState<'ja' | 'en'>('ja');

  // 言語設定
  const translations = {
    ja: {
      title: "植生解析",
      upload: "画像をアップロード",
      analyze: "解析開始",
      processing: "処理中...",
      results: {
        coverage: "植生被覆率",
        pixels: "植生ピクセル数",
        total: "総ピクセル数",
        indices: "植生指数",
        vegetation: "植生部分の指数値",
        whole: "画像全体の指数値"
      }
    },
    en: {
      title: "Vegetation Analysis",
      upload: "Upload Image",
      analyze: "Start Analysis",
      processing: "Processing...",
      results: {
        coverage: "Vegetation Coverage",
        pixels: "Vegetation Pixels",
        total: "Total Pixels",
        indices: "Vegetation Indices",
        vegetation: "Vegetation Area Indices",
        whole: "Whole Image Indices"
      }
    }
  };

  const t = translations[language];

  // 画像アップロード処理
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      const reader = new FileReader();
      reader.onload = (e) => {
        setOriginalImage(e.target?.result as string);
      };
      reader.readAsDataURL(uploadedFile);
    }
  };

  // 解析実行
  const handleAnalyze = async () => {
    if (!file) return;

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(
        `${process.env.REACT_APP_API_URL}/api/analyze`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      setResult(response.data);
    } catch (error) {
      console.error('Analysis error:', error);
      alert('Error analyzing image');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-4 space-y-6 bg-white rounded-lg shadow">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">{t.title}</h1>
        <button
          onClick={() => setLanguage(prev => prev === 'ja' ? 'en' : 'ja')}
          className="px-4 py-2 bg-gray-100 rounded hover:bg-gray-200"
        >
          {language === 'ja' ? 'English' : '日本語'}
        </button>
      </div>

      <div className="space-y-4">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileUpload}
          className="block w-full text-sm text-gray-500
            file:mr-4 file:py-2 file:px-4
            file:rounded-full file:border-0
            file:text-sm file:font-semibold
            file:bg-blue-50 file:text-blue-700
            hover:file:bg-blue-100"
        />

        <button
          onClick={handleAnalyze}
          disabled={!file || isProcessing}
          className="w-full py-2 px-4 bg-blue-500 text-white rounded 
            hover:bg-blue-600 disabled:bg-gray-300"
        >
          {isProcessing ? t.processing : t.analyze}
        </button>

        {originalImage && (
          <div>
            <h3 className="text-lg font-medium mb-2">Original Image</h3>
            <img
              src={originalImage}
              alt="Original"
              className="w-full h-auto rounded shadow-md"
            />
          </div>
        )}

        {result && (
          <div className="p-4 bg-gray-50 rounded">
            <h3 className="text-lg font-medium mb-2">{t.results.indices}</h3>
            <div className="space-y-2">
              <p>{t.results.coverage}: {result.vegetation_coverage.toFixed(2)}%</p>
              <p>{t.results.pixels}: {result.vegetation_pixels.toLocaleString()}</p>
              <p>{t.results.total}: {result.total_pixels.toLocaleString()}</p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div>
                  <h4 className="font-medium mb-2">{t.results.vegetation}</h4>
                  {Object.entries(result.indices.vegetation).map(([key, value]) => (
                    <p key={key} className="text-sm">
                      {key}: {value.toFixed(4)}
                    </p>
                  ))}
                </div>

                <div>
                  <h4 className="font-medium mb-2">{t.results.whole}</h4>
                  {Object.entries(result.indices.whole).map(([key, value]) => (
                    <p key={key} className="text-sm">
                      {key}: {value.toFixed(4)}
                    </p>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default VegetationAnalysis;