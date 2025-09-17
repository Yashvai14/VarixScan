"use client";

import { useState } from "react";
import Image from "next/image";
import axios from "axios";
import { getRequest, postRequest } from "@/lib/api";


import { 
  Upload, 
  User, 
  Calendar, 
  Activity, 
  AlertCircle, 
  CheckCircle, 
  Loader2,
  FileImage,
  Stethoscope,
  Download,
  FileText
} from "lucide-react";


export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [result, setResult] = useState<string>("");
  const [analysisData, setAnalysisData] = useState<{
    analysis_id: number;
    diagnosis: string;
    confidence: number;
    severity: string;
    detection_count?: number;
    affected_area_ratio?: number;
    recommendations?: string[];
  } | null>(null); 
  const [patientId, setPatientId] = useState<number | null>(null); 
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [error, setError] = useState<string>("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleAnalyze = async () => {
    if (!file || !name || !age || !gender) {
      setError("Please fill all details and upload an image!");
      return;
    }

    setIsAnalyzing(true);
    setError("");
    setResult("");

    const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "https://varixscan.onrender.com";

try {
  // Check if backend is available
  const backendCheck = await axios.get(`${BACKEND_URL}/health`, {
    timeout: 3000
  }).catch(() => null);
  
  if (!backendCheck) {
    // Backend not available - demo mode
    console.log("Backend not available, showing demo analysis...");
    await new Promise(resolve => setTimeout(resolve, 2000));
    const demoResults = [
      { diagnosis: "Mild Varicose Veins", confidence: 92, severity: "Mild" },
      { diagnosis: "No Abnormalities Detected", confidence: 98, severity: "Normal" },
      { diagnosis: "Moderate Varicose Veins", confidence: 87, severity: "Moderate" }
    ];
    const randomResult = demoResults[Math.floor(Math.random() * demoResults.length)];
    setResult(`Patient: ${name}\nDiagnosis: ${randomResult.diagnosis}\nConfidence: ${randomResult.confidence}%\nSeverity: ${randomResult.severity}\n\n⚠️ Demo Mode: Backend server not running. Start the backend for real analysis.`);
    return;
  }

  // Step 1: Create patient
  const patientData = {
    name: name,
    age: parseInt(age),
    gender: gender,
    phone: "+1234567890", 
    email: `${name.toLowerCase().replace(' ', '')}@example.com`
  };
  
  const patientRes = await axios.post(`${BACKEND_URL}/patients/`, patientData, {
    headers: { "Content-Type": "application/json" },
  });
  
  const patientIdValue = patientRes.data.patient_id;
  setPatientId(patientIdValue);
  
  // Step 2: Analyze image
  const formData = new FormData();
  formData.append("file", file);
  formData.append("patient_id", patientIdValue.toString());
  formData.append("language", "en");

  const res = await axios.post(`${BACKEND_URL}/analyze`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  
  setAnalysisData(res.data);
  setResult(`Patient: ${name}\nDiagnosis: ${res.data.diagnosis}\nConfidence: ${res.data.confidence}%\nSeverity: ${res.data.severity}`);
  
} catch (err: unknown) {
  console.error("Error:", err);

  if (axios.isAxiosError(err)) {
    if (err.response) {
      console.error("Response data:", err.response.data);
      setError(`Error: ${err.response.data?.detail || 'Analysis failed'}`);
    } else if (err.code === 'ECONNREFUSED' || err.message?.includes('Network Error')) {
      setError("Backend server not running. Please start the backend server to perform real analysis.");
    } else {
      setError(err.message || "Error analyzing image. Please try again.");
    }
  } else if (err instanceof Error) {
    setError(err.message || "Error analyzing image. Please try again.");
  } else {
    setError("Error analyzing image. Please try again.");
  }
} finally {
  setIsAnalyzing(false);
}
}

// Move generateReport outside of handleAnalyze
const generateReport = async () => {
  if (!patientId || !analysisData) {
    setError("Analysis data not available for report generation");
    return;
  }

  setIsGeneratingReport(true);
  setError("");

  try {
    const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "https://varixscan.onrender.com";
    const generateResponse = await axios.post(
      `${BACKEND_URL}/generate-report/${patientId}?analysis_id=${analysisData.analysis_id}&report_type=standard`
    );

    if (generateResponse.data) {
      const reportId = generateResponse.data.report_id;
      const downloadResponse = await fetch(
        `${BACKEND_URL}/download-report/${reportId}`,
        { 
          method: 'GET',
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        }
      );
      
      if (!downloadResponse.ok) {
        throw new Error('Failed to download report');
      }

      const blob = await downloadResponse.blob();
      const pdfBlob = new Blob([blob], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(pdfBlob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `VarixScan-Report-${name}-${reportId}.pdf`;
      a.rel = 'noopener';
      a.href += `#${Date.now()}`;
      document.body.appendChild(a);
      setTimeout(() => {
        a.click();
        setTimeout(() => {
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);
        }, 100);
      }, 100);
    }
  } catch (err: unknown) {
    console.error("Error generating report:", err);

    if (axios.isAxiosError(err)) {
      setError(err.message || "Failed to generate report. Please try again.");
    } else if (err instanceof Error) {
      setError(err.message || "Failed to generate report. Please try again.");
    } else {
      setError("Failed to generate report. Please try again.");
    }
  } finally {
    setIsGeneratingReport(false);
  }
};


  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light via-white to-vascular-light p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 pt-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-medical-primary rounded-full">
              <Stethoscope className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-medical-dark font-medical">
              AI Vein Detection System
            </h1>
          </div>
          <p className="text-medical-dark/70 text-lg max-w-2xl mx-auto">
            Advanced artificial intelligence technology for accurate varicose vein detection and analysis
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="bg-white shadow-medical-lg rounded-2xl p-8 border border-gray-100">
            <h2 className="text-xl font-semibold text-medical-dark mb-6 flex items-center gap-2">
              <User className="w-5 h-5 text-medical-primary" />
              Patient Information
            </h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-medical-dark mb-2">
                  Patient Name
                </label>
                <input
                  type="text"
                  placeholder="Enter patient name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-medical-primary focus:border-medical-primary transition-colors duration-200"
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-medical-dark mb-2">
                    <Calendar className="w-4 h-4 inline mr-1" />
                    Age
                  </label>
                  <input
                    type="number"
                    placeholder="Age"
                    value={age}
                    onChange={(e) => setAge(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-medical-primary focus:border-medical-primary transition-colors duration-200"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-medical-dark mb-2">
                    Gender
                  </label>
                  <select
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-medical-primary focus:border-medical-primary transition-colors duration-200"
                  >
                    <option value="">Select Gender</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="mt-8">
              <label className="block text-sm font-medium text-medical-dark mb-4">
                <FileImage className="w-4 h-4 inline mr-1" />
                Upload Medical Image
              </label>
              
              <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-medical-primary transition-colors duration-200">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload className="w-12 h-12 text-medical-primary mx-auto mb-4" />
                  <p className="text-medical-dark font-medium mb-2">Click to upload image</p>
                  <p className="text-gray-500 text-sm">Supports: JPG, PNG, GIF (Max 10MB)</p>
                </label>
              </div>

              {preview && (
                <div className="mt-6">
                  <div className="relative w-full h-64 rounded-xl border border-gray-200 shadow-medical overflow-hidden">
                    <Image
                      src={preview}
                      alt="Preview"
                      fill
                      className="object-cover"
                    />
                  </div>
                  <p className="text-sm text-gray-500 mt-2 text-center">Image ready for analysis</p>
                </div>
              )}
            </div>

            {error && (
              <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-red-500" />
                <p className="text-red-700 font-medium">{error}</p>
              </div>
            )}

            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing || !file || !name || !age || !gender}
              className="w-full mt-6 bg-medical-primary text-white py-4 px-6 rounded-xl font-semibold hover:bg-medical-secondary disabled:bg-gray-300 disabled:cursor-not-allowed transition-all duration-200 shadow-medical flex items-center justify-center gap-2"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Activity className="w-5 h-5" />
                  Analyze Image
                </>
              )}
            </button>
          </div>

          {/* Results Section */}
          <div className="bg-white shadow-medical-lg rounded-2xl p-8 border border-gray-100">
            <h2 className="text-xl font-semibold text-medical-dark mb-6 flex items-center gap-2">
              <Activity className="w-5 h-5 text-medical-success" />
              Analysis Results
            </h2>
            
            {!result && !isAnalyzing && (
              <div className="text-center py-12">
                <div className="w-20 h-20 bg-gray-100 rounded-full mx-auto mb-4 flex items-center justify-center">
                  <Stethoscope className="w-8 h-8 text-gray-400" />
                </div>
                <p className="text-gray-500">Upload an image and fill patient details to begin analysis</p>
              </div>
            )}
            
            {isAnalyzing && (
              <div className="text-center py-12">
                <Loader2 className="w-12 h-12 text-medical-primary animate-spin mx-auto mb-4" />
                <p className="text-medical-dark font-medium">Analyzing image...</p>
                <p className="text-gray-500 mt-2">This may take a few moments</p>
              </div>
            )}

            {result && (
              <div className="space-y-4">
                <div className="p-6 bg-gradient-to-r from-medical-light to-vascular-light rounded-xl border border-medical-primary/20">
                  <div className="flex items-center gap-2 mb-3">
                    <CheckCircle className="w-5 h-5 text-medical-success" />
                    <span className="font-semibold text-medical-dark">Analysis Complete</span>
                  </div>
                  <div className="whitespace-pre-line text-medical-dark leading-relaxed">
                    {result}
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <button 
                    onClick={generateReport}
                    disabled={isGeneratingReport || !analysisData}
                    className="flex items-center justify-center gap-2 bg-medical-accent text-white py-2 px-4 rounded-lg hover:bg-medical-primary disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors duration-200"
                  >
                    {isGeneratingReport ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <Download className="w-4 h-4" />
                        Download Report
                      </>
                    )}
                  </button>
                  <button 
                    onClick={() => window.location.href = '/reports'}
                    className="flex items-center justify-center gap-2 bg-gray-100 text-medical-dark py-2 px-4 rounded-lg hover:bg-gray-200 transition-colors duration-200"
                  >
                    <FileText className="w-4 h-4" />
                    View All Reports
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
