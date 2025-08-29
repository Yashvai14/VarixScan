"use client";

import { useState } from "react";
import axios from "axios";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [result, setResult] = useState<string>("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleAnalyze = async () => {
    if (!file || !name || !age || !gender) {
      alert("Please fill all details and upload an image!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", name);
    formData.append("age", age);
    formData.append("gender", gender);

    try {
      const res = await axios.post("http://localhost:8000/analyze", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data.diagnosis + ` (Confidence: ${res.data.confidence}%)`);
    } catch (err) {
      console.error(err);
      alert("Error analyzing image");
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-r from-violet-500 to-blue-500 p-6">
      <div className="bg-white shadow-xl rounded-2xl p-8 w-full max-w-lg">
        <h1 className="text-2xl font-bold text-center mb-6 text-gray-800">
          Varicose Vein Detection
        </h1>

        {/* Patient Info */}
        <input
          type="text"
          placeholder="Patient Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full mb-3 border p-2 rounded"
        />
        <input
          type="number"
          placeholder="Age"
          value={age}
          onChange={(e) => setAge(e.target.value)}
          className="w-full mb-3 border p-2 rounded"
        />
        <select
          value={gender}
          onChange={(e) => setGender(e.target.value)}
          className="w-full mb-3 border p-2 rounded"
        >
          <option value="">Select Gender</option>
          <option value="Male">Male</option>
          <option value="Female">Female</option>
        </select>

        {/* File Upload */}
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="mb-4 w-full"
        />

        {preview && (
          <img
            src={preview}
            alt="Preview"
            className="w-full h-64 object-cover rounded-xl mb-4"
          />
        )}

        <button
          onClick={handleAnalyze}
          className="w-full bg-violet-600 text-white py-2 rounded-xl shadow hover:bg-violet-700 transition"
        >
          Analyze Image
        </button>

        {result && (
          <div className="mt-6 p-4 bg-gray-100 rounded-xl">
            <p className="text-gray-800 font-medium">{result}</p>
          </div>
        )}
      </div>
    </div>
  );
}
