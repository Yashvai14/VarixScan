"use client";

import { useEffect, useState } from "react";
import axios from "axios";
import { jsPDF } from "jspdf";

interface Report {
  id: number;
  name: string;
  age: string;
  gender: string;
  diagnosis: string;
  confidence: number;
  date: string;
}

export default function ReportsPage() {
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchReports = async () => {
      try {
        const res = await axios.get("http://localhost:8000/reports");
        setReports(res.data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchReports();
  }, []);

  const downloadPDF = (report: Report) => {
    const doc = new jsPDF();
    doc.setFontSize(18);
    doc.text("VeinSight Diagnostic Report", 20, 20);

    doc.setFontSize(12);
    doc.text(`Patient Name: ${report.name}`, 20, 40);
    doc.text(`Age: ${report.age}`, 20, 50);
    doc.text(`Gender: ${report.gender}`, 20, 60);
    doc.text(`Date: ${report.date}`, 20, 70);

    doc.text("Diagnosis Result:", 20, 90);
    doc.text(`${report.diagnosis} (Confidence: ${report.confidence}%)`, 20, 100);

    doc.save(`${report.name}-report.pdf`);
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-white text-xl">
        Loading reports...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-r from-violet-500 to-blue-500 p-6">
      <div className="bg-white shadow-xl rounded-2xl p-8 w-full max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold text-center mb-6 text-gray-800">
          Patient Reports
        </h1>

        {reports.length === 0 ? (
          <p className="text-center text-gray-600">No reports available yet.</p>
        ) : (
          <table className="w-full border-collapse">
            <thead>
              <tr className="bg-gray-200 text-gray-700">
                <th className="p-2">Name</th>
                <th className="p-2">Age</th>
                <th className="p-2">Gender</th>
                <th className="p-2">Diagnosis</th>
                <th className="p-2">Confidence</th>
                <th className="p-2">Date</th>
                <th className="p-2">Action</th>
              </tr>
            </thead>
            <tbody>
              {reports.map((report) => (
                <tr key={report.id} className="text-center border-b">
                  <td className="p-2">{report.name}</td>
                  <td className="p-2">{report.age}</td>
                  <td className="p-2">{report.gender}</td>
                  <td className="p-2">{report.diagnosis}</td>
                  <td className="p-2">{report.confidence}%</td>
                  <td className="p-2">{report.date}</td>
                  <td className="p-2">
                    <button
                      onClick={() => downloadPDF(report)}
                      className="bg-blue-600 text-white px-3 py-1 rounded-xl shadow hover:bg-blue-700 transition"
                    >
                      PDF
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
