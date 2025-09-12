"use client";

import { useEffect, useState } from "react";
import { Download, FileText, Calendar, User, Activity } from "lucide-react";
import { reportService, type Report } from '../../lib/supabase';

interface ReportWithPatient extends Report {
  patients?: {
    id: number;
    name: string;
  };
  analyses?: {
    id: number;
    diagnosis: string;
    severity: string;
  };
}

export default function ReportsPage() {
  const [reports, setReports] = useState<ReportWithPatient[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchReports = async () => {
      try {
        setLoading(true);
        setError(null);
        const reportsData = await reportService.getReports();
        setReports(reportsData);
      } catch (err) {
        console.error('Error fetching reports:', err);
        setError('Failed to load reports. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    fetchReports();
  }, []);

  const downloadPDF = async (report: ReportWithPatient) => {
    try {
      // Check if we need to generate the report first
      if (!report.pdf_path) {
        // Generate report using backend API
        const generateResponse = await fetch(
          `http://localhost:8000/generate-report/${report.patient_id}?analysis_id=${report.analysis_id}&report_type=${report.report_type}`,
          { method: 'POST' }
        );
        
        if (!generateResponse.ok) {
          throw new Error('Failed to generate report');
        }
        
        const generateData = await generateResponse.json();
        report.id = generateData.report_id; // Update with the new report ID
      }
      
      // Download the generated PDF with headers to prevent IDM interception
      const downloadResponse = await fetch(`http://localhost:8000/download-report/${report.id}`, {
        method: 'GET',
        headers: {
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        }
      });
      
      if (!downloadResponse.ok) {
        throw new Error('Failed to download report');
      }
      
      // Get the filename from Content-Disposition header if available
      const contentDisposition = downloadResponse.headers.get('Content-Disposition');
      let filename = `VarixScan-Report-${report.patients?.name || 'Patient'}-${report.id}.pdf`;
      
      if (contentDisposition && contentDisposition.includes('filename=')) {
        const matches = contentDisposition.match(/filename[^;=\n]*=(['"]*)([^'"\n]*?)\1/);
        if (matches && matches[2]) {
          filename = matches[2];
        }
      }
      
      // Create blob with explicit PDF mime type and download
      const blob = await downloadResponse.blob();
      const pdfBlob = new Blob([blob], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(pdfBlob);
      
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = filename;
      a.rel = 'noopener'; // Security best practice
      
      // Add timestamp to prevent caching issues
      a.href += `#${Date.now()}`;
      
      document.body.appendChild(a);
      
      // Force click with timeout to ensure DOM is ready
      setTimeout(() => {
        a.click();
        setTimeout(() => {
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);
        }, 100);
      }, 100);
      
    } catch (error) {
      console.error('Error downloading PDF:', error);
      alert('Failed to download PDF report. Please try again.');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-medical-light via-white to-vascular-light p-6">
        <div className="max-w-7xl mx-auto">
          <div className="bg-white rounded-2xl shadow-medical-lg p-8">
            <div className="flex items-center justify-center py-12">
              <div className="text-center">
                <Activity className="w-12 h-12 text-medical-primary mx-auto mb-4 animate-spin" />
                <p className="text-xl text-medical-dark">Loading reports...</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light via-white to-vascular-light p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-3 bg-medical-primary rounded-full">
              <FileText className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-medical-dark font-medical">
              Medical Reports
            </h1>
          </div>
          <p className="text-medical-dark/70">
            Complete history of all generated patient reports and analysis results
          </p>
        </div>

        {/* Main Content */}
        <div className="bg-white rounded-2xl shadow-medical-lg p-8 border border-gray-100">
          {error ? (
            <div className="text-center py-12">
              <div className="p-4 bg-red-50 rounded-lg border border-red-200 inline-block">
                <p className="text-red-600 font-medium">{error}</p>
                <button 
                  onClick={() => window.location.reload()} 
                  className="mt-2 text-red-500 hover:text-red-700 underline"
                >
                  Try Again
                </button>
              </div>
            </div>
          ) : reports.length === 0 ? (
            <div className="text-center py-12">
              <FileText className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-gray-600 mb-2">No Reports Available</h3>
              <p className="text-gray-500 mb-4">Reports will appear here after analyzing patient images</p>
              <button className="bg-medical-primary text-white px-6 py-2 rounded-lg hover:bg-medical-secondary transition-colors duration-200">
                Start Analysis
              </button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-4 px-4 text-sm font-semibold text-gray-700">
                      <div className="flex items-center gap-2">
                        <FileText className="w-4 h-4" />
                        Report ID
                      </div>
                    </th>
                    <th className="text-left py-4 px-4 text-sm font-semibold text-gray-700">
                      <div className="flex items-center gap-2">
                        <User className="w-4 h-4" />
                        Patient
                      </div>
                    </th>
                    <th className="text-left py-4 px-4 text-sm font-semibold text-gray-700">
                      Report Type
                    </th>
                    <th className="text-left py-4 px-4 text-sm font-semibold text-gray-700">
                      Analysis
                    </th>
                    <th className="text-left py-4 px-4 text-sm font-semibold text-gray-700">
                      <div className="flex items-center gap-2">
                        <Calendar className="w-4 h-4" />
                        Generated
                      </div>
                    </th>
                    <th className="text-left py-4 px-4 text-sm font-semibold text-gray-700">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {reports.map((report) => (
                    <tr key={report.id} className="border-b border-gray-50 hover:bg-gray-50 transition-colors duration-200">
                      <td className="py-4 px-4">
                        <span className="font-mono text-medical-primary font-medium">
                          R{String(report.id).padStart(3, '0')}
                        </span>
                      </td>
                      <td className="py-4 px-4">
                        <span className="font-medium text-medical-dark">
                          {report.patients?.name || 'Unknown Patient'}
                        </span>
                      </td>
                      <td className="py-4 px-4">
                        <span className="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800 capitalize">
                          {report.report_type}
                        </span>
                      </td>
                      <td className="py-4 px-4">
                        {report.analyses && (
                          <div>
                            <div className="text-sm font-medium text-medical-dark">
                              {report.analyses.diagnosis}
                            </div>
                            <div className="text-xs text-gray-500">
                              {report.analyses.severity} Severity
                            </div>
                          </div>
                        )}
                      </td>
                      <td className="py-4 px-4 text-sm text-gray-600">
                        {new Date(report.created_at).toLocaleDateString('en-US', {
                          year: 'numeric',
                          month: 'short',
                          day: 'numeric'
                        })}
                      </td>
                      <td className="py-4 px-4">
                        <button
                          onClick={() => downloadPDF(report)}
                          className="flex items-center gap-2 bg-medical-primary text-white px-4 py-2 rounded-lg hover:bg-medical-secondary transition-colors duration-200 text-sm"
                        >
                          <Download className="w-4 h-4" />
                          Download PDF
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
