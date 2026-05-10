// API Service functions using standard fetch
import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
});

export default api;

export interface Patient {
  id: number;
  name: string;
  age: number;
  gender: string;
  phone?: string;
  email?: string;
  created_at?: string;
}

export interface Analysis {
  id: number;
  patient_id: number;
  diagnosis: string;
  severity: string;
  confidence: number;
  detection_count?: number;
  affected_area_ratio?: number;
  image_path?: string;
  recommendations?: string[];
  created_at: string;
}

export interface Report {
  id: number;
  patient_id: number;
  analysis_id: number;
  report_type: string;
  pdf_path: string;
  created_at: string;
}

export interface ChatMessage {
  id: number;
  session_id: string;
  user_message: string;
  ai_response: string;
  language: string;
  created_at: string;
}

export const patientService = {
  async createPatient(patientData: Omit<Patient, 'id' | 'created_at'>) {
    const res = await fetch(`${API_BASE_URL}/patients/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patientData)
    });
    if (!res.ok) throw new Error('Failed to create patient');
    return res.json();
  },

  async getPatients() {
    const res = await fetch(`${API_BASE_URL}/api/patients/`);
    if (!res.ok) throw new Error('Failed to fetch patients');
    return res.json();
  },

  async getPatientAnalyses(patientId: number) {
    const res = await fetch(`${API_BASE_URL}/api/patients/${patientId}/analyses`);
    if (!res.ok) throw new Error('Failed to fetch patient analyses');
    return res.json();
  }
};

export const analysisService = {
  async getAnalyses() {
    const res = await fetch(`${API_BASE_URL}/api/analyses`);
    if (!res.ok) throw new Error('Failed to fetch analyses');
    return res.json();
  }
};

export const reportService = {
  async getReports() {
    const res = await fetch(`${API_BASE_URL}/api/reports`);
    if (!res.ok) throw new Error('Failed to fetch reports');
    return res.json();
  },

  async getPatientReports(patientId: number) {
    const res = await fetch(`${API_BASE_URL}/api/patients/${patientId}/reports`);
    if (!res.ok) throw new Error('Failed to fetch patient reports');
    return res.json();
  }
};

export const dashboardService = {
  async getDashboardStats() {
    try {
      const res = await fetch(`${API_BASE_URL}/api/dashboard/stats`);
      if (!res.ok) throw new Error('Failed to fetch dashboard stats');
      return res.json();
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
      return { totalPatients: 0, totalAnalyses: 0, recentAnalyses: 0, avgConfidence: 0 };
    }
  },

  async getRecentAnalyses(limit = 10) {
    try {
      const res = await fetch(`${API_BASE_URL}/api/dashboard/recent-analyses?limit=${limit}`);
      if (!res.ok) throw new Error('Failed to fetch recent analyses');
      return res.json();
    } catch (error) {
      console.error('Error fetching recent analyses:', error);
      return [];
    }
  }
};

export const chatService = {
  async getChatHistory(sessionId: string, limit = 50) {
    const res = await fetch(`${API_BASE_URL}/api/chat/history/${sessionId}?limit=${limit}`);
    if (!res.ok) throw new Error('Failed to fetch chat history');
    return res.json();
  },

  async sendMessage(message: string, sessionId: string, language = 'en') {
    const res = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId, language })
    });
    if (!res.ok) throw new Error('Failed to send message');
    return res.json();
  }
};
