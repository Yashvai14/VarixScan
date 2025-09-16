"use client";

import React, { useState, useEffect } from "react";
import { 
  BarChart3, 
  TrendingUp, 
  Users, 
  Activity,
  Calendar,
  FileText,
  Heart,
  Shield,
  Clock,
  AlertTriangle,
  CheckCircle,
  Eye,
  Download
} from "lucide-react";
import { dashboardService, type Analysis, type Patient } from '../../lib/supabase';

interface DashboardStats {
  totalPatients: number;
  totalAnalyses: number;
  recentAnalyses: number;
  avgConfidence: number;
}

interface RecentAnalysis extends Analysis {
  patients?: Patient;
}

export default function DashboardPage() {
  const [timeRange, setTimeRange] = useState("7d");
  const [stats, setStats] = useState<DashboardStats>({ 
    totalPatients: 0, 
    totalAnalyses: 0, 
    recentAnalyses: 0, 
    avgConfidence: 0 
  });
  const [recentAnalyses, setRecentAnalyses] = useState<RecentAnalysis[]>([]);
  const [loading, setLoading] = useState(true);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
    
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        const [dashboardStats, analysesData] = await Promise.all([
          dashboardService.getDashboardStats(),
          dashboardService.getRecentAnalyses(10)
        ]);
        
        setStats(dashboardStats);
        setRecentAnalyses(analysesData);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [timeRange]);

  const statsCards = [
    {
      title: "Total Scans",
      value: stats.totalAnalyses.toString(),
      change: "+" + Math.floor((stats.recentAnalyses / Math.max(stats.totalAnalyses, 1)) * 100) + "%",
      trend: "up",
      icon: <Activity className="w-6 h-6" />,
      color: "text-blue-600"
    },
    {
      title: "Active Patients",
      value: stats.totalPatients.toString(),
      change: "+5%", 
      trend: "up",
      icon: <Users className="w-6 h-6" />,
      color: "text-green-600"
    },
    {
      title: "Accuracy Rate",
      value: stats.avgConfidence.toString() + "%",
      change: "+0.2%",
      trend: "up",
      icon: <Shield className="w-6 h-6" />,
      color: "text-purple-600"
    },
    {
      title: "Recent Analyses",
      value: stats.recentAnalyses.toString(),
      change: "-15%",
      trend: "down",
      icon: <Clock className="w-6 h-6" />,
      color: "text-orange-600"
    }
  ];

  // Note: recentAnalyses is now managed by state (line 40) and populated from Supabase

  const alerts = [
    {
      type: "warning",
      message: "3 patients require follow-up consultation",
      time: "2 hours ago"
    },
    {
      type: "info", 
      message: "Weekly report generated successfully",
      time: "1 day ago"
    },
    {
      type: "success",
      message: "AI model accuracy improved to 95.8%",
      time: "2 days ago"
    }
  ];


  const getAlertIcon = (type: string) => {
    switch (type) {
      case "warning":
        return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      case "success":
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      default:
        return <Activity className="w-5 h-5 text-blue-500" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light via-white to-vascular-light p-4 sm:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 sm:mb-8">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 lg:gap-8">
            <div className="flex-1">
              <div className="flex flex-col sm:flex-row sm:items-center gap-3 mb-3 sm:mb-2">
                <div className="p-2 sm:p-3 bg-medical-primary rounded-full w-fit">
                  <BarChart3 className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
                </div>
                <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-medical-dark font-medical">
                  Medical Dashboard
                </h1>
              </div>
              <p className="text-sm sm:text-base text-medical-dark/70 max-w-2xl">
                Overview of your vascular health platform analytics and patient management
              </p>
            </div>

            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 sm:gap-3">
              <select 
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
                className="px-3 sm:px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-medical-primary focus:border-medical-primary text-sm sm:text-base"
              >
                <option value="7d">Last 7 days</option>
                <option value="30d">Last 30 days</option>
                <option value="90d">Last 3 months</option>
              </select>
              
              <button className="flex items-center justify-center gap-2 bg-medical-primary text-white px-3 sm:px-4 py-2 rounded-lg hover:bg-medical-secondary transition-colors duration-200 text-sm sm:text-base">
                <Download className="w-3 h-3 sm:w-4 sm:h-4" />
                <span className="hidden sm:inline">Export Report</span>
                <span className="sm:hidden">Export</span>
              </button>
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 mb-6 sm:mb-8">
          {loading ? (
            // Loading skeleton
            [...Array(4)].map((_, index) => (
              <div key={index} className="bg-white p-4 sm:p-6 rounded-xl sm:rounded-2xl shadow-medical border border-gray-100 animate-pulse">
                <div className="h-5 sm:h-6 bg-gray-200 rounded mb-3 sm:mb-4"></div>
                <div className="h-6 sm:h-8 bg-gray-200 rounded mb-2"></div>
                <div className="h-3 sm:h-4 bg-gray-200 rounded w-1/2"></div>
              </div>
            ))
          ) : (
            statsCards.map((stat, index) => (
              <div key={index} className="bg-white p-4 sm:p-6 rounded-xl sm:rounded-2xl shadow-medical border border-gray-100 hover:shadow-medical-lg transition-shadow duration-200">
                <div className="flex items-center justify-between mb-3 sm:mb-4">
                  <div className={`p-2 rounded-lg bg-gray-50 ${stat.color}`}>
                    {React.cloneElement(stat.icon, { className: "w-4 h-4 sm:w-5 sm:h-5 lg:w-6 lg:h-6" })}
                  </div>
                  <div className={`text-xs sm:text-sm font-medium flex items-center gap-1 ${
                    stat.trend === 'up' ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {stat.trend === 'up' ? (
                      <TrendingUp className="w-3 h-3 sm:w-4 sm:h-4" />
                    ) : (
                      <TrendingUp className="w-3 h-3 sm:w-4 sm:h-4 rotate-180" />
                    )}
                    {stat.change}
                  </div>
                </div>
                
                <div className="text-xl sm:text-2xl lg:text-3xl font-bold text-medical-dark mb-1">{stat.value}</div>
                <div className="text-xs sm:text-sm text-gray-600">{stat.title}</div>
              </div>
            ))
          )}
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 lg:gap-8">
          {/* Recent Analyses */}
          <div className="xl:col-span-2">
            <div className="bg-white rounded-xl sm:rounded-2xl shadow-medical-lg p-4 sm:p-6 border border-gray-100">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 sm:gap-0 mb-4 sm:mb-6">
                <h2 className="text-lg sm:text-xl font-semibold text-medical-dark flex items-center gap-2">
                  <FileText className="w-4 h-4 sm:w-5 sm:h-5 text-medical-primary" />
                  Recent Analyses
                </h2>
                <button className="text-medical-primary hover:text-medical-secondary text-sm font-medium w-fit">
                  View All
                </button>
              </div>

              {/* Mobile Card View - Hidden on larger screens */}
              <div className="block lg:hidden space-y-3">
                {loading ? (
                  [...Array(3)].map((_, index) => (
                    <div key={index} className="bg-gray-50 p-4 rounded-lg animate-pulse">
                      <div className="flex justify-between items-start mb-2">
                        <div className="h-4 bg-gray-200 rounded w-16"></div>
                        <div className="h-6 bg-gray-200 rounded w-20"></div>
                      </div>
                      <div className="h-4 bg-gray-200 rounded w-32 mb-2"></div>
                      <div className="h-4 bg-gray-200 rounded w-24"></div>
                    </div>
                  ))
                ) : recentAnalyses.length === 0 ? (
                  <div className="py-8 text-center text-gray-500">
                    No analyses found. Start by analyzing some images!
                  </div>
                ) : (
                  recentAnalyses.map((analysis) => (
                    <div key={analysis.id} className="bg-gray-50 p-4 rounded-lg hover:bg-gray-100 transition-colors duration-200">
                      <div className="flex justify-between items-start mb-2">
                        <span className="text-sm font-mono text-medical-primary font-semibold">
                          A{String(analysis.id).padStart(3, '0')}
                        </span>
                        <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800">
                          Completed
                        </span>
                      </div>
                      <div className="text-sm font-medium text-medical-dark mb-1">
                        {analysis.patients?.name || 'Unknown Patient'}
                      </div>
                      <div className="text-xs text-gray-600 mb-2" suppressHydrationWarning>
                        {isClient ? new Date(analysis.created_at).toLocaleDateString() : 'Loading...'}
                      </div>
                      <div className="flex justify-between items-center">
                        <div className="text-sm text-medical-dark">{analysis.diagnosis}</div>
                        <div className="flex items-center gap-3">
                          <span className={`text-sm font-medium ${
                            analysis.confidence >= 90 ? 'text-green-600' :
                            analysis.confidence >= 80 ? 'text-yellow-600' : 'text-red-600'
                          }`}>
                            {Math.round(analysis.confidence)}%
                          </span>
                          <button className="text-medical-primary hover:text-medical-secondary">
                            <Eye className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
              
              {/* Desktop Table View - Hidden on mobile */}
              <div className="hidden lg:block overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-100">
                      <th className="text-left py-3 text-sm font-medium text-gray-600">ID</th>
                      <th className="text-left py-3 text-sm font-medium text-gray-600">Patient</th>
                      <th className="text-left py-3 text-sm font-medium text-gray-600">Date</th>
                      <th className="text-left py-3 text-sm font-medium text-gray-600">Result</th>
                      <th className="text-left py-3 text-sm font-medium text-gray-600">Confidence</th>
                      <th className="text-left py-3 text-sm font-medium text-gray-600">Status</th>
                      <th className="text-left py-3 text-sm font-medium text-gray-600">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {loading ? (
                      // Loading skeleton for table rows
                      [...Array(5)].map((_, index) => (
                        <tr key={index} className="border-b border-gray-50 animate-pulse">
                          <td className="py-4"><div className="h-4 bg-gray-200 rounded w-16"></div></td>
                          <td className="py-4"><div className="h-4 bg-gray-200 rounded w-20"></div></td>
                          <td className="py-4"><div className="h-4 bg-gray-200 rounded w-24"></div></td>
                          <td className="py-4"><div className="h-4 bg-gray-200 rounded w-32"></div></td>
                          <td className="py-4"><div className="h-4 bg-gray-200 rounded w-12"></div></td>
                          <td className="py-4"><div className="h-6 bg-gray-200 rounded w-20"></div></td>
                          <td className="py-4"><div className="h-4 bg-gray-200 rounded w-4"></div></td>
                        </tr>
                      ))
                    ) : recentAnalyses.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="py-8 text-center text-gray-500">
                          No analyses found. Start by analyzing some images!
                        </td>
                      </tr>
                    ) : (
                      recentAnalyses.map((analysis) => (
                        <tr key={analysis.id} className="border-b border-gray-50 hover:bg-gray-50">
                          <td className="py-4 text-sm font-mono text-medical-primary">A{String(analysis.id).padStart(3, '0')}</td>
                          <td className="py-4 text-sm text-medical-dark font-medium">
                            {analysis.patients?.name || 'Unknown Patient'}
                          </td>
                          <td className="py-4 text-sm text-gray-600" suppressHydrationWarning>
                            {isClient ? new Date(analysis.created_at).toLocaleDateString() : 'Loading...'}
                          </td>
                          <td className="py-4 text-sm text-medical-dark">{analysis.diagnosis}</td>
                          <td className="py-4 text-sm">
                            <span className={`font-medium ${
                              analysis.confidence >= 90 ? 'text-green-600' :
                              analysis.confidence >= 80 ? 'text-yellow-600' : 'text-red-600'
                            }`}>
                              {Math.round(analysis.confidence)}%
                            </span>
                          </td>
                          <td className="py-4">
                            <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800">
                              Completed
                            </span>
                          </td>
                          <td className="py-4">
                            <button className="text-medical-primary hover:text-medical-secondary">
                              <Eye className="w-4 h-4" />
                            </button>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-4 sm:space-y-6">
            {/* Quick Actions */}
            <div className="bg-white rounded-xl sm:rounded-2xl shadow-medical-lg p-4 sm:p-6 border border-gray-100">
              <h3 className="text-base sm:text-lg font-semibold text-medical-dark mb-3 sm:mb-4">Quick Actions</h3>
              
              <div className="space-y-2 sm:space-y-3">
                <button className="w-full flex items-center gap-2 sm:gap-3 p-2 sm:p-3 text-left rounded-lg hover:bg-medical-light transition-colors duration-200">
                  <Activity className="w-4 h-4 sm:w-5 sm:h-5 text-medical-primary flex-shrink-0" />
                  <span className="text-sm sm:text-base text-medical-dark">New Analysis</span>
                </button>
                
                <button className="w-full flex items-center gap-2 sm:gap-3 p-2 sm:p-3 text-left rounded-lg hover:bg-medical-light transition-colors duration-200">
                  <Users className="w-4 h-4 sm:w-5 sm:h-5 text-medical-primary flex-shrink-0" />
                  <span className="text-sm sm:text-base text-medical-dark">Add Patient</span>
                </button>
                
                <button className="w-full flex items-center gap-2 sm:gap-3 p-2 sm:p-3 text-left rounded-lg hover:bg-medical-light transition-colors duration-200">
                  <FileText className="w-4 h-4 sm:w-5 sm:h-5 text-medical-primary flex-shrink-0" />
                  <span className="text-sm sm:text-base text-medical-dark">Generate Report</span>
                </button>
                
                <button className="w-full flex items-center gap-2 sm:gap-3 p-2 sm:p-3 text-left rounded-lg hover:bg-medical-light transition-colors duration-200">
                  <Calendar className="w-4 h-4 sm:w-5 sm:h-5 text-medical-primary flex-shrink-0" />
                  <span className="text-sm sm:text-base text-medical-dark">Schedule Follow-up</span>
                </button>
              </div>
            </div>

            {/* System Alerts */}
            <div className="bg-white rounded-xl sm:rounded-2xl shadow-medical-lg p-4 sm:p-6 border border-gray-100">
              <h3 className="text-base sm:text-lg font-semibold text-medical-dark mb-3 sm:mb-4 flex items-center gap-2">
                <Heart className="w-4 h-4 sm:w-5 sm:h-5 text-red-500" />
                System Alerts
              </h3>
              
              <div className="space-y-3 sm:space-y-4">
                {alerts.map((alert, index) => (
                  <div key={index} className="flex items-start gap-2 sm:gap-3 p-3 rounded-lg bg-gray-50">
                    {React.cloneElement(getAlertIcon(alert.type), { className: "w-4 h-4 sm:w-5 sm:h-5 flex-shrink-0 mt-0.5" })}
                    <div className="flex-1 min-w-0">
                      <p className="text-xs sm:text-sm text-medical-dark font-medium leading-relaxed">{alert.message}</p>
                      <p className="text-xs text-gray-500 mt-1">{alert.time}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* System Status */}
            <div className="bg-white rounded-xl sm:rounded-2xl shadow-medical-lg p-4 sm:p-6 border border-gray-100">
              <h3 className="text-base sm:text-lg font-semibold text-medical-dark mb-3 sm:mb-4">System Status</h3>
              
              <div className="space-y-2 sm:space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs sm:text-sm text-gray-600">AI Model</span>
                  <div className="flex items-center gap-1 sm:gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-xs sm:text-sm text-green-600 font-medium">Online</span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-xs sm:text-sm text-gray-600">Database</span>
                  <div className="flex items-center gap-1 sm:gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-xs sm:text-sm text-green-600 font-medium">Connected</span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-xs sm:text-sm text-gray-600">API Status</span>
                  <div className="flex items-center gap-1 sm:gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-xs sm:text-sm text-green-600 font-medium">Healthy</span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-xs sm:text-sm text-gray-600">Storage</span>
                  <div className="flex items-center gap-1 sm:gap-2">
                    <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                    <span className="text-xs sm:text-sm text-yellow-600 font-medium">75% Used</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
