"use client";

import { useState, useEffect } from "react";
import { 
  Watch, 
  Heart, 
  Activity, 
  TrendingUp, 
  TrendingDown,
  Zap,
  Clock,
  AlertCircle,
  CheckCircle,
  Bluetooth,
  Battery,
  Smartphone,
  BarChart3
} from "lucide-react";

interface HealthMetric {
  id: string;
  name: string;
  value: number;
  unit: string;
  status: "normal" | "warning" | "alert";
  trend: "up" | "down" | "stable";
  icon: React.ReactNode;
}

interface WearableDevice {
  id: string;
  name: string;
  type: string;
  connected: boolean;
  battery: number;
  lastSync: string;
}

export default function WearableMonitoring() {
  const [selectedDevice, setSelectedDevice] = useState<string>("smartwatch");
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [isClient, setIsClient] = useState(false);

  // Mock health data
  const [healthMetrics, setHealthMetrics] = useState<HealthMetric[]>([
    {
      id: "heart_rate",
      name: "Heart Rate",
      value: 72,
      unit: "bpm",
      status: "normal",
      trend: "stable",
      icon: <Heart className="w-5 h-5" />
    },
    {
      id: "steps",
      name: "Steps Today",
      value: 8432,
      unit: "steps",
      status: "normal",
      trend: "up",
      icon: <Activity className="w-5 h-5" />
    },
    {
      id: "leg_elevation",
      name: "Leg Elevation Time",
      value: 45,
      unit: "min/day",
      status: "warning",
      trend: "down",
      icon: <TrendingUp className="w-5 h-5" />
    },
    {
      id: "movement_frequency",
      name: "Movement Frequency",
      value: 12,
      unit: "times/hour",
      status: "normal",
      trend: "stable",
      icon: <Zap className="w-5 h-5" />
    }
  ]);

  const devices: WearableDevice[] = [
    {
      id: "smartwatch",
      name: "Apple Watch Series 9",
      type: "Smartwatch",
      connected: true,
      battery: 85,
      lastSync: "2 minutes ago"
    },
    {
      id: "fitness_tracker",
      name: "Fitbit Charge 6",
      type: "Fitness Tracker",
      connected: false,
      battery: 0,
      lastSync: "3 hours ago"
    },
    {
      id: "compression_sensor",
      name: "Smart Compression Socks",
      type: "Medical Device",
      connected: true,
      battery: 67,
      lastSync: "1 minute ago"
    }
  ];

  useEffect(() => {
    // Set isClient to true after component mounts
    setIsClient(true);
    
    const timer = setInterval(() => {
      setCurrentTime(new Date());
      
      // Simulate real-time data updates
      if (isMonitoring) {
        setHealthMetrics(prev => prev.map(metric => ({
          ...metric,
          value: metric.id === "heart_rate" 
            ? Math.max(60, Math.min(100, metric.value + (Math.random() - 0.5) * 4))
            : metric.value
        })));
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [isMonitoring]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "normal": return "text-green-600 bg-green-50 border-green-200";
      case "warning": return "text-yellow-600 bg-yellow-50 border-yellow-200";
      case "alert": return "text-red-600 bg-red-50 border-red-200";
      default: return "text-gray-600 bg-gray-50 border-gray-200";
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "up": return <TrendingUp className="w-4 h-4 text-green-500" />;
      case "down": return <TrendingDown className="w-4 h-4 text-red-500" />;
      default: return <div className="w-4 h-4 rounded-full bg-gray-300" />;
    }
  };

  const getBatteryColor = (battery: number) => {
    if (battery > 50) return "bg-green-500";
    if (battery > 20) return "bg-yellow-500";
    return "bg-red-500";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light via-white to-vascular-light p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 pt-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-medical-accent rounded-full">
              <Watch className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-medical-dark font-medical">
              Wearable Health Monitoring
            </h1>
          </div>
          <p className="text-medical-dark/70 text-lg max-w-3xl mx-auto">
            Continuous monitoring of your vascular health through advanced wearable technology
          </p>
        </div>

        <div className="grid lg:grid-cols-4 gap-6">
          {/* Device Connection Panel */}
          <div className="bg-white shadow-medical-lg rounded-2xl p-6 border border-gray-100">
            <h2 className="text-lg font-semibold text-medical-dark mb-4 flex items-center gap-2">
              <Bluetooth className="w-5 h-5 text-medical-primary" />
              Connected Devices
            </h2>
            
            <div className="space-y-4">
              {devices.map(device => (
                <div
                  key={device.id}
                  className={`p-4 rounded-xl border-2 cursor-pointer transition-all duration-200 ${
                    selectedDevice === device.id
                      ? 'border-medical-primary bg-medical-light'
                      : 'border-gray-200 hover:border-medical-accent'
                  }`}
                  onClick={() => setSelectedDevice(device.id)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {device.connected ? (
                        <div className="w-3 h-3 bg-green-500 rounded-full" />
                      ) : (
                        <div className="w-3 h-3 bg-gray-400 rounded-full" />
                      )}
                      <span className="font-medium text-medical-dark text-sm">
                        {device.name}
                      </span>
                    </div>
                    {device.connected && (
                      <div className="flex items-center gap-1">
                        <Battery className="w-3 h-3" />
                        <div className="w-6 h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div 
                            className={`h-full ${getBatteryColor(device.battery)} transition-all duration-300`}
                            style={{ width: `${device.battery}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <div className="text-xs text-gray-500">
                    {device.type} • Last sync: {device.lastSync}
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 pt-4 border-t border-gray-200">
              <button
                onClick={() => setIsMonitoring(!isMonitoring)}
                className={`w-full py-3 px-4 rounded-xl font-semibold transition-all duration-200 flex items-center justify-center gap-2 ${
                  isMonitoring
                    ? 'bg-red-500 hover:bg-red-600 text-white'
                    : 'bg-medical-primary hover:bg-medical-secondary text-white'
                }`}
              >
                {isMonitoring ? (
                  <>
                    <AlertCircle className="w-4 h-4" />
                    Stop Monitoring
                  </>
                ) : (
                  <>
                    <Activity className="w-4 h-4" />
                    Start Monitoring
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Real-time Metrics */}
          <div className="lg:col-span-2 space-y-6">
            {/* Live Metrics Grid */}
            <div className="grid md:grid-cols-2 gap-4">
              {healthMetrics.map(metric => (
                <div
                  key={metric.id}
                  className={`bg-white p-6 rounded-2xl shadow-medical border-2 ${getStatusColor(metric.status)}`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      {metric.icon}
                      <span className="font-medium text-medical-dark">{metric.name}</span>
                    </div>
                    {getTrendIcon(metric.trend)}
                  </div>
                  
                  <div className="text-2xl font-bold text-medical-dark mb-1">
                    {typeof metric.value === 'number' ? metric.value.toLocaleString() : metric.value}
                    <span className="text-sm font-normal text-gray-500 ml-2">
                      {metric.unit}
                    </span>
                  </div>
                  
                  <div className="flex items-center gap-1">
                    {metric.status === 'normal' && <CheckCircle className="w-4 h-4 text-green-500" />}
                    {metric.status === 'warning' && <AlertCircle className="w-4 h-4 text-yellow-500" />}
                    {metric.status === 'alert' && <AlertCircle className="w-4 h-4 text-red-500" />}
                    <span className="text-xs text-gray-600 capitalize">{metric.status}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Activity Timeline */}
            <div className="bg-white shadow-medical-lg rounded-2xl p-6 border border-gray-100">
              <h3 className="text-lg font-semibold text-medical-dark mb-4 flex items-center gap-2">
                <Clock className="w-5 h-5 text-medical-primary" />
Today&apos;s Activity Timeline
              </h3>
              
              <div className="space-y-3">
                {[
                  { time: "09:00", activity: "Morning walk", duration: "30 min", status: "completed" },
                  { time: "11:30", activity: "Leg elevation", duration: "15 min", status: "completed" },
                  { time: "14:15", activity: "Afternoon break", duration: "10 min", status: "completed" },
                  { time: "16:00", activity: "Compression therapy", duration: "20 min", status: "active" },
                  { time: "18:30", activity: "Evening exercise", duration: "25 min", status: "scheduled" }
                ].map((item, index) => (
                  <div key={index} className="flex items-center gap-4 p-3 rounded-lg hover:bg-gray-50">
                    <div className={`w-3 h-3 rounded-full ${
                      item.status === 'completed' ? 'bg-green-500' :
                      item.status === 'active' ? 'bg-blue-500 animate-pulse' :
                      'bg-gray-300'
                    }`} />
                    
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-medical-dark">{item.activity}</span>
                        <span className="text-sm text-gray-500">{item.time}</span>
                      </div>
                      <div className="text-sm text-gray-600">{item.duration}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Insights & Recommendations */}
          <div className="space-y-6">
            {/* Current Status */}
            <div className="bg-white shadow-medical-lg rounded-2xl p-6 border border-gray-100">
              <h3 className="text-lg font-semibold text-medical-dark mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-medical-success" />
                Health Status
              </h3>
              
              {isMonitoring ? (
                <div className="space-y-4">
                  <div className="p-4 bg-green-50 rounded-xl border border-green-200">
                    <div className="flex items-center gap-2 mb-2">
                      <CheckCircle className="w-5 h-5 text-green-600" />
                      <span className="font-medium text-green-800">Active Monitoring</span>
                    </div>
                    <p className="text-sm text-green-700">
                      Your wearable devices are actively monitoring your health metrics.
                    </p>
                  </div>
                  
                  <div className="text-center py-2">
                    <div className="text-sm text-gray-500">Live since</div>
                    <div className="font-mono text-lg font-semibold text-medical-primary" suppressHydrationWarning>
                      {isClient ? currentTime.toLocaleTimeString() : '--:--:--'}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <div className="w-16 h-16 bg-gray-100 rounded-full mx-auto mb-4 flex items-center justify-center">
                    <Watch className="w-6 h-6 text-gray-400" />
                  </div>
                  <p className="text-gray-500">Start monitoring to see live health data</p>
                </div>
              )}
            </div>

            {/* AI Recommendations */}
            <div className="bg-white shadow-medical-lg rounded-2xl p-6 border border-gray-100">
              <h3 className="text-lg font-semibold text-medical-dark mb-4 flex items-center gap-2">
                <Smartphone className="w-5 h-5 text-vascular-secondary" />
                AI Recommendations
              </h3>
              
              <div className="space-y-3">
                <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                  <div className="font-medium text-blue-800 text-sm mb-1">
                    Increase leg elevation time
                  </div>
                  <p className="text-xs text-blue-700">
                    Current: 45 min/day • Target: 60 min/day
                  </p>
                </div>
                
                <div className="p-3 bg-green-50 rounded-lg border border-green-200">
                  <div className="font-medium text-green-800 text-sm mb-1">
                    Great step count today!
                  </div>
                  <p className="text-xs text-green-700">
                    8,432 steps achieved your daily goal
                  </p>
                </div>
                
                <div className="p-3 bg-yellow-50 rounded-lg border border-yellow-200">
                  <div className="font-medium text-yellow-800 text-sm mb-1">
                    Take movement breaks
                  </div>
                  <p className="text-xs text-yellow-700">
                    Stand and move every 30 minutes
                  </p>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-white shadow-medical-lg rounded-2xl p-6 border border-gray-100">
              <h3 className="text-lg font-semibold text-medical-dark mb-4">
                Quick Actions
              </h3>
              
              <div className="space-y-2">
                <button className="w-full p-3 text-left rounded-lg hover:bg-gray-50 transition-colors duration-200">
                  <div className="font-medium text-medical-dark text-sm">Export Data</div>
                  <div className="text-xs text-gray-500">Download weekly report</div>
                </button>
                
                <button className="w-full p-3 text-left rounded-lg hover:bg-gray-50 transition-colors duration-200">
                  <div className="font-medium text-medical-dark text-sm">Share with Doctor</div>
                  <div className="text-xs text-gray-500">Send health summary</div>
                </button>
                
                <button className="w-full p-3 text-left rounded-lg hover:bg-gray-50 transition-colors duration-200">
                  <div className="font-medium text-medical-dark text-sm">Set Reminder</div>
                  <div className="text-xs text-gray-500">Custom health alerts</div>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
