"use client";

import { useState } from "react";
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  User, 
  Calendar,
  Weight,
  Briefcase,
  Heart,
  Activity,
  Clock,
  ArrowRight,
  Info
} from "lucide-react";

interface RiskFactor {
  id: string;
  label: string;
  weight: number;
  category: string;
}

const riskFactors: RiskFactor[] = [
  { id: "age", label: "Age over 50", weight: 2, category: "Demographics" },
  { id: "gender", label: "Female", weight: 1.5, category: "Demographics" },
  { id: "pregnancy", label: "Multiple pregnancies", weight: 2, category: "Medical History" },
  { id: "family_history", label: "Family history of varicose veins", weight: 3, category: "Genetics" },
  { id: "prolonged_standing", label: "Prolonged standing/sitting", weight: 2, category: "Lifestyle" },
  { id: "obesity", label: "Obesity (BMI > 30)", weight: 2.5, category: "Physical" },
  { id: "previous_dvt", label: "Previous DVT or blood clots", weight: 3, category: "Medical History" },
  { id: "hormone_therapy", label: "Hormone replacement therapy", weight: 1.5, category: "Medical History" },
  { id: "smoking", label: "Smoking", weight: 1.5, category: "Lifestyle" },
  { id: "sedentary", label: "Sedentary lifestyle", weight: 1.5, category: "Lifestyle" }
];

export default function RiskAssessment() {
  const [selectedFactors, setSelectedFactors] = useState<string[]>([]);
  const [age, setAge] = useState("");
  const [bmi, setBmi] = useState("");
  const [showResults, setShowResults] = useState(false);

  const toggleFactor = (factorId: string) => {
    setSelectedFactors(prev => 
      prev.includes(factorId) 
        ? prev.filter(id => id !== factorId)
        : [...prev, factorId]
    );
  };

  const calculateRisk = () => {
    let totalScore = 0;
    selectedFactors.forEach(factorId => {
      const factor = riskFactors.find(f => f.id === factorId);
      if (factor) totalScore += factor.weight;
    });

    // Age factor
    if (parseInt(age) > 50) totalScore += 2;
    if (parseInt(age) > 65) totalScore += 1;

    // BMI factor
    const bmiValue = parseFloat(bmi);
    if (bmiValue > 30) totalScore += 2.5;
    else if (bmiValue > 25) totalScore += 1;

    return totalScore;
  };

  const getRiskLevel = (score: number) => {
    if (score <= 2) return { level: "Low", color: "text-green-600", bg: "bg-green-50", border: "border-green-200" };
    if (score <= 5) return { level: "Moderate", color: "text-yellow-600", bg: "bg-yellow-50", border: "border-yellow-200" };
    if (score <= 8) return { level: "High", color: "text-orange-600", bg: "bg-orange-50", border: "border-orange-200" };
    return { level: "Very High", color: "text-red-600", bg: "bg-red-50", border: "border-red-200" };
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setShowResults(true);
  };

  const riskScore = calculateRisk();
  const riskLevel = getRiskLevel(riskScore);

  const groupedFactors = riskFactors.reduce((acc, factor) => {
    if (!acc[factor.category]) acc[factor.category] = [];
    acc[factor.category].push(factor);
    return acc;
  }, {} as Record<string, RiskFactor[]>);

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light via-white to-vascular-light p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 pt-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-medical-warning rounded-full">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-medical-dark font-medical">
              Personalized Risk Assessment
            </h1>
          </div>
          <p className="text-medical-dark/70 text-lg max-w-3xl mx-auto">
            Evaluate your personal risk factors for developing varicose veins with our comprehensive assessment tool
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Assessment Form */}
          <div className="lg:col-span-2">
            <form onSubmit={handleSubmit} className="bg-white shadow-medical-lg rounded-2xl p-8 border border-gray-100">
              <h2 className="text-xl font-semibold text-medical-dark mb-6 flex items-center gap-2">
                <User className="w-5 h-5 text-medical-primary" />
                Personal Information
              </h2>

              {/* Basic Info */}
              <div className="grid md:grid-cols-2 gap-4 mb-8">
                <div>
                  <label className="block text-sm font-medium text-medical-dark mb-2">
                    <Calendar className="w-4 h-4 inline mr-1" />
                    Age
                  </label>
                  <input
                    type="number"
                    value={age}
                    onChange={(e) => setAge(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-medical-primary focus:border-medical-primary transition-colors duration-200"
                    placeholder="Enter your age"
                    required
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-medical-dark mb-2">
                    <Weight className="w-4 h-4 inline mr-1" />
                    BMI
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={bmi}
                    onChange={(e) => setBmi(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-medical-primary focus:border-medical-primary transition-colors duration-200"
                    placeholder="Enter your BMI"
                    required
                  />
                </div>
              </div>

              {/* Risk Factors by Category */}
              <div className="space-y-6">
                {Object.entries(groupedFactors).map(([category, factors]) => (
                  <div key={category}>
                    <h3 className="text-lg font-semibold text-medical-dark mb-4 flex items-center gap-2">
                      {category === "Demographics" && <User className="w-4 h-4" />}
                      {category === "Medical History" && <Heart className="w-4 h-4" />}
                      {category === "Genetics" && <Activity className="w-4 h-4" />}
                      {category === "Lifestyle" && <Clock className="w-4 h-4" />}
                      {category === "Physical" && <Weight className="w-4 h-4" />}
                      {category}
                    </h3>
                    
                    <div className="grid md:grid-cols-2 gap-3">
                      {factors.map(factor => (
                        <label
                          key={factor.id}
                          className={`flex items-center p-4 rounded-xl border-2 cursor-pointer transition-all duration-200 ${
                            selectedFactors.includes(factor.id)
                              ? 'border-medical-primary bg-medical-light'
                              : 'border-gray-200 hover:border-medical-accent'
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={selectedFactors.includes(factor.id)}
                            onChange={() => toggleFactor(factor.id)}
                            className="sr-only"
                          />
                          <div className={`w-5 h-5 rounded border-2 mr-3 flex items-center justify-center ${
                            selectedFactors.includes(factor.id)
                              ? 'border-medical-primary bg-medical-primary'
                              : 'border-gray-300'
                          }`}>
                            {selectedFactors.includes(factor.id) && (
                              <CheckCircle className="w-3 h-3 text-white" />
                            )}
                          </div>
                          <span className="text-medical-dark font-medium">{factor.label}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              <button
                type="submit"
                className="w-full mt-8 bg-medical-primary text-white py-4 px-6 rounded-xl font-semibold hover:bg-medical-secondary transition-all duration-200 shadow-medical flex items-center justify-center gap-2"
              >
                <Shield className="w-5 h-5" />
                Calculate My Risk
                <ArrowRight className="w-4 h-4" />
              </button>
            </form>
          </div>

          {/* Results Panel */}
          <div className="bg-white shadow-medical-lg rounded-2xl p-8 border border-gray-100">
            <h2 className="text-xl font-semibold text-medical-dark mb-6 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-medical-warning" />
              Risk Analysis
            </h2>

            {!showResults ? (
              <div className="text-center py-12">
                <div className="w-20 h-20 bg-gray-100 rounded-full mx-auto mb-4 flex items-center justify-center">
                  <Shield className="w-8 h-8 text-gray-400" />
                </div>
                <p className="text-gray-500">Complete the assessment to see your risk level</p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Risk Level Display */}
                <div className={`p-6 rounded-xl border-2 ${riskLevel.bg} ${riskLevel.border}`}>
                  <div className="text-center">
                    <div className={`text-3xl font-bold ${riskLevel.color} mb-2`}>
                      {riskLevel.level} Risk
                    </div>
                    <div className="text-gray-600">
                      Score: {riskScore.toFixed(1)} / 20
                    </div>
                  </div>
                </div>

                {/* Risk Factors Summary */}
                <div>
                  <h4 className="font-semibold text-medical-dark mb-3">Selected Risk Factors:</h4>
                  <div className="space-y-2">
                    {selectedFactors.map(factorId => {
                      const factor = riskFactors.find(f => f.id === factorId);
                      return factor ? (
                        <div key={factorId} className="flex justify-between items-center p-2 bg-gray-50 rounded-lg">
                          <span className="text-sm">{factor.label}</span>
                          <span className="text-xs font-semibold text-medical-primary">+{factor.weight}</span>
                        </div>
                      ) : null;
                    })}
                  </div>
                </div>

                {/* Recommendations */}
                <div>
                  <h4 className="font-semibold text-medical-dark mb-3 flex items-center gap-1">
                    <Info className="w-4 h-4" />
                    Recommendations:
                  </h4>
                  <div className="space-y-2 text-sm text-gray-600">
                    {riskLevel.level === "Low" && (
                      <ul className="list-disc list-inside space-y-1">
                        <li>Maintain regular exercise routine</li>
                        <li>Monitor for any changes in your legs</li>
                        <li>Continue healthy lifestyle choices</li>
                      </ul>
                    )}
                    {riskLevel.level === "Moderate" && (
                      <ul className="list-disc list-inside space-y-1">
                        <li>Consider compression stockings for long periods of standing</li>
                        <li>Elevate legs when resting</li>
                        <li>Regular check-ups with healthcare provider</li>
                        <li>Increase physical activity</li>
                      </ul>
                    )}
                    {(riskLevel.level === "High" || riskLevel.level === "Very High") && (
                      <ul className="list-disc list-inside space-y-1">
                        <li>Consult with a vascular specialist</li>
                        <li>Use medical-grade compression stockings</li>
                        <li>Regular monitoring and screening</li>
                        <li>Consider preventive treatments</li>
                      </ul>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
