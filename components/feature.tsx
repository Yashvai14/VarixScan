'use client'

import { useState } from "react";
import { Activity, Shield, MessageCircle, Watch, BarChart3 } from 'lucide-react';

const features = [
  {
    title: "AI-Powered Vein Detection",
    description: "Upload leg images and let VarixScan detect varicose veins, classify severity, and provide early-stage alerts using deep learning models.",
    icon: <Activity className="w-6 h-6" />
  },
  {
    title: "Personalized Risk Assessment",
    description: "Input lifestyle and health details to get a personalized risk score along with preventive care and lifestyle recommendations.",
    icon: <Shield className="w-6 h-6" />
  },
  {
    title: "Multilingual Health Assistant",
    description: "Interact with VarixScan in your preferred language — Hindi, Marathi, English, and more — powered by ChatGPT for easy guidance.",
    icon: <MessageCircle className="w-6 h-6" />
  },
  {
    title: "Real-Time Health Monitoring",
    description: "Wearable sensors track leg motion and blood flow, enabling continuous monitoring and early warnings for potential complications.",
    icon: <Watch className="w-6 h-6" />
  },
  {
    title: "Comprehensive Health Dashboard",
    description: "View your vein health, risk score, wearable data, and AI suggestions all from one simple, intuitive dashboard.",
    icon: <BarChart3 className="w-6 h-6" />
  },
];

export default function FeaturesSection() {
  const [activeIndex, setActiveIndex] = useState(0);

  return (
    <section className="w-full py-32 bg-gradient-to-br from-vascular-light to-medical-light" id="features">
      <div className="max-w-[1200px] mx-auto" style={{ width: "1200px" }}>
        <h2 className="text-6xl font-bold text-medical-primary mb-4 font-medical">Why Choose VarixScan?</h2>
        <p className="text-medical-dark/70 text-xl mb-20 max-w-2xl">
          Stay ahead in vein health with AI-driven insights, personalized monitoring, and multilingual guidance.
        </p>
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center space-y-8 md:space-y-0 md:space-x-8">
          {/* Feature titles */}
          <div className="flex flex-col space-y-4">
            {features.map((feature, index) => (
              <button
                key={index}
                className={`p-5 rounded-xl text-xl transition-all duration-300 flex items-center gap-3 ${
                  activeIndex === index
                    ? "bg-medical-primary text-white shadow-medical-lg"
                    : "bg-white text-medical-dark hover:bg-medical-light hover:shadow-medical border border-gray-200"
                }`}
                onClick={() => setActiveIndex(index)}
              >
                <div className={`${activeIndex === index ? 'text-white' : 'text-medical-primary'}`}>
                  {feature.icon}
                </div>
                {feature.title}
              </button>
            ))}
          </div>

          {/* Feature description */}
          <div className="bg-white p-8 h-[400px] w-[800px] rounded-xl border border-gray-200 shadow-medical-lg">
            <div className="flex items-center gap-3 mb-5">
              <div className="p-3 bg-medical-primary rounded-xl text-white">
                {features[activeIndex].icon}
              </div>
              <h3 className="text-3xl font-semibold text-medical-primary">
                {features[activeIndex].title}
              </h3>
            </div>
            <p className="text-medical-dark/80 text-base leading-relaxed">{features[activeIndex].description}</p>
          </div>
        </div>
      </div>
    </section>
  );
}
