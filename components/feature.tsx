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
    <section className="w-full py-16 sm:py-20 md:py-24 lg:py-32 bg-gradient-to-br from-vascular-light to-medical-light px-4 sm:px-6 lg:px-8" id="features">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12 sm:mb-16 lg:mb-20">
          <h2 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold text-medical-primary mb-4 sm:mb-6 font-medical leading-tight">
            Why Choose VarixScan?
          </h2>
          <p className="text-medical-dark/70 text-lg sm:text-xl lg:text-xl max-w-3xl mx-auto leading-relaxed">
            Stay ahead in vein health with AI-driven insights, personalized monitoring, and multilingual guidance.
          </p>
        </div>
        
        <div className="flex flex-col xl:flex-row gap-8 lg:gap-12">
          {/* Feature titles */}
          <div className="flex flex-col space-y-3 sm:space-y-4 xl:min-w-[400px]">
            {features.map((feature, index) => (
              <button
                key={index}
                className={`p-4 sm:p-5 rounded-xl text-sm sm:text-base lg:text-lg xl:text-xl transition-all duration-300 flex items-center gap-3 text-left ${
                  activeIndex === index
                    ? "bg-medical-primary text-white shadow-medical-lg"
                    : "bg-white text-medical-dark hover:bg-medical-light hover:shadow-medical border border-gray-200"
                }`}
                onClick={() => setActiveIndex(index)}
              >
                <div className={`flex-shrink-0 ${activeIndex === index ? 'text-white' : 'text-medical-primary'}`}>
                  {feature.icon}
                </div>
                <span className="font-semibold">{feature.title}</span>
              </button>
            ))}
          </div>

          {/* Feature description */}
          <div className="bg-white p-6 sm:p-8 min-h-[300px] sm:min-h-[350px] lg:min-h-[400px] flex-1 rounded-xl border border-gray-200 shadow-medical-lg">
            <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-4 mb-4 sm:mb-6">
              <div className="p-3 bg-medical-primary rounded-xl text-white w-fit">
                {features[activeIndex].icon}
              </div>
              <h3 className="text-xl sm:text-2xl lg:text-3xl font-semibold text-medical-primary leading-tight">
                {features[activeIndex].title}
              </h3>
            </div>
            <p className="text-medical-dark/80 text-sm sm:text-base lg:text-lg leading-relaxed">
              {features[activeIndex].description}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
