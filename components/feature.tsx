'use client'

import { useState } from "react";

const features = [
  {
    title: "AI-Powered Vein Detection",
    description: "Upload leg images and let VarixScan detect varicose veins, classify severity, and provide early-stage alerts using deep learning models.",
  },
  {
    title: "Personalized Risk Assessment",
    description: "Input lifestyle and health details to get a personalized risk score along with preventive care and lifestyle recommendations.",
  },
  {
    title: "Multilingual Health Assistant",
    description: "Interact with VarixScan in your preferred language — Hindi, Marathi, English, and more — powered by ChatGPT for easy guidance.",
  },
  {
    title: "Real-Time Health Monitoring",
    description: "Wearable sensors track leg motion and blood flow, enabling continuous monitoring and early warnings for potential complications.",
  },
  {
    title: "Comprehensive Health Dashboard",
    description: "View your vein health, risk score, wearable data, and AI suggestions all from one simple, intuitive dashboard.",
  },
];

export default function FeaturesSection() {
  const [activeIndex, setActiveIndex] = useState(0);

  return (
    <section className="w-full py-32 bg-white" id="features">
      <div className="max-w-[1200px] mx-auto" style={{ width: "1200px" }}>
        <h2 className="text-6xl font-bold text-lime-500 mb-4">Why Choose VarixScan?</h2>
        <p className="text-gray-600 text-xl mb-20 max-w-2xl">
          Stay ahead in vein health with AI-driven insights, personalized monitoring, and multilingual guidance.
        </p>
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center space-y-8 md:space-y-0 md:space-x-8">
          {/* Feature titles */}
          <div className="flex flex-col space-y-4">
            {features.map((feature, index) => (
              <button
                key={index}
                className={`p-5 rounded-xl text-xl transition-all duration-300 ${
                  activeIndex === index
                    ? "bg-lime-500 text-white border-lime-500"
                    : "bg-gray-100 text-gray-800 hover:bg-gray-200"
                }`}
                onClick={() => setActiveIndex(index)}
              >
                {feature.title}
              </button>
            ))}
          </div>

          {/* Feature description */}
          <div className="bg-gray-50 p-8 h-[400px] w-[800px] rounded-xl border border-gray-200 shadow-md">
            <h3 className="text-3xl font-semibold text-lime-500 mb-5">
              {features[activeIndex].title}
            </h3>
            <p className="text-gray-700 text-base">{features[activeIndex].description}</p>
          </div>
        </div>
      </div>
    </section>
  );
}
