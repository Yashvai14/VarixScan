"use client";

import React, { useRef, useState } from "react";
import { FaChevronDown } from "react-icons/fa";

const faqData = [
  {
    key: "1",
    question: "What is VarixScan and how does it work?",
    answer:
      "VarixScan is an AI-powered varicose vein diagnostic and monitoring platform. It uses deep learning to detect varicose veins from leg images, wearable sensors to monitor blood flow and leg motion, and provides personalized risk assessment and preventive care suggestions.",
  },
  {
    key: "2",
    question: "Is VarixScan suitable for home use?",
    answer:
      "Yes. VarixScan is designed to be patient-friendly and can be used at home to monitor vein health, track symptoms, and receive real-time guidance from the AI assistant in multiple languages.",
  },
  {
    key: "3",
    question: "Which wearable sensors are compatible?",
    answer:
      "VarixScan supports IMU sensors for leg motion tracking and Doppler/pulse sensors for monitoring blood flow. These sensors provide real-time data to the platform for accurate monitoring and alerts.",
  },
  {
    key: "4",
    question: "Can I share my reports with my doctor?",
    answer:
      "Absolutely. VarixScan generates PDF health reports summarizing diagnosis, risk scores, and wearable data, which can be easily shared with healthcare professionals for consultations or follow-ups.",
  },
];

export default function FAQSection() {
  const [expanded, setExpanded] = useState<string | false>("1");
  const contentRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const toggleFAQ = (key: string) => {
    setExpanded(expanded === key ? false : key);
  };

  return (
    <section className="bg-white py-20 px-4">
      <div className="max-w-3xl mx-auto">
        <h2 className="text-3xl sm:text-4xl font-bold text-center mb-10 text-medical-primary">
          Frequently Asked Questions
        </h2>
        <div className="space-y-4">
          {faqData.map((faq) => {
            const isOpen = expanded === faq.key;
            const contentHeight = contentRefs.current[faq.key]?.scrollHeight ?? 0;

            return (
              <div
                key={faq.key}
                className="bg-white rounded-3xl shadow-md transition-all duration-300 border border-gray-200 overflow-hidden"
              >
                <button
                  onClick={() => toggleFAQ(faq.key)}
                  className="w-full flex justify-between items-center px-6 py-5 text-left text-[#111827] font-semibold text-base sm:text-lg focus:outline-none"
                >
                  <span className={`transition-all ${isOpen ? "font-bold text-medical-primary" : ""}`}>
                    {faq.question}
                  </span>
                  <FaChevronDown
                    className={`ml-4 text-gray-500 transform transition-transform duration-500 ${
                      isOpen ? "rotate-180 text-medical-primary" : ""
                    }`}
                  />
                </button>
                <div
                  ref={(el) => { contentRefs.current[faq.key] = el; }}
                  style={{
                    maxHeight: isOpen ? `${contentHeight}px` : "0px",
                    transition: "max-height 0.5s ease, opacity 0.5s ease",
                    opacity: isOpen ? 1 : 0,
                  }}
                  className="px-6 text-gray-600 text-sm leading-relaxed"
                >
                  <div className="py-4">{faq.answer}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
