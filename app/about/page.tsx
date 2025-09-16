"use client";

import React from "react";
import Link from "next/link";
import { 
  Heart, 
  Users, 
  Award, 
  Shield, 
  Brain, 
  Globe,
  Target,
  Stethoscope,
  BookOpen,
  TrendingUp
} from "lucide-react";

const teamMembers = [
  {
    name: "Dr. Sarah Chen",
    role: "Chief Medical Officer",
    specialty: "Vascular Surgery",
    image: "/team-1.jpg",
    description: "15+ years experience in vascular medicine"
  },
  {
    name: "Alex Kumar",
    role: "Lead AI Engineer",
    specialty: "Computer Vision & ML",
    image: "/team-2.jpg", 
    description: "PhD in Computer Science, AI specialist"
  },
  {
    name: "Dr. Maria Rodriguez",
    role: "Clinical Research Director",
    specialty: "Medical Research",
    image: "/team-3.jpg",
    description: "Published researcher in vascular health"
  },
  {
    name: "David Park",
    role: "Head of Technology",
    specialty: "Healthcare Software",
    image: "/team-4.jpg",
    description: "Expert in medical device software"
  }
];

const stats = [
  { label: "Patients Served", value: "50,000+", icon: <Users className="w-8 h-8" /> },
  { label: "Accuracy Rate", value: "95.8%", icon: <Target className="w-8 h-8" /> },
  { label: "Languages Supported", value: "12", icon: <Globe className="w-8 h-8" /> },
  { label: "Medical Partners", value: "200+", icon: <Stethoscope className="w-8 h-8" /> }
];

const features = [
  {
    title: "AI-Powered Detection",
    description: "Advanced machine learning algorithms trained on thousands of medical images for accurate varicose vein detection.",
    icon: <Brain className="w-6 h-6" />
  },
  {
    title: "Medical-Grade Security",
    description: "HIPAA-compliant platform ensuring your medical data is protected with enterprise-level security.",
    icon: <Shield className="w-6 h-6" />
  },
  {
    title: "Expert Validation", 
    description: "All AI results are validated by certified vascular specialists to ensure diagnostic accuracy.",
    icon: <Award className="w-6 h-6" />
  },
  {
    title: "Continuous Learning",
    description: "Our AI system continuously improves through ongoing research and clinical feedback.",
    icon: <TrendingUp className="w-6 h-6" />
  }
];

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light via-white to-vascular-light">
      {/* Hero Section */}
      <div className="pt-12 sm:pt-16 pb-16 sm:pb-20">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6">
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 mb-6 sm:mb-8">
            <div className="p-3 bg-medical-primary rounded-full">
              <Heart className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
            </div>
            <h1 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold text-medical-dark font-medical leading-tight">
              About VarixScan
            </h1>
          </div>
          
          <p className="text-base sm:text-lg lg:text-xl text-medical-dark/70 mb-8 sm:mb-10 leading-relaxed max-w-3xl mx-auto">
            Revolutionizing vascular health through advanced AI technology, making professional-grade 
            varicose vein detection accessible to everyone, everywhere.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
            {stats.map((stat, index) => (
              <div key={index} className="bg-white p-4 sm:p-6 rounded-2xl shadow-medical border border-gray-100">
                <div className="flex items-center justify-center mb-3 text-medical-primary">
                  {React.cloneElement(stat.icon, { className: "w-6 h-6 sm:w-8 sm:h-8" })}
                </div>
                <div className="text-2xl sm:text-3xl font-bold text-medical-dark mb-1">{stat.value}</div>
                <div className="text-xs sm:text-sm text-gray-600">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Mission Section */}
      <div className="py-12 sm:py-16 lg:py-20 bg-white">
        <div className="max-w-6xl mx-auto px-4 sm:px-6">
          <div className="grid lg:grid-cols-2 gap-8 sm:gap-10 lg:gap-12 items-center">
            <div>
              <h2 className="text-2xl sm:text-3xl font-bold text-medical-dark mb-4 sm:mb-6 flex items-center gap-3">
                <Target className="w-6 h-6 sm:w-8 sm:h-8 text-medical-primary" />
                Our Mission
              </h2>
              <p className="text-base sm:text-lg text-gray-700 mb-4 sm:mb-6 leading-relaxed">
                We believe that early detection and proper monitoring of vascular conditions should be 
                accessible to everyone, regardless of location or economic status. Our AI-powered platform 
                democratizes healthcare by bringing expert-level diagnostic capabilities directly to patients 
                and healthcare providers worldwide.
              </p>
              <p className="text-base sm:text-lg text-gray-700 leading-relaxed">
                By combining cutting-edge artificial intelligence with the expertise of certified medical 
                professionals, we&apos;re creating a new standard of care for vascular health that is both 
                accurate and accessible.
              </p>
            </div>
            
            <div className="bg-gradient-to-br from-medical-primary to-vascular-secondary p-6 sm:p-8 rounded-2xl text-white">
              <h3 className="text-xl sm:text-2xl font-semibold mb-3 sm:mb-4">Why It Matters</h3>
              <ul className="space-y-2 sm:space-y-3">
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-white rounded-full mt-2 flex-shrink-0" />
                  <span className="text-sm sm:text-base">Over 40 million people worldwide suffer from varicose veins</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-white rounded-full mt-2 flex-shrink-0" />
                  <span className="text-sm sm:text-base">Early detection can prevent serious complications</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-white rounded-full mt-2 flex-shrink-0" />
                  <span className="text-sm sm:text-base">Many areas lack access to specialized vascular care</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-white rounded-full mt-2 flex-shrink-0" />
                  <span className="text-sm sm:text-base">AI can bridge the gap between patients and specialists</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-medical-dark mb-4">
              Advanced Healthcare Technology
            </h2>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto">
              Our platform combines the latest in artificial intelligence, medical expertise, 
              and user-friendly design to deliver exceptional healthcare solutions.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="bg-white p-8 rounded-2xl shadow-medical border border-gray-100">
                <div className="flex items-center gap-4 mb-4">
                  <div className="p-3 bg-medical-primary rounded-xl text-white">
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-semibold text-medical-dark">{feature.title}</h3>
                </div>
                <p className="text-gray-700 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Team Section */}
      <div className="py-20 bg-white">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-medical-dark mb-4 flex items-center justify-center gap-3">
              <Users className="w-8 h-8 text-medical-primary" />
              Meet Our Expert Team
            </h2>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto">
              Our multidisciplinary team combines medical expertise, technical innovation, 
              and research excellence to advance vascular healthcare.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {teamMembers.map((member, index) => (
              <div key={index} className="bg-gray-50 p-6 rounded-2xl text-center hover:shadow-medical transition-all duration-300">
                <div className="w-24 h-24 bg-medical-primary rounded-full mx-auto mb-4 flex items-center justify-center">
                  <Stethoscope className="w-12 h-12 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-medical-dark mb-1">{member.name}</h3>
                <p className="text-medical-primary font-medium mb-2">{member.role}</p>
                <p className="text-sm text-gray-600 mb-3">{member.specialty}</p>
                <p className="text-xs text-gray-500">{member.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Research & Innovation */}
      <div className="py-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="bg-gradient-to-r from-vascular-light to-medical-light p-12 rounded-3xl">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-medical-dark mb-4 flex items-center justify-center gap-3">
                <BookOpen className="w-8 h-8 text-vascular-secondary" />
                Research & Innovation
              </h2>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              <div className="text-center">
                <div className="text-4xl font-bold text-vascular-secondary mb-2">15+</div>
                <p className="text-medical-dark font-medium">Research Papers Published</p>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-vascular-secondary mb-2">3</div>
                <p className="text-medical-dark font-medium">Clinical Trials Completed</p>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-vascular-secondary mb-2">95.8%</div>
                <p className="text-medical-dark font-medium">Diagnostic Accuracy</p>
              </div>
            </div>

            <div className="mt-8 text-center">
              <p className="text-medical-dark/70 leading-relaxed max-w-4xl mx-auto">
                Our research team continuously works to improve our AI algorithms through clinical studies, 
                peer-reviewed research, and collaboration with leading medical institutions worldwide. 
                We&apos;re committed to advancing the field of vascular medicine through evidence-based innovation.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Call to Action */}
      <div className="py-20 bg-medical-primary text-white">
        <div className="max-w-4xl mx-auto text-center px-6">
          <h2 className="text-3xl font-bold mb-4">
            Join Us in Revolutionizing Healthcare
          </h2>
          <p className="text-xl opacity-90 mb-8">
            Experience the future of vascular health monitoring with our AI-powered platform.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/vericose" className="bg-white text-medical-primary px-8 py-4 rounded-xl font-semibold hover:bg-gray-100 transition-colors duration-200 text-center text-decoration-none">
              Try Free Analysis
            </Link>
            <Link href="/contact" className="border-2 border-white text-white px-8 py-4 rounded-xl font-semibold hover:bg-white hover:text-medical-primary transition-all duration-200 text-center text-decoration-none">
              Contact Our Team
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
