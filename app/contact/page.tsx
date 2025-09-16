"use client";

import { useState } from "react";
import { 
  Mail, 
  Phone, 
  MapPin, 
  Clock,
  Send,
  MessageSquare,
  Globe,
  Shield,
  CheckCircle,
  AlertCircle
} from "lucide-react";

export default function ContactPage() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    subject: "",
    message: "",
    type: "general"
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate form submission
    setTimeout(() => {
      setIsSubmitting(false);
      setSubmitted(true);
      setFormData({ name: "", email: "", subject: "", message: "", type: "general" });
    }, 2000);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const contactMethods = [
    {
      icon: <Mail className="w-6 h-6" />,
      title: "Email Support",
      content: "support@varixscan.com",
      description: "Get help with technical issues or general inquiries"
    },
    {
      icon: <Phone className="w-6 h-6" />,
      title: "Phone Support",
      content: "+1 (555) 123-4567",
      description: "Speak directly with our support team"
    },
    {
      icon: <MapPin className="w-6 h-6" />,
      title: "Office Location",
      content: "123 Medical Plaza, Health City, HC 12345",
      description: "Visit our headquarters"
    },
    {
      icon: <Clock className="w-6 h-6" />,
      title: "Business Hours",
      content: "Mon-Fri: 8AM-8PM EST",
      description: "Weekend support available for urgent matters"
    }
  ];

  const faqItems = [
    {
      question: "Is my medical data secure?",
      answer: "Yes, we use HIPAA-compliant, enterprise-grade security measures to protect all patient data."
    },
    {
      question: "How accurate is the AI detection?",
      answer: "Our AI system has a 95.8% accuracy rate, validated through clinical studies and expert review."
    },
    {
      question: "Can I use this for medical diagnosis?",
      answer: "Our platform provides screening results that should be reviewed by a qualified healthcare provider."
    },
    {
      question: "What image formats are supported?",
      answer: "We support JPG, PNG, GIF, and BMP formats up to 10MB in size."
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-light via-white to-vascular-light">
      {/* Header */}
      <div className="pt-12 sm:pt-16 pb-8 sm:pb-12">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6">
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 mb-6 sm:mb-8">
            <div className="p-3 bg-medical-primary rounded-full">
              <MessageSquare className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
            </div>
            <h1 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold text-medical-dark font-medical leading-tight">
              Contact Us
            </h1>
          </div>
          
          <p className="text-base sm:text-lg lg:text-xl text-medical-dark/70 leading-relaxed max-w-3xl mx-auto">
            Get in touch with our team of medical and technical experts. We&apos;re here to help 
            you make the most of our AI-powered vascular health platform.
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 pb-12 sm:pb-16 lg:pb-20">
        <div className="grid lg:grid-cols-3 gap-6 sm:gap-8">
          {/* Contact Form */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-2xl shadow-medical-lg p-8 border border-gray-100">
              <h2 className="text-2xl font-bold text-medical-dark mb-6 flex items-center gap-3">
                <Send className="w-6 h-6 text-medical-primary" />
                Send us a Message
              </h2>

              {submitted ? (
                <div className="text-center py-12">
                  <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-medical-dark mb-2">
                    Message Sent Successfully!
                  </h3>
                  <p className="text-gray-600 mb-6">
                    Thank you for contacting us. We&apos;ll get back to you within 24 hours.
                  </p>
                  <button 
                    onClick={() => setSubmitted(false)}
                    className="bg-medical-primary text-white px-6 py-2 rounded-xl hover:bg-medical-secondary transition-colors duration-200"
                  >
                    Send Another Message
                  </button>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-medical-dark mb-2">
                        Full Name *
                      </label>
                      <input
                        type="text"
                        name="name"
                        value={formData.name}
                        onChange={handleInputChange}
                        required
                        className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-medical-primary focus:border-medical-primary transition-colors duration-200"
                        placeholder="Enter your full name"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-medical-dark mb-2">
                        Email Address *
                      </label>
                      <input
                        type="email"
                        name="email"
                        value={formData.email}
                        onChange={handleInputChange}
                        required
                        className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-medical-primary focus:border-medical-primary transition-colors duration-200"
                        placeholder="Enter your email"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-medical-dark mb-2">
                      Inquiry Type *
                    </label>
                    <select
                      name="type"
                      value={formData.type}
                      onChange={handleInputChange}
                      className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-medical-primary focus:border-medical-primary transition-colors duration-200"
                    >
                      <option value="general">General Inquiry</option>
                      <option value="technical">Technical Support</option>
                      <option value="medical">Medical Questions</option>
                      <option value="partnership">Partnership</option>
                      <option value="press">Press & Media</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-medical-dark mb-2">
                      Subject *
                    </label>
                    <input
                      type="text"
                      name="subject"
                      value={formData.subject}
                      onChange={handleInputChange}
                      required
                      className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-medical-primary focus:border-medical-primary transition-colors duration-200"
                      placeholder="Brief description of your inquiry"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-medical-dark mb-2">
                      Message *
                    </label>
                    <textarea
                      name="message"
                      value={formData.message}
                      onChange={handleInputChange}
                      required
                      rows={6}
                      className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-medical-primary focus:border-medical-primary transition-colors duration-200"
                      placeholder="Please provide details about your inquiry..."
                    />
                  </div>

                  <div className="flex items-center gap-2 p-4 bg-blue-50 rounded-xl border border-blue-200">
                    <Shield className="w-5 h-5 text-blue-600 flex-shrink-0" />
                    <p className="text-sm text-blue-700">
                      Your information is protected by our privacy policy and will never be shared with third parties.
                    </p>
                  </div>

                  <button
                    type="submit"
                    disabled={isSubmitting}
                    className="w-full bg-medical-primary text-white py-4 px-6 rounded-xl font-semibold hover:bg-medical-secondary disabled:bg-gray-300 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center gap-2"
                  >
                    {isSubmitting ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        Sending...
                      </>
                    ) : (
                      <>
                        <Send className="w-5 h-5" />
                        Send Message
                      </>
                    )}
                  </button>
                </form>
              )}
            </div>
          </div>

          {/* Contact Information & FAQ */}
          <div className="space-y-8">
            {/* Contact Methods */}
            <div className="bg-white rounded-2xl shadow-medical-lg p-6 border border-gray-100">
              <h3 className="text-xl font-semibold text-medical-dark mb-6">
                Get in Touch
              </h3>
              
              <div className="space-y-6">
                {contactMethods.map((method, index) => (
                  <div key={index} className="flex items-start gap-4">
                    <div className="p-3 bg-medical-light rounded-xl text-medical-primary flex-shrink-0">
                      {method.icon}
                    </div>
                    <div>
                      <h4 className="font-semibold text-medical-dark mb-1">{method.title}</h4>
                      <p className="text-medical-primary font-medium mb-1">{method.content}</p>
                      <p className="text-sm text-gray-600">{method.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick FAQ */}
            <div className="bg-white rounded-2xl shadow-medical-lg p-6 border border-gray-100">
              <h3 className="text-xl font-semibold text-medical-dark mb-6">
                Frequently Asked Questions
              </h3>
              
              <div className="space-y-4">
                {faqItems.map((item, index) => (
                  <div key={index} className="border-b border-gray-100 pb-4 last:border-b-0 last:pb-0">
                    <h4 className="font-semibold text-medical-dark mb-2 text-sm">
                      {item.question}
                    </h4>
                    <p className="text-sm text-gray-600 leading-relaxed">
                      {item.answer}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Emergency Notice */}
            <div className="bg-gradient-to-r from-red-50 to-red-100 rounded-2xl p-6 border-2 border-red-200">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-red-800 mb-2">
                    Medical Emergency?
                  </h4>
                  <p className="text-sm text-red-700 mb-3">
                    This platform is not for emergency medical situations. If you&apos;re experiencing 
                    a medical emergency, please contact your local emergency services immediately.
                  </p>
                  <p className="text-xs text-red-600">
                    Emergency: Call 911 (US) or your local emergency number
                  </p>
                </div>
              </div>
            </div>

            {/* Global Support */}
            <div className="bg-gradient-to-r from-vascular-light to-medical-light rounded-2xl p-6">
              <div className="flex items-center gap-3 mb-3">
                <Globe className="w-6 h-6 text-vascular-secondary" />
                <h4 className="font-semibold text-medical-dark">Global Support</h4>
              </div>
              <p className="text-sm text-gray-700 mb-3">
                We provide support in multiple languages and time zones to serve our global community.
              </p>
              <div className="text-xs text-gray-600 space-y-1">
                <div>🇺🇸 English: 24/7 Support</div>
                <div>🇪🇸 Español: Mon-Fri 9AM-6PM EST</div>
                <div>🇫🇷 Français: Mon-Fri 3AM-12PM EST</div>
                <div>🇮🇳 हिंदी/मराठी: Mon-Fri 10:30PM-7:30AM EST</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
