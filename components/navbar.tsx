'use client';
import React, { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { 
  Activity, 
  Shield, 
  Watch, 
  MessageCircle, 
  FileText,
  Stethoscope,
  Heart,
  Mail,
  BarChart3,
  Menu,
  X,
  ChevronDown
} from 'lucide-react';

const NavBar = () => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isMobileDropdownOpen, setIsMobileDropdownOpen] = useState(false);

  return (
    <nav className="bg-gradient-to-r from-medical-light to-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4 md:py-6">
          {/* Logo */}
          <div className="flex-shrink-0">
            <Link href="/" className="hover:opacity-80 transition-opacity duration-200">
              <Image 
                src="/varixscan-logo.svg" 
                alt="VarixScan Logo" 
                width={140} 
                height={45}
                className="md:w-[180px] md:h-[60px]" 
                priority 
              />
            </Link>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden lg:flex lg:items-center">
            {/* Navigation Links */}
            <div className="bg-white rounded-full shadow-medical-lg border border-gray-100 px-6 py-2 mr-4">
              <ul className="flex space-x-4 text-sm font-semibold relative font-medical">
                {/* Features Dropdown */}
                <li
                  className="relative cursor-pointer group"
                  onMouseEnter={() => setIsDropdownOpen(true)}
                  onMouseLeave={() => setIsDropdownOpen(false)}
                >
                  <div className="flex items-center gap-1 hover:text-medical-primary text-medical-dark transition-all duration-300 py-2 px-2 rounded-lg hover:bg-medical-light/50">
                    <Stethoscope className="w-4 h-4" />
                    Features
                  </div>
                  {isDropdownOpen && (
                    <ul className="absolute top-full left-0 bg-white shadow-medical-lg rounded-xl py-2 w-80 z-50 border border-gray-100 mt-1">
                      <li>
                        <Link
                          href="/vericose"
                          className="flex items-center gap-3 px-4 py-3 hover:bg-vascular-light transition-colors duration-200"
                        >
                          <Activity className="w-5 h-5 text-medical-primary" />
                          <div>
                            <span className="font-medium text-medical-dark">AI Vein Detection</span>
                            <p className="text-xs text-gray-500 mt-0.5">Advanced AI-powered vein analysis</p>
                          </div>
                        </Link>
                      </li>
                      <li>
                        <Link
                          href="/risk-assessment"
                          className="flex items-center gap-3 px-4 py-3 hover:bg-vascular-light transition-colors duration-200"
                        >
                          <Shield className="w-5 h-5 text-medical-warning" />
                          <div>
                            <span className="font-medium text-medical-dark">Risk Assessment</span>
                            <p className="text-xs text-gray-500 mt-0.5">Personalized health risk evaluation</p>
                          </div>
                        </Link>
                      </li>
                      <li>
                        <Link
                          href="/wearable-monitoring"
                          className="flex items-center gap-3 px-4 py-3 hover:bg-vascular-light transition-colors duration-200"
                        >
                          <Watch className="w-5 h-5 text-medical-accent" />
                          <div>
                            <span className="font-medium text-medical-dark">Wearable Monitoring</span>
                            <p className="text-xs text-gray-500 mt-0.5">Continuous health tracking</p>
                          </div>
                        </Link>
                      </li>
                      <li>
                        <Link
                          href="/multilingual-assistant"
                          className="flex items-center gap-3 px-4 py-3 hover:bg-vascular-light transition-colors duration-200"
                        >
                          <MessageCircle className="w-5 h-5 text-vascular-secondary" />
                          <div>
                            <span className="font-medium text-medical-dark">AI Assistant</span>
                            <p className="text-xs text-gray-500 mt-0.5">Multilingual medical support</p>
                          </div>
                        </Link>
                      </li>
                      <li>
                        <Link
                          href="/reports"
                          className="flex items-center gap-3 px-4 py-3 hover:bg-vascular-light transition-colors duration-200"
                        >
                          <FileText className="w-5 h-5 text-medical-success" />
                          <div>
                            <span className="font-medium text-medical-dark">Health Reports</span>
                            <p className="text-xs text-gray-500 mt-0.5">Comprehensive medical reports</p>
                          </div>
                        </Link>
                      </li>
                    </ul>
                  )}
                </li>

                {/* Other Pages */}
                <li>
                  <Link
                    href="/about"
                    className="flex items-center gap-1 hover:text-medical-primary text-medical-dark transition-all duration-300 py-2 px-2 rounded-lg hover:bg-medical-light/50"
                  >
                    <Heart className="w-4 h-4" />
                    About
                  </Link>
                </li>
                <li>
                  <Link
                    href="/contact"
                    className="flex items-center gap-1 hover:text-medical-primary text-medical-dark transition-all duration-300 py-2 px-2 rounded-lg hover:bg-medical-light/50"
                  >
                    <Mail className="w-4 h-4" />
                    Contact
                  </Link>
                </li>
              </ul>
            </div>
          </div>
          
          {/* Dashboard Button - separate from navigation */}
          <div className="hidden lg:flex">
            <Link href="/dashboard">
              <button className="flex items-center gap-2 py-2 px-6 text-white font-semibold cursor-pointer bg-medical-primary rounded-full hover:bg-medical-secondary transition-all duration-200 shadow-medical">
                <BarChart3 className="w-4 h-4" />
                Dashboard
              </button>
            </Link>
          </div>

          {/* Mobile menu button */}
          <div className="lg:hidden">
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="p-2 rounded-md text-medical-dark hover:text-medical-primary hover:bg-medical-light/50 transition-colors duration-200"
            >
              {isMobileMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {isMobileMenuOpen && (
        <div className="lg:hidden bg-white border-t border-gray-200 shadow-lg">
          <div className="px-4 py-6 space-y-4">
            {/* Features dropdown for mobile */}
            <div>
              <button
                onClick={() => setIsMobileDropdownOpen(!isMobileDropdownOpen)}
                className="flex items-center justify-between w-full px-3 py-2 text-left text-medical-dark font-semibold hover:text-medical-primary transition-colors duration-200"
              >
                <div className="flex items-center gap-2">
                  <Stethoscope className="w-4 h-4" />
                  Features
                </div>
                <ChevronDown className={`w-4 h-4 transform transition-transform duration-200 ${isMobileDropdownOpen ? 'rotate-180' : ''}`} />
              </button>
              {isMobileDropdownOpen && (
                <div className="mt-2 pl-4 space-y-2">
                  <Link
                    href="/vericose"
                    className="flex items-center gap-3 px-3 py-2 text-sm text-medical-dark hover:text-medical-primary hover:bg-medical-light/50 rounded-lg transition-colors duration-200"
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    <Activity className="w-4 h-4 text-medical-primary" />
                    AI Vein Detection
                  </Link>
                  <Link
                    href="/risk-assessment"
                    className="flex items-center gap-3 px-3 py-2 text-sm text-medical-dark hover:text-medical-primary hover:bg-medical-light/50 rounded-lg transition-colors duration-200"
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    <Shield className="w-4 h-4 text-medical-warning" />
                    Risk Assessment
                  </Link>
                  <Link
                    href="/wearable-monitoring"
                    className="flex items-center gap-3 px-3 py-2 text-sm text-medical-dark hover:text-medical-primary hover:bg-medical-light/50 rounded-lg transition-colors duration-200"
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    <Watch className="w-4 h-4 text-medical-accent" />
                    Wearable Monitoring
                  </Link>
                  <Link
                    href="/multilingual-assistant"
                    className="flex items-center gap-3 px-3 py-2 text-sm text-medical-dark hover:text-medical-primary hover:bg-medical-light/50 rounded-lg transition-colors duration-200"
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    <MessageCircle className="w-4 h-4 text-vascular-secondary" />
                    AI Assistant
                  </Link>
                  <Link
                    href="/reports"
                    className="flex items-center gap-3 px-3 py-2 text-sm text-medical-dark hover:text-medical-primary hover:bg-medical-light/50 rounded-lg transition-colors duration-200"
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    <FileText className="w-4 h-4 text-medical-success" />
                    Health Reports
                  </Link>
                </div>
              )}
            </div>

            {/* Other menu items */}
            <Link
              href="/about"
              className="flex items-center gap-2 px-3 py-2 text-medical-dark font-semibold hover:text-medical-primary hover:bg-medical-light/50 rounded-lg transition-colors duration-200"
              onClick={() => setIsMobileMenuOpen(false)}
            >
              <Heart className="w-4 h-4" />
              About
            </Link>
            <Link
              href="/contact"
              className="flex items-center gap-2 px-3 py-2 text-medical-dark font-semibold hover:text-medical-primary hover:bg-medical-light/50 rounded-lg transition-colors duration-200"
              onClick={() => setIsMobileMenuOpen(false)}
            >
              <Mail className="w-4 h-4" />
              Contact
            </Link>

            {/* Dashboard button for mobile */}
            <Link
              href="/dashboard"
              onClick={() => setIsMobileMenuOpen(false)}
              className="block w-full"
            >
              <button className="flex items-center justify-center gap-2 w-full py-3 px-6 text-white font-semibold bg-medical-primary rounded-xl hover:bg-medical-secondary transition-all duration-200 shadow-medical">
                <BarChart3 className="w-4 h-4" />
                Dashboard
              </button>
            </Link>
          </div>
        </div>
      )}
    </nav>
  );
};

export default NavBar;
