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
  BarChart3
} from 'lucide-react';

const NavBar = () => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  return (
    <nav className="flex justify-center items-center py-6 p-4 bg-gradient-to-r from-medical-light to-white">
      <div
        className="flex justify-between shadow-medical-lg py-4 px-6 rounded-full bg-white items-center border border-gray-100"
        style={{ width: '1200px' }}
      >
        {/* Logo */}
        <div>
          <Link href="/" className="hover:opacity-80 transition-opacity duration-200">
            <Image src="/varixscan-logo.svg" alt="VarixScan Logo" width={180} height={60} priority />
          </Link>
        </div>

        {/* Navigation Links */}
        <ul className="flex space-x-6 text-[16px] font-semibold relative font-medical">
          {/* Features Dropdown */}
          <li
            className="relative cursor-pointer group"
            onMouseEnter={() => setIsDropdownOpen(true)}
            onMouseLeave={() => setIsDropdownOpen(false)}
          >
            <div className="flex items-center gap-1 hover:text-medical-primary text-medical-dark transition-all duration-300 py-2 px-3 rounded-lg hover:bg-medical-light/50">
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
              className="flex items-center gap-1 hover:text-medical-primary text-medical-dark transition-all duration-300 py-2 px-3 rounded-lg hover:bg-medical-light/50"
            >
              <Heart className="w-4 h-4" />
              About
            </Link>
          </li>
          <li>
            <Link
              href="/contact"
              className="flex items-center gap-1 hover:text-medical-primary text-medical-dark transition-all duration-300 py-2 px-3 rounded-lg hover:bg-medical-light/50"
            >
              <Mail className="w-4 h-4" />
              Contact
            </Link>
          </li>
        </ul>

        {/* Dashboard Button */}
        <div>
          <Link href="/dashboard">
            <button className="flex items-center gap-2 py-2 px-6 text-white font-semibold cursor-pointer bg-medical-primary rounded-full hover:bg-medical-secondary transition-all duration-200 shadow-medical">
              <BarChart3 className="w-4 h-4" />
              Dashboard
            </button>
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;
