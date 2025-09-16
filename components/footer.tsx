'use client';

import React from 'react';
import { FaFacebookF, FaLinkedinIn, FaTwitter } from 'react-icons/fa';
import Image from 'next/image';
import Link from 'next/link';
import { Heart, Mail, Phone, MapPin } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-gradient-to-br from-medical-dark to-vascular-dark text-white">
      <div className="max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 lg:py-12 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6 sm:gap-8">
        <div className="sm:col-span-2 md:col-span-1 flex flex-col gap-3 sm:gap-4">
          <Image src="/varixscan-logo.svg" alt="VarixScan Logo" width={180} height={60} className="sm:w-[200px] sm:h-[70px]" />
          <p className="text-xs sm:text-sm text-white/80 leading-relaxed max-w-sm">
            AI-powered varicose vein detection and health monitoring for better patient care.
          </p>
        </div>
        <div>
          <h4 className="font-semibold mb-3 sm:mb-4 text-sm sm:text-base lg:text-lg">NAVIGATION</h4>
          <ul className="space-y-2 sm:space-y-3 text-xs sm:text-sm">
            <li><Link href="/" className="hover:text-medical-accent transition-colors">Home</Link></li>
            <li><Link href="/about" className="hover:text-medical-accent transition-colors">About Us</Link></li>
            <li><Link href="/vericose" className="hover:text-medical-accent transition-colors">AI Detection</Link></li>
            <li><Link href="/risk-assessment" className="hover:text-medical-accent transition-colors">Risk Assessment</Link></li>
            <li><Link href="/dashboard" className="hover:text-medical-accent transition-colors">Dashboard</Link></li>
          </ul>
        </div>
        <div>
          <h4 className="font-semibold mb-3 sm:mb-4 text-sm sm:text-base lg:text-lg">SERVICES</h4>
          <ul className="space-y-2 sm:space-y-3 text-xs sm:text-sm">
            <li><Link href="/vericose" className="hover:text-medical-accent transition-colors">Vein Detection</Link></li>
            <li><Link href="/wearable-monitoring" className="hover:text-medical-accent transition-colors">Wearable Monitoring</Link></li>
            <li><Link href="/multilingual-assistant" className="hover:text-medical-accent transition-colors">AI Assistant</Link></li>
            <li><Link href="/reports" className="hover:text-medical-accent transition-colors">Health Reports</Link></li>
          </ul>
        </div>
        <div className="sm:col-span-2 md:col-span-1">
          <h4 className="font-semibold mb-3 sm:mb-4 text-sm sm:text-base lg:text-lg">CONTACT INFO</h4>
          <ul className="space-y-2 sm:space-y-3 text-xs sm:text-sm">
            <li className="flex items-center gap-2">
              <Mail className="w-3 h-3 sm:w-4 sm:h-4 text-medical-accent flex-shrink-0" />
              <span className="break-all">support@varixscan.com</span>
            </li>
            <li className="flex items-center gap-2">
              <Phone className="w-3 h-3 sm:w-4 sm:h-4 text-medical-accent flex-shrink-0" />
              <span>+1 (555) 123-4567</span>
            </li>
            <li className="flex items-center gap-2">
              <MapPin className="w-3 h-3 sm:w-4 sm:h-4 text-medical-accent flex-shrink-0" />
              <span>Medical District, Healthcare City</span>
            </li>
          </ul>
        </div>
      </div>
      <div className="border-t border-white/20 mt-6 sm:mt-8">
        <div className="max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-xs sm:text-sm text-white/80 flex items-center gap-2 text-center md:text-left">
            <Heart className="w-3 h-3 sm:w-4 sm:h-4 text-medical-accent" />
            &copy; 2025 VarixScan Medical Clinics. All Rights Reserved.
          </p>
          <div className="flex space-x-3 sm:space-x-4">
            <a href="#" className="text-white/60 hover:text-medical-accent transition-colors p-2 rounded-full hover:bg-white/10">
              <FaFacebookF className="text-sm sm:text-base" />
            </a>
            <a href="#" className="text-white/60 hover:text-medical-accent transition-colors p-2 rounded-full hover:bg-white/10">
              <FaLinkedinIn className="text-sm sm:text-base" />
            </a>
            <a href="#" className="text-white/60 hover:text-medical-accent transition-colors p-2 rounded-full hover:bg-white/10">
              <FaTwitter className="text-sm sm:text-base" />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
