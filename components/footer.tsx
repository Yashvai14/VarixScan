'use client';

import React from 'react';
import { FaFacebookF, FaLinkedinIn, FaTwitter } from 'react-icons/fa';
import Image from 'next/image';
import Link from 'next/link';
import { Heart, Mail, Phone, MapPin } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-gradient-to-br from-medical-dark to-vascular-dark text-white">
      <div className="max-w-screen-xl mx-auto px-6 py-12 grid grid-cols-1 md:grid-cols-4 gap-8">
        <div className="md:col-span-1 flex flex-col gap-4">
          <Image src="/varixscan-logo.svg" alt="VarixScan Logo" width={200} height={70} />
          <p className="text-sm text-white/80 leading-relaxed">
            AI-powered varicose vein detection and health monitoring for better patient care.
          </p>
        </div>
        <div>
          <h4 className="font-semibold mb-4 text-lg">NAVIGATION</h4>
          <ul className="space-y-3 text-sm">
            <li><Link href="/" className="hover:text-medical-accent transition-colors">Home</Link></li>
            <li><Link href="/about" className="hover:text-medical-accent transition-colors">About Us</Link></li>
            <li><Link href="/vericose" className="hover:text-medical-accent transition-colors">AI Detection</Link></li>
            <li><Link href="/risk-assessment" className="hover:text-medical-accent transition-colors">Risk Assessment</Link></li>
            <li><Link href="/dashboard" className="hover:text-medical-accent transition-colors">Dashboard</Link></li>
          </ul>
        </div>
        <div>
          <h4 className="font-semibold mb-4 text-lg">SERVICES</h4>
          <ul className="space-y-3 text-sm">
            <li><Link href="/vericose" className="hover:text-medical-accent transition-colors">Vein Detection</Link></li>
            <li><Link href="/wearable-monitoring" className="hover:text-medical-accent transition-colors">Wearable Monitoring</Link></li>
            <li><Link href="/multilingual-assistant" className="hover:text-medical-accent transition-colors">AI Assistant</Link></li>
            <li><Link href="/reports" className="hover:text-medical-accent transition-colors">Health Reports</Link></li>
          </ul>
        </div>
        <div>
          <h4 className="font-semibold mb-4 text-lg">CONTACT INFO</h4>
          <ul className="space-y-3 text-sm">
            <li className="flex items-center gap-2">
              <Mail className="w-4 h-4 text-medical-accent" />
              support@varixscan.com
            </li>
            <li className="flex items-center gap-2">
              <Phone className="w-4 h-4 text-medical-accent" />
              +1 (555) 123-4567
            </li>
            <li className="flex items-center gap-2">
              <MapPin className="w-4 h-4 text-medical-accent" />
              Medical District, Healthcare City
            </li>
          </ul>
        </div>
      </div>
      <div className="border-t border-white/20 mt-8">
        <div className="max-w-screen-xl mx-auto px-6 py-6 flex flex-col md:flex-row justify-between items-center">
          <p className="text-sm text-white/80 flex items-center gap-2">
            <Heart className="w-4 h-4 text-medical-accent" />
            &copy; 2025 VarixScan Medical Clinics. All Rights Reserved.
          </p>
          <div className="flex space-x-4 mt-4 md:mt-0">
            <a href="#" className="text-white/60 hover:text-medical-accent transition-colors p-2 rounded-full hover:bg-white/10">
              <FaFacebookF />
            </a>
            <a href="#" className="text-white/60 hover:text-medical-accent transition-colors p-2 rounded-full hover:bg-white/10">
              <FaLinkedinIn />
            </a>
            <a href="#" className="text-white/60 hover:text-medical-accent transition-colors p-2 rounded-full hover:bg-white/10">
              <FaTwitter />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
