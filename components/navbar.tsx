'use client';
import React, { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { Sprout, Sun, Map, TrendingUp, Calendar } from 'lucide-react'; // Optional: You can swap icons for healthcare if needed

const NavBar = () => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  return (
    <nav className="flex justify-center items-center py-8 p-4">
      <div
        className="flex justify-between shadow-xl py-4 px-6 rounded-full bg-white items-center"
        style={{ width: '1200px' }}
      >
        {/* Logo */}
        <div>
          <Link href="/">
            <Image src="/varixscan-logo.png" alt="VarixScan Logo" width={150} height={50} />
          </Link>
        </div>

        {/* Navigation Links */}
        <ul className="flex space-x-6 text-[16px] font-semibold relative">
          {/* Features Dropdown */}
          <li
            className="relative cursor-pointer hover:font-bold hover:text-gray-800 text-gray-600 transition-all duration-300"
            onMouseEnter={() => setIsDropdownOpen(true)}
            onMouseLeave={() => setIsDropdownOpen(false)}
          >
            Features
            {isDropdownOpen && (
              <ul className="absolute top-8 left-0 bg-white shadow-lg rounded-xl py-2 w-72 z-50">
                <li>
                  <Link
                    href="/ai-diagnosis"
                    className="flex items-center gap-3 px-4 py-2 hover:bg-gray-100"
                  >
                    <Sprout className="w-5 h-5 text-green-600" />
                    <span>AI Vein Detection</span>
                  </Link>
                </li>
                <li>
                  <Link
                    href="/risk-assessment"
                    className="flex items-center gap-3 px-4 py-2 hover:bg-gray-100"
                  >
                    <Sun className="w-5 h-5 text-yellow-500" />
                    <span>Personalized Risk Assessment</span>
                  </Link>
                </li>
                <li>
                  <Link
                    href="/wearable-monitoring"
                    className="flex items-center gap-3 px-4 py-2 hover:bg-gray-100"
                  >
                    <Map className="w-5 h-5 text-blue-600" />
                    <span>Wearable Health Monitoring</span>
                  </Link>
                </li>
                <li>
                  <Link
                    href="/multilingual-assistant"
                    className="flex items-center gap-3 px-4 py-2 hover:bg-gray-100"
                  >
                    <TrendingUp className="w-5 h-5 text-purple-600" />
                    <span>Multilingual AI Assistant</span>
                  </Link>
                </li>
                <li>
                  <Link
                    href="/health-reports"
                    className="flex items-center gap-3 px-4 py-2 hover:bg-gray-100"
                  >
                    <Calendar className="w-5 h-5 text-red-500" />
                    <span>Downloadable Health Reports</span>
                  </Link>
                </li>
              </ul>
            )}
          </li>

          {/* Other Pages */}
          <li>
            <Link
              href="/about"
              className="hover:font-bold hover:text-gray-800 text-gray-600"
            >
              About
            </Link>
          </li>
          <li>
            <Link
              href="/contact"
              className="hover:font-bold hover:text-gray-800 text-gray-600"
            >
              Contact
            </Link>
          </li>
        </ul>

        {/* Dashboard Button */}
        <div>
          <Link href="/dashboard">
            <button className="py-2 px-6 text-gray-800 font-bold cursor-pointer bg-lime-400 rounded-3xl hover:bg-lime-500 transition">
              Dashboard
            </button>
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;
