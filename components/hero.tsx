'use client'
import React from 'react'
import { Activity, Heart, Shield } from 'lucide-react'

const Hero = () => {
  return (
    <div className='flex flex-col items-center justify-center bg-gradient-to-br from-medical-light via-white to-vascular-light min-h-screen px-6'>
      <div className='flex flex-col lg:flex-row items-center justify-between py-24 gap-12 max-w-7xl w-full'>
        <div className='flex flex-col justify-center lg:w-1/2'>
          <h1 className='text-medical-primary font-bold mb-8 text-3xl md:text-4xl lg:text-5xl font-medical'>VarixScan – Smart Care for Your Veins</h1>
          <h2 className='text-xl md:text-2xl font-semibold mb-6 text-medical-dark'>Real-time varicose vein detection, personalized risk assessment, and health monitoring — designed for everyone.</h2>
          <p className='text-medical-dark/70 leading-relaxed'>
            VarixScan is your AI-powered varicose vein assistant. From early detection and severity analysis to personalized lifestyle advice and wearable sensor monitoring, VarixScan empowers patients and healthcare providers with actionable insights for better vein health.
          </p>
          <div className='flex gap-4 mt-8'>
            <button className='bg-medical-primary py-3 font-semibold px-6 rounded-xl hover:bg-medical-secondary transition-all duration-200 text-white shadow-medical flex items-center gap-2'>
              <Activity className='w-5 h-5' />
              Get Started
            </button>
            <button className='border border-medical-primary py-3 font-semibold px-6 rounded-xl hover:bg-medical-primary hover:text-white transition-all duration-200 text-medical-primary flex items-center gap-2'>
              <Heart className='w-5 h-5' />
              Learn More
            </button>
          </div>
        </div>
        <div className='relative w-[300px] h-[300px] md:w-[350px] md:h-[350px] lg:w-[400px] lg:h-[400px] rounded-3xl overflow-hidden shadow-medical-lg bg-gradient-to-br from-medical-primary to-vascular-secondary'>
          <video 
            autoPlay 
            loop 
            muted 
            playsInline
            className='w-full h-full object-cover'
            onError={(e) => {
              console.log('Video failed to load, showing fallback');
              e.currentTarget.style.display = 'none';
              const fallback = e.currentTarget.nextElementSibling as HTMLElement;
              if (fallback) fallback.style.display = 'flex';
            }}
          >
            <source src='/hero-video.mp4' type='video/mp4' />
            Your browser does not support the video tag.
          </video>
          {/* Fallback content if video doesn't load */}
          <div className='absolute inset-0 w-full h-full flex items-center justify-center text-white bg-gradient-to-br from-medical-primary to-vascular-secondary' style={{display: 'none'}}>
            <div className='text-center'>
              <Shield className='w-20 h-20 mx-auto mb-4 opacity-80' />
              <h3 className='text-xl font-semibold mb-2'>AI-Powered Detection</h3>
              <p className='text-sm opacity-90'>Advanced medical imaging analysis</p>
            </div>
          </div>
          
          {/* Video overlay with subtle branding */}
          <div className='absolute bottom-4 right-4 bg-white/20 backdrop-blur-sm rounded-lg px-3 py-2'>
            <p className='text-white text-xs font-semibold'>Live Demo</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Hero
