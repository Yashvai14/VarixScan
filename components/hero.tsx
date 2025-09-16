'use client'
import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import { Activity, Heart, Shield } from 'lucide-react'

const Hero = () => {
  const [isClient, setIsClient] = useState(false)
  const [videoError, setVideoError] = useState(false)

  useEffect(() => {
    setIsClient(true)
  }, [])

  const handleVideoError = () => {
    setVideoError(true)
  }

  return (
    <div className='flex flex-col items-center justify-center bg-gradient-to-br from-medical-light via-white to-vascular-light min-h-screen px-4 sm:px-6 lg:px-8'>
      <div className='flex flex-col lg:flex-row items-center justify-between py-12 sm:py-16 md:py-20 lg:py-24 gap-8 sm:gap-10 lg:gap-12 max-w-7xl w-full'>
        <div className='flex flex-col justify-center lg:w-1/2 text-center lg:text-left'>
          <h1 className='text-medical-primary font-bold mb-6 sm:mb-8 text-2xl sm:text-3xl md:text-4xl lg:text-5xl xl:text-6xl font-medical leading-tight'>
            VarixScan – Smart Care for Your Veins
          </h1>
          <h2 className='text-lg sm:text-xl md:text-2xl font-semibold mb-4 sm:mb-6 text-medical-dark leading-relaxed'>
            Real-time varicose vein detection, personalized risk assessment, and health monitoring — designed for everyone.
          </h2>
          <p className='text-sm sm:text-base text-medical-dark/70 leading-relaxed mb-6 sm:mb-8 max-w-2xl'>
            VarixScan is your AI-powered varicose vein assistant. From early detection and severity analysis to personalized lifestyle advice and wearable sensor monitoring, VarixScan empowers patients and healthcare providers with actionable insights for better vein health.
          </p>
          <div className='flex flex-col sm:flex-row gap-3 sm:gap-4'>
            <Link 
              href='/vericose' 
              className='bg-medical-primary py-3 sm:py-3 font-semibold px-6 sm:px-6 rounded-xl hover:bg-medical-secondary transition-all duration-200 text-white shadow-medical flex items-center justify-center gap-2 text-decoration-none text-sm sm:text-base'
            >
              <Activity className='w-4 h-4 sm:w-5 sm:h-5' />
              Get Started
            </Link>
            <Link 
              href='/about' 
              className='border border-medical-primary py-3 sm:py-3 font-semibold px-6 sm:px-6 rounded-xl hover:bg-medical-primary hover:text-white transition-all duration-200 text-medical-primary flex items-center justify-center gap-2 text-decoration-none text-sm sm:text-base'
            >
              <Heart className='w-4 h-4 sm:w-5 sm:h-5' />
              Learn More
            </Link>
          </div>
        </div>
        <div className='relative w-[280px] h-[280px] sm:w-[320px] sm:h-[320px] md:w-[350px] md:h-[350px] lg:w-[400px] lg:h-[400px] xl:w-[450px] xl:h-[450px] rounded-2xl sm:rounded-3xl overflow-hidden shadow-medical-lg bg-gradient-to-br from-medical-primary to-vascular-secondary flex-shrink-0'>
          {isClient && !videoError ? (
            <video 
              autoPlay 
              loop 
              muted 
              playsInline
              className='w-full h-full object-cover'
              onError={handleVideoError}
              suppressHydrationWarning
            >
              <source src='/hero-video.mp4' type='video/mp4' />
              Your browser does not support the video tag.
            </video>
          ) : (
            <div className='absolute inset-0 w-full h-full flex items-center justify-center text-white bg-gradient-to-br from-medical-primary to-vascular-secondary'>
              <div className='text-center px-4'>
                <Shield className='w-16 h-16 sm:w-20 sm:h-20 mx-auto mb-3 sm:mb-4 opacity-80' />
                <h3 className='text-lg sm:text-xl font-semibold mb-2'>AI-Powered Detection</h3>
                <p className='text-xs sm:text-sm opacity-90'>Advanced medical imaging analysis</p>
              </div>
            </div>
          )}
          
          {/* Video overlay with subtle branding - only show when video is playing */}
          {isClient && !videoError && (
            <div className='absolute bottom-3 right-3 sm:bottom-4 sm:right-4 bg-white/20 backdrop-blur-sm rounded-lg px-2 py-1 sm:px-3 sm:py-2'>
              <p className='text-white text-xs font-semibold'>Live Demo</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Hero
