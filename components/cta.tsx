import React from 'react'
import Link from 'next/link'
import { Activity, Heart, ArrowRight } from 'lucide-react'

const Cta = () => {
  return (
    <section className="w-full">
        <div className='flex flex-col items-center justify-center py-16 sm:py-20 md:py-24 lg:py-32 mb-0 bg-gradient-to-r from-medical-primary to-vascular-secondary px-4 sm:px-6 lg:px-8'>
            <div className="max-w-4xl mx-auto text-center">
                <h1 className='text-2xl sm:text-3xl md:text-4xl lg:text-5xl xl:text-6xl text-white font-bold mb-6 sm:mb-8 font-medical leading-tight'>
                    Try VarixScan – Your Vein Health Companion!
                </h1>
                <p className='text-center text-sm sm:text-base md:text-lg lg:text-xl text-white/90 max-w-3xl mx-auto leading-relaxed mb-8 sm:mb-10'>
                    VarixScan is your AI-powered assistant for varicose vein care, offering early detection, personalized risk assessment, and comprehensive health monitoring.
                </p>
                <div className='flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center'>
                    <Link 
                        href='/vericose' 
                        className='bg-white text-medical-primary py-3 sm:py-4 font-semibold px-6 sm:px-8 rounded-xl hover:bg-gray-100 transition-all duration-200 shadow-medical flex items-center justify-center gap-2 text-decoration-none text-sm sm:text-base'
                    >
                        <Activity className='w-4 h-4 sm:w-5 sm:h-5' />
                        Get Started
                        <ArrowRight className='w-3 h-3 sm:w-4 sm:h-4' />
                    </Link>
                    <Link 
                        href='/about' 
                        className='border-2 border-white text-white py-3 sm:py-4 font-semibold px-6 sm:px-8 rounded-xl hover:bg-white hover:text-medical-primary transition-all duration-200 flex items-center justify-center gap-2 text-decoration-none text-sm sm:text-base'
                    >
                        <Heart className='w-4 h-4 sm:w-5 sm:h-5' />
                        Learn More
                    </Link>
                </div>
            </div>
        </div>
    </section>
  )
}

export default Cta
