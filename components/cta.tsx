import React from 'react'
import { Activity, Heart, ArrowRight } from 'lucide-react'

const Cta = () => {
  return (
    <div>
        <div className='flex flex-col items-center justify-center py-18 mb-20 bg-gradient-to-r from-medical-primary to-vascular-secondary py-16'>
            <h1 className='text-5xl text-white font-bold mb-8 font-medical text-center max-w-4xl'>Try VarixScan – Your Vein Health Companion!</h1>
            <p className='text-center text-[18px] text-white/90 max-w-3xl leading-relaxed'>
                VarixScan is your AI-powered assistant for varicose vein care, offering early detection, personalized risk assessment, and comprehensive health monitoring.
            </p>
            <div className='flex gap-4 mt-8'>
                <button className='bg-white text-medical-primary py-3 font-semibold px-6 rounded-xl hover:bg-gray-100 transition-all duration-200 shadow-medical flex items-center gap-2'>
                    <Activity className='w-5 h-5' />
                    Get Started
                    <ArrowRight className='w-4 h-4' />
                </button>
                <button className='border-2 border-white text-white py-3 font-semibold px-6 rounded-xl hover:bg-white hover:text-medical-primary transition-all duration-200 flex items-center gap-2'>
                    <Heart className='w-5 h-5' />
                    Learn More
                </button>
            </div>
        </div>
    </div>
  )
}

export default Cta
