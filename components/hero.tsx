import React from 'react'

const Hero = () => {
  return (
    <div className='flex flex-col items-center justify-center'>
      <div className='flex items-center justify-between py-24' style={{width: "1200px"}}>
        <div className='flex flex-col justify-center' style={{width:"600px"}}>
          <h1 className='text-lime-500 font-bold mb-8 text-5xl'>VarixScan – Smart Care for Your Veins</h1>
          <h2 className='text-2xl font-semibold mb-6 text-gray-700'>Real-time varicose vein detection, personalized risk assessment, and health monitoring — designed for everyone.</h2>
          <p className='text-gray-600'>
            VarixScan is your AI-powered varicose vein assistant. From early detection and severity analysis to personalized lifestyle advice and wearable sensor monitoring, VarixScan empowers patients and healthcare providers with actionable insights for better vein health.
          </p>
          <div className='flex gap-4 mt-8'>
            <button className='bg-lime-500 py-3 font-semibold px-6 rounded-xl hover:bg-lime-600 text-white'>
              Get Started
            </button>
            <button className='border border-lime-400 py-3 font-semibold px-6 rounded-xl hover:bg-white hover:text-gray-600 text-gray-500'>
              Learn More
            </button>
          </div>
        </div>
        <div className='w-[400px] h-[400px] rounded-3xl bg-gray-300'>
          {/* You can replace this with an image or illustration */}
        </div>
      </div>
    </div>
  )
}

export default Hero
