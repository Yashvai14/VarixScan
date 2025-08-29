import React from 'react'

const Cta = () => {
  return (
    <div>
        <div className='flex flex-col items-center justify-center py-18 mb-20 bg-gray-50 py-16'>
            <h1 className='text-5xl text-lime-500 font-bold mb-8'>Try VarixScan – Your Vein Health Companion!</h1>
            <p className='text-center text-[18px]'>
                VarixScan is your AI-powered assistant for varicose vein care, offering early detection, personalized risk assessment
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
    </div>
  )
}

export default Cta
