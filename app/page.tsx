import React from 'react'
import Hero from '../components/hero'
import Cta from '../components/cta'
import Feature from '../components/feature'
import Faq from '../components/faq'
import Footer from '../components/footer'

const home = () => {
  return (
    <>
    <Hero />
    <Feature />
    <Faq />
    <Cta/>
    <Footer/>
    </>
  )
}

export default home