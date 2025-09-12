/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
     "./app/**/*.{js,ts,jsx,tsx}",
  "./pages/**/*.{js,ts,jsx,tsx}",
  "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Medical color scheme
        medical: {
          primary: '#2563eb', // Professional blue
          secondary: '#1e40af', // Darker blue
          accent: '#3b82f6', // Light blue
          success: '#059669', // Medical green
          warning: '#d97706', // Amber
          danger: '#dc2626', // Medical red
          light: '#f8fafc', // Light gray
          dark: '#1e293b', // Dark gray
        },
        // Vascular theme colors
        vascular: {
          primary: '#1e40af', // Deep blue
          secondary: '#3730a3', // Indigo
          accent: '#0ea5e9', // Sky blue
          light: '#e0f2fe', // Very light blue
          dark: '#0c4a6e', // Dark blue
        }
      },
      fontFamily: {
        'medical': ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'medical': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'medical-lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
      }
    },
  },
  plugins: [],
}

