/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        wnba: {
          orange: '#FF6B2B',
          dark: '#0A0E1A',
          card: '#111827',
          border: '#1F2937',
        }
      }
    },
  },
  plugins: [],
}
