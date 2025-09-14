/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        border: '#e2e8f0',
        background: '#ffffff',
        foreground: '#1a202c',
      },
    },
  },
  plugins: [],
};
