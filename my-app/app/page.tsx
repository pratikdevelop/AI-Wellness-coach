/* eslint-disable react/no-unescaped-entities */
"use client";
export default function Home() {
  return (
    <div className="flex relative flex-col p-3 overflow-y-auto">
      <section className="gradient-bg text-white py-24 text-center">
        <div className="container mx-auto">
          <h2 className="text-5xl font-bold mb-6">Achieve Your Fitness Goals</h2>
          <p className="text-lg mb-8 max-w-3xl mx-auto">
            Transform your health and well-being with AI-powered insights and
            personalized coaching.
          </p>
          <a
            href="/signup"
            className="bg-white text-blue-600 px-10 py-4 rounded-full font-semibold shadow-lg hover:shadow-xl hover:bg-gray-100 transition duration-300"
          >
            Get Started Today
          </a>
        </div>
      </section>

      <section className="py-16 bg-gray-50">
        <div className="container mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">
            Why Choose AI Wellness Coach?
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-10">
            {["AI-Powered Insights", "Track Progress", "Personalized Goals"].map((title, index) => (
              <div key={index} className="bg-white shadow-md rounded-lg p-6 text-center hover:scale-105 transition-transform duration-300">
                <div className="flex items-center justify-center bg-blue-600 text-white w-16 h-16 rounded-full mx-auto mb-6">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 10h11M9 21V3m-6 6h18M3 15h11M3 21h11m0 0V9m0 0h7" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold mb-4">{title}</h3>
                <p className="text-gray-600">
                  {index === 0 && "Get actionable recommendations based on your unique health data."}
                  {index === 1 && "Monitor your BMI, activity levels, and calorie goals over time."}
                  {index === 2 && "Set and achieve realistic fitness goals with AI guidance."}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="bg-gray-100 py-16">
        <div className="container mx-auto text-center">
          <h2 className="text-3xl font-bold mb-12">What Users Are Saying</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {["This app changed my life! I lost weight and built healthier habits.", "Tracking my progress has never been this easy and intuitive.", "The personalized AI recommendations are spot on."].map((quote, index) => (
              <div key={index} className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition duration-300">
                <p className="text-gray-600 italic">"{quote}"</p>
                <p className="mt-4 font-bold text-blue-600">- {index === 0 ? "Emily R." : index === 1 ? "James W." : "Sarah L."}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="py-16 text-center">
        <div className="container mx-auto">
          <h2 className="text-3xl font-bold mb-6">Join Us Today</h2>
          <p className="text-lg mb-8 max-w-2xl mx-auto">
            Sign up now and take the first step towards a healthier, happier you!
          </p>
          <a
            href="/signup"
            className="bg-blue-600 text-white px-10 py-4 rounded-full font-semibold shadow-lg hover:shadow-xl hover:bg-blue-700 transition duration-300"
          >
            Sign Up Now
          </a>
        </div>
      </section>

      <footer className="bg-blue-600 text-white py-8 text-center">
        <div className="container mx-auto">
          <p className="font-medium">&copy; 2025 AI Wellness Coach. All Rights Reserved.</p>
          <p className="text-sm mt-2">Email us: support@aiwellnesscoach.com</p>
        </div>
      </footer>
    </div>


  );
}
