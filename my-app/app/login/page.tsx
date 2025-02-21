/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable react/no-unescaped-entities */
/* eslint-disable react-hooks/rules-of-hooks */
'use client';
import React, { useState } from 'react'

const page = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const loginUser = async() => {
         try {
           const response = await fetch(
             "http://127.0.0.1:5000/api/login",
             {
               method: "POST",
               headers: {
                 "Content-Type": "application/json",
               },
               body: JSON.stringify({
                 email,
                 password,
               }),
             }
           );

           const result = await response.json();

           if (response.ok) {
             // Store the JWT token in localStorage
             localStorage.setItem("authToken", result.token);

             console.log("Login successful! Redirecting...");
             window.location.href = "/dashboard"; // Redirect to the user dashboard after successful login
           } else {
             console.log("Login failed: " + result.error); // Show error message
           }
         } catch (error: any) {
           console.log("An error occurred: " + error.message);
         }
        
    }
    
  return (
    <div>
          <main className="container mx-auto mt-16">
        <div className="text-center mb-10">
            <h2 className="text-3xl font-bold text-gray-800 mb-4">Welcome Back!</h2>
            <p className="text-gray-600">Log in to your account and start your wellness journey.</p>
        </div>

        <form id="login-form" className="max-w-lg mx-auto bg-white shadow-lg rounded-lg p-8">
            <div className="mb-6">
                <label htmlFor="email" className="block text-gray-700 font-medium mb-2">
                    Enter your email
                </label>
                      <input type="text" id="email" name="email" value={email} onChange={
                          (e) => setEmail(e.target.value)
                } placeholder="Enter your email"
                    className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required/>
            </div>

            <div className="mb-6">
                <label htmlFor="password" className="block text-gray-700 font-medium mb-2">Password</label>
                <input type="password" id="password" name="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Enter your password"
                    className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required/>
            </div>

            <button type="button" onClick={loginUser}
                className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition duration-300">
                Log In
            </button>
        </form>

        <div className="text-center mt-6">
            <p className="text-gray-600">Don't have an account? 
                <a href="/signup" className="text-blue-600 font-medium hover:underline">Sign Up</a>
            </p>
        </div>
    </main>
    </div>
  )
}

export default page
