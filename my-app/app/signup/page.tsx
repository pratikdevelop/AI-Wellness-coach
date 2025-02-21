/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable react-hooks/rules-of-hooks */
'use client';
import React, { useState } from 'react'

const page = () => {
    const [username, setUsername] = useState("test34");
    const [email, setEmail] = useState('test34@yopmail.com');
    const [password, setPassword] = useState('Access@#$1234');
    const [confirmPassword, setConfirmPassword] = useState('Access@#$1234');
    

    const handleSubmit = async() => {

 // Validate password match
 if (password !== confirmPassword) {
   alert("Passwords do not match!");
   return;
 }

 // Prepare data to send to API
 const userData = {
   username: username,
   email: email,
   password: password,
 };

 try {
   const response = await fetch("http://127.0.0.1:5000/api/signup", {
     method: "POST",
     headers: {
       "Content-Type": "application/json",
     },
     body: JSON.stringify(userData),
   });

   const result = await response.json();

   if (response.ok) {
     alert("Signup successful! Please log in.");
     window.location.href = "/login"; // Redirect to login page
   } else {
     alert("Signup failed: " + result.message); // Show error message
   }
 } catch (error: any) {
   alert("An error occurred: " + error.message);
 }
    }

  return (
    <div>
          <main className="container mx-auto mt-16">
        <div className="text-center mb-10">
            <h2 className="text-3xl font-bold text-gray-800 mb-4">Create Your Account</h2>
            <p className="text-gray-600">Start your journey to better health and wellness today.</p>
        </div>

        <form id="signup-form" className="max-w-lg mx-auto bg-white shadow-lg rounded-lg p-8">
            <div className="mb-6">
                <label htmlFor="username" className="block text-gray-700 font-medium mb-2">Username</label>
                      <input type="text" id="username" name="username" placeholder="Enter your username"
                          value={
                              username
                          }
                          onChange={(e) => setUsername(e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required/>
            </div>

            <div className="mb-6">
                <label htmlFor="email" className="block text-gray-700 font-medium mb-2">Email</label>
                      <input type="email" id="email" name="email" placeholder="Enter your email"
                          value={
                              email
                          }
                          onChange={(e) => setEmail(e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required/>
            </div>

            <div className="mb-6">
                <label htmlFor="password" className="block text-gray-700 font-medium mb-2">Password</label>
                      <input type="password" id="password" name="password" placeholder="Enter your password"
                          value={
                              password
                          }
                          onChange={(e) => setPassword(e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required/>
            </div>

            <div className="mb-6">
                <label htmlFor="confirm-password" className="block text-gray-700 font-medium mb-2">Confirm Password</label>
                <input type="password" id="confirm-password" name="confirm-password"
                          placeholder="Confirm your password"
                          value={
                              confirmPassword
                          }
                          onChange={(e) => setConfirmPassword(e.target.value)}

                    className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required/>
            </div>

                  <button type="button"
                      onClick={handleSubmit}
                className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition duration-300">
                Sign Up
            </button>
        </form>

        <div className="text-center mt-6">
            <p className="text-gray-600">Already have an account? 
                <a href="/login" className="text-blue-600 font-medium hover:underline">Log In</a>
            </p>
        </div>
    </main>

    </div>
  )
}

export default page
