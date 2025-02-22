/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable react-hooks/rules-of-hooks */
"use client";
import React, { useEffect, useState } from "react";
import dynamic from 'next/dynamic';
const BarChart = dynamic(() => import('@mui/x-charts').then(mod => mod.BarChart), { ssr: false });

const Page = () => {

  const [user, setUser] = useState<any>({});
  const [predictionData, setPredictionData] = useState<any>(null);
  const [fitbitData, setFitbitData] = useState<any>(null);

  useEffect(() => {
    console.log("useEffect hook is called");
    fetchUserData();
  }, []);

  async function fetchUserData() {
    try {
      const response = await fetch("http://127.0.0.1:5000/api/profile", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer " + localStorage.getItem("authToken"),
        },
      });
      if (!response.ok) throw new Error("Failed to fetch user data");
      const data = await response.json();
      setUser(data.profile);
      setFitbitData(data.profile.fitbit_data); // Set Fitbit data
    } catch (error) {
      console.error("Error fetching user data:", error);
    }
  }

  async function handlePrediction() {
    const formData = new FormData(
      document.getElementById("wellness-form") as HTMLFormElement
    );
    const data = {
      age: Number(formData.get("age")),
      steps: Number(formData.get("steps")),
      weight: Number(formData.get("weight")),
      height: Number(formData.get("height")),
      activity_level: formData.get("activity_level"),
    };
    try {
      const response = await fetch("http://127.0.0.1:5000/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer " + localStorage.getItem("authToken"),
        },
        body: JSON.stringify(data),
      });
      if (!response.ok) throw new Error("Prediction failed");
      const predictionResult = await response.json();
      setPredictionData(predictionResult);
      setUser({ ...user, ...predictionResult }); // Update user data with prediction results
    } catch (error) {
      console.error("Error predicting data:", error);
    }
  }


  return (
    <div>
      <main className="container mx-auto mt-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-lg font-bold text-gray-700">Predicted Calories</h2>
            <p id="calories" className="text-2xl font-bold text-blue-500 mt-4">
              {predictionData?.predicted_calories || user?.calories}
            </p>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-lg font-bold text-gray-700">BMI</h2>
            <p id="bmi" className="text-2xl font-bold text-green-500 mt-4">
              {predictionData?.bmi || user?.bmi}
            </p>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-lg font-bold text-gray-700">
              Daily Calorie Goal
            </h2>
            <p
              id="calorieGoal"
              className="text-2xl font-bold text-yellow-500 mt-4"
            >
              {predictionData?.calorie_goal || user?.goals}
            </p>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-lg font-bold text-gray-700">Nutrition</h2>
            <div className="mt-4">
              <p className="text-gray-600">Carbs: {predictionData?.nutrition?.carbs || user?.nutrition?.carbs}g</p>
              <p className="text-gray-600">Protein: {predictionData?.nutrition?.protein || user?.nutrition?.protein}g</p>
              <p className="text-gray-600">Fat: {predictionData?.nutrition?.fat || user?.nutrition?.fat}g</p>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-lg font-bold text-gray-700">Workout Suggestion</h2>
            <p className="text-gray-600 mt-4">
              {predictionData?.workout_suggestion || user?.workouts}
            </p>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-lg font-bold text-gray-700">Fitbit Data</h2>
            <p className="text-lg text-gray-600 mt-4">
              {fitbitData ? JSON.stringify(fitbitData) : "No Fitbit data available"}
            </p>
          </div>
        </div>


    <div className="mt-8">
      <h2 className="text-xl font-bold text-gray-700">Wellness Metrics Chart</h2>
      <div className="mt-4" style={{ width: "100%", height: "400px" }}>
        <BarChart
          xAxis={[
            {
              scaleType: 'band',
              data: [
                'Predicted Calories', 
                'BMI', 
                'Calorie Goal'
              ]
            }
          ]}
          series={[
            {
              data: [
                predictionData?.predicted_calories || user?.calories,
                predictionData?.bmi || user?.bmi,
                predictionData?.calorie_goal || user?.goals
              ],
            }
          ]}

        />
      </div>
    </div>


        <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-700">AI Wellness Coach</h2>
          <form id="wellness-form" className="mt-4">
            <div className="mb-4">
              <label htmlFor="age" className="block text-gray-600">
                Age
              </label>
              <input
                type="number"
                id="age"
                name="age"
                className="w-full border rounded px-3 py-2"
                required
              />
            </div>
            <div className="mb-4">
              <label htmlFor="steps" className="block text-gray-600">
                Steps
              </label>
              <input
                type="number"
                id="stepsInput"
                name="steps"
                className="w-full border rounded px-3 py-2"
                required
              />
            </div>
            <div className="mb-4">
              <label htmlFor="weight" className="block text-gray-600">
                Weight (kg)
              </label>
              <input
                type="number"
                id="weight"
                name="weight"
                className="w-full border rounded px-3 py-2"
                required
              />
            </div>
            <div className="mb-4">
              <label htmlFor="height" className="block text-gray-600">
                Height (cm)
              </label>
              <input
                type="number"
                id="height"
                name="height"
                className="w-full border rounded px-3 py-2"
                required
              />
            </div>
            <div className="mb-4">
              <label htmlFor="activity_level" className="block text-gray-600">
                Activity Level
              </label>
              <select
                id="activity_level"
                name="activity_level"
                className="w-full border rounded px-3 py-2"
                required
              >
                <option value="Sedentary">Sedentary</option>
                <option value="Lightly Active">Lightly Active</option>
                <option value="Active">Active</option>
                <option value="Very Active">Very Active</option>
              </select>
            </div>
            <button
              type="button"
              onClick={handlePrediction}
              className="bg-blue-600 text-white px-4 py-2 rounded"
            >
              Predict
            </button>
          </form>

          <div id="result" className="mt-6">
            {predictionData && (
              <div>
                <h3 className="text-lg font-bold text-gray-700">Prediction Results:</h3>
                <p>Predicted Calories: {predictionData.predicted_calories}</p>
                <p>BMI: {predictionData.bmi}</p>
                <p>Calorie Goal: {predictionData.calorie_goal}</p>
                <p>Nutrition:</p>
                <ul>
                  <li>Carbs: {predictionData.nutrition.carbs}g</li>
                  <li>Protein: {predictionData.nutrition.protein}g</li>
                  <li>Fat: {predictionData.nutrition.fat}g</li>
                </ul>
                <p>Workout Suggestion: {predictionData.workout_suggestion}</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Page;
