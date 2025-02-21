/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable react-hooks/rules-of-hooks */
"use client";
import { Chart } from "chart.js";
import React, { useEffect, useState } from "react";

const page = () => {
  const [user, setUser] = useState<any>({});
  const [predictionData, setPredictionData] = useState<any>(null);

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
      updateChart(data.profile);
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
      setPredictionData(predictionResult)
      setUser({ ...user, predictionResult });
      
      updateChart(predictionResult)
    } catch (error) {
      console.error("Error predicting data:", error);
    }
  }

  function updateChart(data: { steps_today: any; calories: any; bmi: any }) {
    const ctx = document.getElementById("activityChart") as HTMLCanvasElement;

    if (ctx) {
      new Chart(ctx, {
        type: "bar",
        data: {
          labels: ["Steps", "Calories Burned", "BMI"],
          datasets: [
            {
              label: "Activity Summary",
              data: [data.steps_today || 0, data.calories || 0, data.bmi || 0],
              backgroundColor: [
                "rgba(75, 192, 192, 0.6)",
                "rgba(255, 206, 86, 0.6)",
                "rgba(153, 102, 255, 0.6)",
              ],
              borderColor: [
                "rgba(75, 192, 192, 1)",
                "rgba(255, 206, 86, 1)",
                "rgba(153, 102, 255, 1)",
              ],
              borderWidth: 1,
            },
          ],
        },
        options: {
          responsive: true,
          scales: { y: { beginAtZero: true } },
        },
      });
    }
  }

  return (
    <div>
      <main className="container mx-auto mt-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-lg font-bold text-gray-700">Calories Burned</h2>
            <p id="calories" className="text-2xl font-bold text-blue-500 mt-4">
              {user?.calories}
            </p>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-lg font-bold text-gray-700">BMI</h2>
            <p id="bmi" className="text-2xl font-bold text-green-500 mt-4">
              {user?.bmi}
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
              {user?.calorie_goal}
            </p>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-lg font-bold text-gray-700">Steps Today</h2>
            <p id="steps" className="text-2xl font-bold text-purple-500 mt-4">
              {user?.steps_today}
            </p>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-lg font-bold text-gray-700">Activity Level</h2>
            <p
              id="activityLevel"
              className="text-2xl font-bold text-red-500 mt-4"
            >
              {user?.activity_level}
            </p>
          </div>
        </div>

        <div className="mt-8">
          <h2 className="text-xl font-bold text-gray-700">Activity Chart</h2>
          <canvas id="activityChart" className="w-full h-64 mt-4"></canvas>
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
                <h3>Prediction Results:</h3>
                <p>Calories: {predictionData.calories}</p>
                <p>BMI: {predictionData.bmi}</p>
                <p>Calorie Goal: {predictionData.calorie_goal}</p>
                <p>Steps Today: {predictionData.steps_today}</p>
                <p>Activity Level: {predictionData.activity_level}</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default page;
