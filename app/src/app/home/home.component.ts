import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { RouterModule } from '@angular/router';

@Component({
  selector: 'app-home',
  imports: [RouterModule, CommonModule],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css'
})
export class HomeComponent {
  features = [
    { title: 'AI-Powered Insights', description: 'Get actionable recommendations based on your unique health data.' },
    { title: 'Track Progress', description: 'Monitor your BMI, activity levels, and calorie goals over time.' },
    { title: 'Personalized Goals', description: 'Set and achieve realistic fitness goals with AI guidance.' }
  ];

  testimonials = [
    { quote: 'This app changed my life! I lost weight and built healthier habits.', name: 'Emily R.' },
    { quote: 'Tracking my progress has never been this easy and intuitive.', name: 'James W.' },
    { quote: 'The personalized AI recommendations are spot on.', name: 'Sarah L.' }
  ];
}
