import { Component } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-meal-plan',
  imports: [CommonModule],
  templateUrl: './meal-plan.component.html',
  styleUrl: './meal-plan.component.css'
})
export class MealPlanComponent {
  mealPlan: any;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${localStorage.getItem('authToken')}` });
    this.http.get('/api/meal_plan', { headers }).subscribe(
      (response: any) => this.mealPlan = response.meal_plan,
      error => console.error('Error fetching meal plan:', error)
    );
  }
}

