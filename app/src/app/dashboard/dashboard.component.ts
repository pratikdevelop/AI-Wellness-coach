import { Component, OnInit, ViewChild } from '@angular/core';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';
import { BaseChartDirective } from 'ng2-charts';
import { ChartData, ChartEvent } from 'chart.js';
import { CommonModule } from '@angular/common';
import { UserDataService } from '../user-data.service'; // Import the UserDataService
import { environment } from '../../environments/environment';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';

@Component({
  selector: 'app-dashboard',
  imports: [FormsModule, ReactiveFormsModule, BaseChartDirective, CommonModule, MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatButtonModule,
    ReactiveFormsModule,],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent implements OnInit {
  user: any = {};
  predictionData: any = null;
  fitbitData: any = null;
  wellnessForm: FormGroup;
  formData = {
    age: null,
    steps: null,
    weight: null,
    height: null,
    activity_level: ''
  };

  @ViewChild(BaseChartDirective) chart: BaseChartDirective<'bar'> | undefined;

  public barChartOptions: any = {
    scales: {
      x: {},
      y: {
        min: 10,
      },
    },
    plugins: {
      legend: {
        display: true,
      },
      datalabels: {
        anchor: 'end',
        align: 'end',
      },
    },
  };
  
  public barChartType = 'bar' as const;

  public barChartData: ChartData<'bar'> = {
    labels: ['Predicted Calories', 'BMI', 'Calorie Goal'],
    datasets: [
      { data: [65, 59, 80], label: 'Series A' },
    ],
  };

  // Inject the UserDataService to get and update user data
  constructor(private userDataService: UserDataService, private fb: FormBuilder) {
    this.wellnessForm = this.fb.group({
      age: ['', Validators.required],
      steps: ['', Validators.required],
      weight: ['', Validators.required],
      height: ['', Validators.required],
      activity_level: ['']
    });
  }

  ngOnInit() {
    this.fetchUserData();
  }

  async fetchUserData() {
    try {
      const response = await fetch(`${environment.api}/profile`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer " + localStorage.getItem("authToken"),
        }
      });
      if (!response.ok) throw new Error("Failed to fetch user data");
      const data = await response.json();
      this.user = data.profile;
      this.fitbitData = data.profile.fitbit_data;

      
      this.wellnessForm.get('height')?.setValue(this.user.height);
      this.wellnessForm.get('weight')?.setValue(this.user.weight);
      this.wellnessForm.get('activity_level')?.setValue(this.user.activity_level);
      this.wellnessForm.get('steps')?.setValue(this.user.steps);
      this.wellnessForm.get('age')?.setValue(this.user.age);
      // Update the user data in the UserDataService
      this.userDataService.updateUserData(this.user);
    } catch (error) {
      console.error("Error fetching user data:", error);
    }
  }

  async handlePrediction() {
    const data = {
      age: this.formData.age,
      steps: this.formData.steps,
      weight: this.formData.weight,
      height: this.formData.height,
      activity_level: this.formData.activity_level
    };

    try {
      const response = await fetch(`${environment.api}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer " + localStorage.getItem("authToken"),
        },
        body: JSON.stringify(data),
      });
      if (!response.ok) throw new Error("Prediction failed");
      const predictionResult = await response.json();
      this.predictionData = predictionResult;
      this.user = { ...this.user, ...predictionResult };

      // Update the user data with prediction results in the UserDataService
      this.userDataService.updateUserData(this.user);
    } catch (error) {
      console.error("Error predicting data:", error);
    }
  }
}
