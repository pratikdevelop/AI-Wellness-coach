import { CommonModule } from '@angular/common';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-insights',
  imports: [CommonModule, MatButtonModule],
  templateUrl: './insights.component.html',
  styleUrl: './insights.component.css'
})
export class InsightsComponent {
  insights: any;

  constructor(private http: HttpClient) {}

  fetchInsights() {
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${localStorage.getItem('authToken')}` });
    this.http.get(`${environment.api}/mental_health_insights`, { headers }).subscribe(
      (response: any) => this.insights = response,
      error => console.error('Error fetching insights:', error)
    );
  }
}
