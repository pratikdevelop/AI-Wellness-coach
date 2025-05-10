import { CommonModule } from '@angular/common';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component } from '@angular/core';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIcon } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-mood',
  imports: [ MatFormFieldModule, ReactiveFormsModule, FormsModule, MatSelectModule, MatButtonModule, CommonModule, MatInputModule],
  templateUrl: './mood.component.html',
  styleUrl: './mood.component.css'
})
export class MoodComponent {
  moodForm: FormGroup;

  constructor(private fb: FormBuilder, private http: HttpClient) {
    this.moodForm = this.fb.group({
      mood: ['', Validators.required],
      stress_level: ['', [Validators.required, Validators.min(1), Validators.max(10)]],
      notes: ['']
    });
  }

  logMood() {
    if (this.moodForm.valid) {
      const userId = localStorage.getItem('user_id');
      const headers = new HttpHeaders({ 'Authorization': `Bearer ${localStorage.getItem('authToken')}` });
      this.http.post(`${environment.api}/log_mood`, { ...this.moodForm.value}, { headers }).subscribe(
        response => {
          console.log('Mood logged:', response);
          this.moodForm.reset();
        },
        error => console.error('Error logging mood:', error)
      );
    }
  }
}
