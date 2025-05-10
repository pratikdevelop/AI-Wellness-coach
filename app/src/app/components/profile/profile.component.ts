import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import {
  MatProgressBarModule
} from '@angular/material/progress-bar'
import { MatIconModule } from '@angular/material/icon';
import { UserDataService } from '../../user-data.service';
@Component({
  selector: 'app-profile',
  imports:[CommonModule, MatProgressBarModule, MatIconModule],

  templateUrl: './profile.component.html',
  styleUrls: ['./profile.component.css']
})
export class ProfileComponent implements OnInit {
  user: any;
  badges: any;
  goalKeys: string[] = [];

  constructor(private http: HttpClient, private userService: UserDataService) {}

  ngOnInit() {
    this.loadProfile();
    this.loadBadges();
  }

  loadProfile() {
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${localStorage.getItem('authToken')}` });
    this.http.get('http://localhost:5000/api/profile', { headers }).subscribe(
      (response: any) => {
        this.user = response;
        this.goalKeys = Object.keys(this.user.progress || {});
        this.userService.updateUserData(this.user)

      },
      error => console.error('Error fetching profile:', error)
    );
  }

  loadBadges() {
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${localStorage.getItem('authToken')}` });
    this.http.get('http://localhost:5000/api/badges', { headers }).subscribe(
      (response: any) => this.badges = response,
      error => console.error('Error fetching badges:', error)
    );
  }

  calculateProgress(progress: any): number {
    if (!progress || !progress.goal || !progress.current) return 0;
    return Math.min((progress.current / progress.goal) * 100, 100);
  }

  editProfile() {
    // Placeholder for edit functionality (e.g., navigate to an edit form)
    console.log('Edit profile clicked');
  }
}