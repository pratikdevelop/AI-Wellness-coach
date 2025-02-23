import { Component, inject } from '@angular/core';
import { Router, RouterModule, RouterOutlet } from '@angular/router';
import { UserDataService } from './user-data.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-root',
  standalone:true,
  imports: [RouterOutlet, RouterModule, CommonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'app';
  user: any = {};
  isLoggedIn= localStorage.getItem('authToken')  !== null
  router = inject(Router)

  constructor(private userDataService: UserDataService) {}

  ngOnInit() {
    console.log(
      this.isLoggedIn
    );
    
    // Subscribe to the user data observable to get updates
    this.userDataService.userData$.subscribe((userData) => {
      this.user = userData; // Update the user data whenever it changes
      console.log(
        'User data updated:',
        this.user
      );
      
    });
  }
  logout() {
    // Clear user data from local storage or session storage
    localStorage.removeItem('authToken');
    
    // Optionally, reset user data in the service
    this.userDataService.updateUserData(null);
    
    // Redirect to login or home page after logout
    this.router.navigateByUrl('/login'); // Or use Router for routing
  }
}
