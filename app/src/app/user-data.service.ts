import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class UserDataService {
  // Create a BehaviorSubject with an initial value of an empty object
  private userDataSubject = new BehaviorSubject<any>(null);
  
  // Observable to allow other components to subscribe
  userData$ = this.userDataSubject.asObservable();

  constructor() {}

  // Method to update the user data
  updateUserData(data: any) {
    this.userDataSubject.next(data);
  }

  // Method to get the current user data
  getUserData() {
    return this.userDataSubject.value;
  }
}
