

import { Component, inject } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import axios from 'axios';
import { ReactiveFormsModule } from '@angular/forms';
import { MatInputModule } from '@angular/material/input';
import{ MatFormFieldModule } from '@angular/material/form-field';
import {   MatIconModule } from '@angular/material/icon';
import {  MatButtonModule } from '@angular/material/button';
import { CommonModule } from '@angular/common';
import { Router, RouterModule } from '@angular/router';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { environment } from '../../environments/environment';

@Component({
  selector: 'app-login',
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatInputModule,
    MatFormFieldModule,
    MatButtonModule,
    MatIconModule,
    RouterModule,
    MatSnackBarModule
  ],

  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent {
  loginForm: FormGroup;
  hidePassword = true;
  router = inject(Router);
  snackBar = inject(MatSnackBar)

  constructor(private fb: FormBuilder) {
    this.loginForm = this.fb.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(6)]]
    });
  }

  onSubmit(): void {
    if (this.loginForm.invalid) {
      return;
    }

    const { email, password } = this.loginForm.value;

    axios
      .post(`${environment.api}/login`, { email, password })
      .then(response => {
        const result = response.data;
        if (result.token) {
          localStorage.setItem('authToken', result.token);
          this.snackBar.open('Login successful! Redirecting...', 'close', {
            duration: 2000,

          });
          this.router.navigateByUrl('/dashboard');
        }
      })
      .catch(error => {
        console.error('An error occurred:', error.message);
          this.snackBar.open(`Error in login, ${error.message}` , 'close', {
            duration: 2000,
          })
      });
  }
}

