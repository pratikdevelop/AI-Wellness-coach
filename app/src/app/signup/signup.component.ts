import { CommonModule } from '@angular/common';
import { Component, inject, signal } from '@angular/core';
import {ReactiveFormsModule, FormGroup, FormControl, Validators, FormsModule, ValidatorFn, AbstractControl, ValidationErrors} from '@angular/forms';
import {MatSnackBar, MatSnackBarModule} from '@angular/material/snack-bar'
import axios from 'axios';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon'
import { environment } from '../../environments/environment';


@Component({
  selector: 'app-signup',
  imports: [ReactiveFormsModule, CommonModule, FormsModule, MatSnackBarModule,  MatFormFieldModule,
    MatInputModule,
    MatIconModule,
    MatButtonModule,],
  templateUrl: './signup.component.html',
  styleUrl: './signup.component.css'
})
export class SignupComponent {
  snackBar = inject(MatSnackBar);
  signupForm: FormGroup = new FormGroup({
    username: new FormControl('', [Validators.required, Validators.minLength(3)]),
    email: new FormControl('', [Validators.required, Validators.email]),
    password: new FormControl('', [Validators.required, Validators.minLength(6)]),
    confirmPassword: new FormControl('', [
      Validators.required,
      Validators.minLength(6),
      this.passwordMatchValidator(),
    ]),
  });
  passwordVisible =  signal<boolean>(false);
  confirmPasswordVisible =signal<boolean>(false)

  // Toggle password visibility
  togglePasswordVisibility(type: string) {
    if (type === 'password') {
      this.passwordVisible.update((prev) => !prev);
    } else {
      this.confirmPasswordVisible.update((prev) => !prev);
    }
      
  }
  
   passwordMatchValidator(): ValidatorFn {
    return (control: AbstractControl): ValidationErrors | null => {
      const password = control.parent?.get('password')?.value;
      const confirmPassword = control.value;
      return password && confirmPassword && password !== confirmPassword
        ? { passwordsMismatch: true }
        : null;
    };
  }

  handleSubmit(): void {
    if (this.signupForm.invalid) return ;
    axios.post(`${environment.api}/signup`, {
        headers: {
          "Content-Type": "application/json",
        },
        body: this.signupForm.value,
      }).then((response) =>{
        console.log(response.data);
        this.snackBar.open('Signup Successfull', 'Close', {
          duration: 2000,
          });
          }).catch((error) => {
            console.log(error);
            this.snackBar.open('Signup Failed', 'Close', {
              duration: 2000,
              });
      })

}
    
   
}

