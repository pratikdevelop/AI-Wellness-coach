import { CommonModule } from '@angular/common';
import { Component, inject, signal } from '@angular/core';
import { ReactiveFormsModule, FormGroup, FormControl, Validators, FormsModule, ValidatorFn, AbstractControl, ValidationErrors, FormBuilder } from '@angular/forms';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { HttpClient } from '@angular/common/http';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { environment } from '../../environments/environment';

@Component({
  selector: 'app-signup',
  standalone: true,
  imports: [
    ReactiveFormsModule,
    CommonModule,
    FormsModule,
    MatSnackBarModule,
    MatFormFieldModule,
    MatInputModule,
    MatIconModule,
    MatButtonModule
  ],
  templateUrl: './signup.component.html',
  styleUrl: './signup.component.css'
})
export class SignupComponent {
  private snackBar = inject(MatSnackBar);
  private fb = inject(FormBuilder);
  private http = inject(HttpClient);
  // Validators
  passwordMatchValidator: ValidatorFn = (control: AbstractControl): ValidationErrors | null => {
    const password = control.get('password')?.value;
    const confirmPassword = control.get('confirmPassword')?.value;
    return password && confirmPassword && password !== confirmPassword
      ? { passwordsMismatch: true }
      : null;
  };
  
  // Form Controls
  signupForm = this.fb.group({
    name: ['', [Validators.required, Validators.minLength(3)]],
    username: ['', [Validators.required, Validators.minLength(3)]],
    phone: ['', [Validators.required, Validators.pattern(/^[0-9]{10,15}$/)]],
    email: ['', [Validators.required, Validators.email]],
    password: ['', [Validators.required, Validators.minLength(6)]],
    confirmPassword: ['', [Validators.required]]
  }, { 
    validators: this.passwordMatchValidator 
  });

  // Profile Picture Handling
  profilePicFile: File | null = null;
  profilePicPreview: string | ArrayBuffer | null = null;
  profilePicError: string | null = null;

  // Password Visibility
  passwordVisible = signal<boolean>(false);
  confirmPasswordVisible = signal<boolean>(false);



  // Toggle password visibility
  togglePasswordVisibility(type: string) {
    if (type === 'password') {
      this.passwordVisible.update(prev => !prev);
    } else {
      this.confirmPasswordVisible.update(prev => !prev);
    }
  }

  // Handle file selection
  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      const file = input.files[0];
      
      // Validate file
      if (file.size > 2 * 1024 * 1024) { // 2MB limit
        this.profilePicError = 'File size must be less than 2MB';
        return;
      }
      
      if (!['image/jpeg', 'image/png'].includes(file.type)) {
        this.profilePicError = 'Only JPEG/PNG files are allowed';
        return;
      }
      
      this.profilePicError = null;
      this.profilePicFile = file;
      
      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        this.profilePicPreview = reader.result;
      };
      reader.readAsDataURL(file);
    }
  }

  // Handle form submission
  handleSubmit(): void {
    if (this.signupForm.invalid || this.profilePicError) return;

    const formData = new FormData();
    
    // Append all form values
    Object.keys(this.signupForm.value).forEach(key => {
      const value = this.signupForm.get(key)?.value;
      if (value !== null && value !== undefined) {
        formData.append(key, value);
      }
    });
    
    // Append profile pic if exists
    if (this.profilePicFile) {
      formData.append('profilePic', this.profilePicFile);
    }
    
    // Send to backend using HttpClient (better than axios for Angular)
    this.http.post(`${environment.api}/auth/signup`, formData).subscribe({
      next: (response) => {
        console.log(response);
        this.snackBar.open('Signup Successful', 'Close', { duration: 2000 });
      },
      error: (error) => {
        console.error(error);
        this.snackBar.open('Signup Failed', 'Close', { duration: 2000 });
      }
    });
  }
}