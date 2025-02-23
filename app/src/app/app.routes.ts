import { Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { SignupComponent } from './signup/signup.component';
import { LoginComponent } from './login/login.component';

export const routes: Routes = [
    {
        path: '',
        component: HomeComponent,
    },
    {
        path: 'dashboard',
        component: DashboardComponent
    },
    {
        path: 'signup',
        component: SignupComponent
    },
{
    path: 'login',
    component: LoginComponent
}
];
