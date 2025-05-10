import { Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { SignupComponent } from './signup/signup.component';
import { LoginComponent } from './login/login.component';
import { InsightsComponent } from './components/insights/insights.component';
import { MoodComponent } from './components/mood/mood.component';
import { ProfileComponent } from './components/profile/profile.component';
import { RelaxationComponent } from './components/relaxation/relaxation.component';

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

    { path: 'profile', component: ProfileComponent },
  { path: 'mood', component: MoodComponent },
  { path: 'insights', component: InsightsComponent },
  { path: 'relaxation', component: RelaxationComponent },
  { path: '', redirectTo: '/login', pathMatch: 'full' },
{
    path: 'login',
    component: LoginComponent
}
];
