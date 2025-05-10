import { trigger, transition, style, animate, query, stagger } from '@angular/animations';
import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { RouterModule } from '@angular/router';

@Component({
  selector: 'app-home',
  imports: [RouterModule, CommonModule],
  animations: [
    // Hero fade-in
    trigger('fadeIn', [
      transition(':enter', [
        style({ opacity: 0 }),
        animate('1s ease-in', style({ opacity: 1 }))
      ])
    ]),
    // Staggered features
    trigger('staggerFadeIn', [
      transition('* => *', [
        query(':enter', [
          style({ opacity: 0, transform: 'translateY(20px)' }),
          stagger(100, [
            animate('0.5s ease-out', style({ opacity: 1, transform: 'none' }))
          ])
        ], { optional: true })
      ])
    ]),
    // Stats slide-in
    trigger('slideInRight', [
      transition(':enter', [
        style({ transform: 'translateX(100%)', opacity: 0 }),
        animate('0.8s ease-out', style({ transform: 'none', opacity: 1 }))
      ])
    ]),
    // CTA pulse
    trigger('pulse', [
      transition(':enter', [
        style({ transform: 'scale(0.95)' }),
        animate('1.5s ease-in-out', style({ transform: 'scale(1)' }))
      ])
    ])
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css'
})
export class HomeComponent {
  features = [
    { icon: '📊', title: 'AI Health Tracking', description: 'Real-time insights with Fitbit sync.' },
    { icon: '🧠', title: 'Mental Wellness', description: 'Track mood and stress levels.' },
    { icon: '💪', title: 'Smart Workouts', description: 'Personalized exercise plans.' }
  ];

  stats = [
    { value: '10K', label: 'Active Users' },
    { value: '95%', label: 'Satisfaction' },
    { value: '50K', label: 'Calories Tracked' },
    { value: '1M', label: 'Steps Logged' }
  ];

  testimonials = [
    { quote: 'This app changed my life! Lost 20lbs in 3 months.', name: 'Jane D.' },
    { quote: 'The mood tracker helped me manage stress better.', name: 'Alex M.' }
  ];
}
