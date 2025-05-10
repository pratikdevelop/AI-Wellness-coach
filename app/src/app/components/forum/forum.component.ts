import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-forum',
  imports:[CommonModule],  
  template: '<div *ngFor="let post of posts"><h3>{{post.content}}</h3><button (click)="addComment(post._id)">Add Comment</button></div>'
})
export class ForumComponent implements OnInit {
  posts: any[] = [];

  constructor(private http: HttpClient) {}

  ngOnInit() {
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${localStorage.getItem('token')}` });
    this.http.get('/api/forum/posts', { headers }).subscribe(
      (response: any) => this.posts = response,
      error => console.error('Error fetching posts:', error)
    );
  }

  addComment(postId: string) {
    // Implement comment form and POST to /api/forum/posts/{postId}/comments
  }
}