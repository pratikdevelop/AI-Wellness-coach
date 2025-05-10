import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RelaxationComponent } from './relaxation.component';

describe('RelaxationComponent', () => {
  let component: RelaxationComponent;
  let fixture: ComponentFixture<RelaxationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RelaxationComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(RelaxationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
