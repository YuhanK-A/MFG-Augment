function [ solution] = cubic_poly_solve(b,c,d )
%UNTITLED13 Summary of this function goes here
%root_vec = roots([1, cubic_coff1, 0,cubic_coff2]);
%       cc = max(root_vec(imag(root_vec) == 0));
% cc=cubic_poly_solve(cubic_coff1,0,cubic_coff2); 
b3over3=(b/3)*(b/3)*(b/3);
    
p=c-b*(b/3);
q=d+2*b3over3-b*(c/3);
solution=0;
real3rdRoot1=-.5;   %equals cos(2*M_PI/3);
im3rdRoot1=0.86602540378;   %equals sin(2*M_PI/3);
real3rdRoot2=-.5;  %equals cos(4*M_PI/3)=real3rdRoot1;
im3rdRoot2=-0.86602540378; %equals sin(4*M_PI/3)=-im3rdRoot1;  
if p==0
        
        solution=-sign(q)*exp(log(abs(q))/3.0);
        
else
       discrim=(q/2)*(q/2)+(p/3)*(p/3)*(p/3);
        
        s=sqrt(abs(discrim));
        
        if discrim<0
            
            theta=atan2(s,-q/2);
            
           x=s*s+q*q/4;
           rc=exp(log(x)/6);
            
          thetac=theta/3;
            
           real=rc*cos(thetac);
           im=rc*sin(thetac);
            
            solution1=2*real;
            
            
          solution2=2*(real*real3rdRoot1-im*im3rdRoot1);
          solution3=2*(real*real3rdRoot2-im*im3rdRoot2);
            
            solution=max(solution1,max(solution2,solution3));
            
            
        elseif discrim>0
            
            u3=-q/2+s;
            v3=-q/2-s;
            
            u=sign(u3)*exp(log(abs(u3))/3);
            v=sign(v3)*exp(log(abs(v3))/3);
            
            solution=u+v;
            
        else
            solution=max(3*q/p, -3*q/(2*p));
            
        end
end
    
    solution = solution-b/3;
    
end


