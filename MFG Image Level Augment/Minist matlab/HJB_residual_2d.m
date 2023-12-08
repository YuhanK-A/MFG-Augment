function [pd_r ] = HJB_residual_2d(phi,phi1,Q,M1,M2,N,ht,hx1,hx2)
%UNTITLED18 Summary of this function goes here
%   %calculate residual so on
%fokker planck equation

%HJE
hj = zeros(M1,M2,N);
for l = 1:(N-1)
    for i = 1:(M1)
        ix = M1 - mod(1-i,M1);
        ix0=1;
        if i ==1
            ix0=0;
        end
        for j = 1:(M2)
            jx = M2 - mod(1-j,M2);
            jx0=1;
            if j ==1
                jx0=0;
            end
            za = ix0*(phi(i,j,l) - phi(ix,j,l))/hx1;
            zb = jx0*(phi(i,j,l) - phi(i,jx,l))/hx2;
            hj(i,j,l) = min(0, (phi(i,j,l+1)-phi(i,j,l))/ht -0.5*(za^2+zb^2) + Q(i,j)); %%how about m (1) or m(N)
            
            
        end
    end
end
l=N;
for i = 1:(M1)
    ix = M1 - mod(1-i,M1);
    ix0=1;
    if i ==1
        ix0=0;
    end
    for j = 1:(M2)
        jx = M2 - mod(1-j,M2);
        jx0=1;
        if j ==1
            jx0=0;
        end
        za = ix0*(phi(i,j,l) - phi(ix,j,l))/hx1;
        zb =jx0*(phi(i,j,l) - phi(i,jx,l))/hx2;
        hj(i,j,l) = min(0, (phi1(i,j)-phi(i,j,l))/ht -0.5*(za^2+zb^2) + Q(i,j)); %%how about m (1) or m(N)
        
        
    end
end
pd_r = sum( sum(sum(hj.^2)))*hx1*hx2*ht;
end

