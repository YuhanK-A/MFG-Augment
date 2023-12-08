function [pd_r ] = FokkerPlanck_residual_ot_2D( rho,rho0,rho1,m,M1,M2,N,ht,hx1,hx2)
%UNTITLED18 Summary of this function goes here
%   %calculate residual so on
%fokker planck equation
rho_m_vec = zeros(M1,M2,N-1);

% for l = 2:(N-1)
%     for i = 1:M1
%         for j =1:M2
%             ix = mod(i,M1)+1;
%             jx =mod(j,M2)+1;
%             rho_m_vec(i,j,l-1) = ( (rho(i,j,l)-rho(i,j,l-1))/ht + (m(ix,j,1,l)-m(i,j,1,l))/hx1+ (m(i,jx,2,l)-m(i,j,2,l))/hx2 );
%             
%         end
%     end
% end
l=1;
    for i = 1:(M1)
        ix = mod(i,M1)+1;
        ix0=1;
        if i ==M1
            ix0=0;
        end
        for j = 1:(M2)
            jx = mod(j,M2)+1;
            jx0=1;
            if j ==M2
                jx0=0;
            end
            rho_m_vec(i,j,l) = ( (rho(i,j,l)-rho0(i,j))/ht + (ix0*m(ix,j,1,l)-m(i,j,1,l))/hx1+ (jx0*m(i,jx,2,l)-m(i,j,2,l))/hx2 ); %%how about m (1) or m(N)
        end
    end
    for l=2:(N-1)
        for i = 1:(M1)
            ix = mod(i,M1)+1;
            ix0=1;
            if i ==M1
                ix0=0;
            end
            for j = 1:(M2)
                jx = mod(j,M2)+1;
                jx0=1;
                if j ==M2
                    jx0=0;
                end
                rho_m_vec(i,j,l) = ( (rho(i,j,l)-rho(i,j,l-1))/ht + (ix0*m(ix,j,1,l)-m(i,j,1,l))/hx1+ (jx0*m(i,jx,2,l)-m(i,j,2,l))/hx2 ); %%how about m (1) or m(N)
            end
        end
    end
    
    l=N;
    for i = 1:(M1)
        ix = mod(i,M1)+1;
        ix0=1;
        if i ==M1
            ix0=0;
        end
        for j = 1:(M2)
            jx = mod(j,M2)+1;
            jx0=1;
            if j ==M2
                jx0=0;
            end
            rho_m_vec(i,j,l) = ( (rho1(i,j)-rho(i,j,l-1))/ht + (ix0*m(ix,j,1,l)-m(i,j,1,l))/hx1+ (jx0*m(i,jx,2,l)-m(i,j,2,l))/hx2 ); %%how about m (1) or m(N)
        end
    end
pd_r =sum( sum(sum(rho_m_vec.^2)))*hx1*hx2*ht;
end