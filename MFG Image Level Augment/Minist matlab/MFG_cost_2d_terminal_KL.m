function [pd_v ] = MFG_cost_2d_terminal_KL( rho,m,phi,phi1,rho0,rho1,rho_target,M1,M2,N,c_terminal,Q,hx1,hx2,ht)
%UNTITLED18 Summary of this function goes here
%   %calculate residual so on
%fokker planck equation
pd_v = [0,0,0,0,0];
rho_m_vec = zeros(M1,M2,N);
cq=0;
for l = 1:(N)
    for i = 1:M1
        for j = 1:M2
            if rho(i,j,l)>1e-15
                rho_m_vec(i,j,l) = (m(i,j,1,l).^2 + m(i,j,2,l).^2)/2/ rho(i,j,l);
            end
             cq = cq + rho(i,j,l)*Q(i,j);
        end
    end
end

pd_v(1) = sum(sum(sum(rho_m_vec)))*hx1*hx2*ht;
pd_v(2)  = cq*hx1*hx2*ht;

ll=0;
mm = 0;
for i = 1:M1
    for j = 1:M2
    ll = ll+ rho0(i,j)*phi(i,j,1) ;
    if rho1(i,j)>1e-20
    mm = mm+ rho1(i,j)*(log(rho1(i,j))-log(rho_target(i,j)));
    end
    end
end
pd_v(3) = c_terminal*mm*hx1*hx2;
pd_v(4) = ll*hx1*hx2;

% %rho(log(rho) + Q)
% zz=0;
% jj =0;
% for l = 1:(N)
%     for i = 1:M1
%         for j = 1:M2
%             if rho(i,j,l)>1e-10
%                 zz = zz + rho(i,j,l)*log(rho(i,j,l));
%                 jj = jj + rho(i,j,l)*Q(i,j);
%             end
%         end
%     end
% end
% pd_v(3) = c_kl*hx1*hx2*ht;
% pd_v(4) = jj*hx1*hx2*ht;

pd_v(5) = pd_v(1) + pd_v(2) + pd_v(3) - pd_v(4);
end

