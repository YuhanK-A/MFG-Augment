function [F_phi_updates] = solvePoisson3d_T_0_eps(M1,M2,N,rho_m_vec,ht,coeff_x,coeff_t,fv)
%UNTITLED Summary of this function goes here
%  3D:t,x1,x2
F_phi_updates = zeros(M1,M2,N);
F_rho_m_vec = zeros(M1,M2,N);
% do fourier transform
for l =1:(N) 
     F_rho_m_vec(:,:,l) = mirt_dctn(rho_m_vec(:,:,l));
    %F_rho_m_vec(:,:,l) = fft2(rho_m_vec(:,:,l));
%    F_rho_m_vec(:,:,l) = fft2(squeeze(rho_m_vec(:,:,l)));
end
% %for each fourier mode, solve the (c1-(cx 2pi xi).^2 )* I + ct nalba_t =
% %rhs
phi_fouir_part = zeros(M1,M2,N);
negative_onesa = -(1*coeff_t/ht/ht)*ones(N-1,1);
negative_onesc =negative_onesa ;

for i = 1: (M1) %j is the corresponding fourier mode
    for j = 1:M2
            f =  squeeze(F_rho_m_vec(i,j,:));
            cc =  coeff_x*(fv(i,j)) + 2*coeff_t/ht/ht;
            thomas_b = cc*ones(N,1);  %%??
            thomas_b(N) =  thomas_b(N)-1*coeff_t/ht/ht ;
            thomas_b(1) =  thomas_b(1)-1*coeff_t/ht/ht+1/ht; %add eps,
            thomas_n = N;
            s = ThomasAlgorithm(negative_onesa,thomas_b,negative_onesc,f,thomas_n);
            phi_fouir_part(i,j,:) = s;
    end
end


for l =1:(N) 
    F_phi_updates(:,:,l) = mirt_idctn(phi_fouir_part(:,:,l));
end

end

