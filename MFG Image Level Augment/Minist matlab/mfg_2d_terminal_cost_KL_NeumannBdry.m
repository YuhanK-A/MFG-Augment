function rho = mfg_2d_terminal_cost_KL_NeumannBdry(x)
c_terminal =1;

M1 =28%space discretization [0,1.0]
M2 = 28
N=28 %time discretization
ht=1.0/(N); %time step
hx1 = 1.0/M1;
hx2= 1/M2;
m = zeros(M1,M2,2,N); %

rho1 = zeros(M1,M2);
rho0 = rho1;
rho_target =rho1;
Q =rho1;

sigma =1/16;
sigmaq = 0.2;

rho0 = x(:,:,1) + 1e-5;
rho_target = x(:,:,2) + 1e-5;

% for k = 1:4
%     xx1 = cos((k-1)*pi/2)*0.25+0.5;
%     xx2 = sin((k-1)*pi/2)*0.25+0.5;
%     for i = 1:M1
%         x1 =(i-1)*hx1;
%         for j = 1:M2
%             x2 =(j-1)*hx2 ;
%             rho_target(i,j) =  rho_target(i,j) + 1/4/sigma/sqrt(2*pi)*1/sigma/sqrt(2*pi)*exp(-0.5*(((x1-xx1)^2+(x2-xx2)^2)/(sigma^2)));
%         end
%     end
%     
% end


rho_target= 1*rho_target/(sum(sum(rho_target)))/hx1/hx2;
rho0 = 1*rho0/(sum(sum(rho0)))/hx1/hx2;
rho1 = rho_target;
beta = log(rho1)+1;

rho = repmat(rho0,[1,1,N]);
phi = 0.0*ones(M1,M2,N);%zeros(N,M);
phi1 = c_terminal*ones(M1,M2);


%parameters for PDHG
max_itr =2000;
%pdhg parameters
tau1 = 9.9; %rho
tau2 = 9.9;  %m
tau3 = 0.1; %phi
tau5 = 0.1; %beta


coeff_x = 1;%tau_phi_t/tau_phi_x;%1e-1;
coeff_t = 1.0;%1e-1;

%fourier coefficient for Spatial Laplacian
fLapalacian = zeros(M1,M2);
for i = 1:(M1)
    for j = 1:M2
        fLapalacian(i,j) =1/hx1/hx1*(2*sin(pi*(i-1)/2/(M1-0)))^2 +1/hx2/hx2*(2*sin(pi*(j-1)/2/(M2-0)))^2;
    end
end


%record_pd = zeros(max_itr,3);
record_pdr = zeros(max_itr,1);
record_pdr_hjb = zeros(max_itr,1);
record_cost = zeros(max_itr,5);

myfun = @(x,c_rho,c_alpha,c_tau) c_tau*exp(x-1) + x-c_alpha - c_tau*c_rho;  % parameterized function

tic

for itr = 1:max_itr
    if mod(itr,100) ==0
        itr
        sum(sum(rho(:,:,1)))
        sum(sum(rho(:,:,N)))
        %                 max(max(max(rho)))
        %                 max(max(max(phi)))
        
        %         record_pdr_hjb(itr-1,:)
        %         record_cost(itr-1,:)
        record_pdr(itr) =FokkerPlanck_residual_ot_2D( rho,rho0,rho1,m,M1,M2,N,ht,hx1,hx2);
        record_pdr(itr,:)
        record_pdr_hjb(itr) = HJB_residual_2d(phi,phi1,Q,M1,M2,N,ht,hx1,hx2);
        record_pdr_hjb(itr)
        record_cost(itr,:) = MFG_cost_2d_terminal_KL( rho,m,phi,phi1,rho0,rho1,rho_target,M1,M2,N,c_terminal,Q,hx1,hx2,ht);
        record_cost(itr,:)
    end
    %%descent step
    
    
    %     record_cost(itr,:) = mfg_cost_2D( rho,m,phi,phi1,rho0,rho1,rho_target,M1,M2,N,c_kl,c_terminal,Q);
    
    %pd_residual_ot_2d_gamma( rho,m,phi,phi0,M1,M2,N,ht,hx1,hx2);
    %      record_pd(itr,:) = pd_value_ot_2d_gamma( rho,m,phi,phi0,rho_0,rho_1,M1,M2,N);
    phi_bar =phi;
    beta_bar = beta;
    phi1_bar = phi1;
    
    %update phi
    %  (H1)
    rho_m_vec = zeros(M1,M2,N);
    
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
    
    phiu = solvePoisson3d_T_0_eps(M1,M2,N,-rho_m_vec,ht,coeff_x,coeff_t,fLapalacian);
    phi = phi+tau3* phiu;
    u1 = phiu(:,:,N)-(rho1-rho(:,:,N))*ht;
    phi1 = phi1 + tau3 * u1;
    
    %update beta
    
    for i = 1:M1
        for j = 1:M2
            fun = @(x) myfun(x,rho(i,j,N),beta(i,j),tau5*c_terminal);    % function of x alone (alpha)
            beta(i,j) = fzero(fun,beta(i,j));
        end
    end
    
    phi_bar = 2*phi - phi_bar;
    phi1_bar = 2*phi1 - phi1_bar;
    beta_bar = 2*beta - beta_bar;
    
    %ascent step
    %update m rho simutaneouslyfor l = 2:(N-1)
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
                
                za = m(i,j,1,l) -ix0*(phi_bar(i,j,l) - phi_bar(ix,j,l))/hx1*tau2;
                zb = m(i,j,2,l) -jx0*(phi_bar(i,j,l) - phi_bar(i,jx,l))/hx2*tau2;
                z = za^2 + zb^2;
                
                alphaxt = rho(i,j,l)+tau1*(-Q(i,j) -1/ht*(phi_bar(i,j,l+1)-phi_bar(i,j,l)));
                d = tau2/2 * (z);
                b = alphaxt + tau1;
                cc=cubic_poly_solve(-b,0,-d);
                rho(i,j,l) = max(cc-tau1,0);
                m(i,j,1,l) = rho(i,j,l)/(rho(i,j,l) + tau2) * za;
                m(i,j,2,l) = rho(i,j,l) /(rho(i,j,l) + tau2) * zb;
                
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
            
            za = m(i,j,1,l) -ix0*(phi_bar(i,j,l) - phi_bar(ix,j,l))/hx1*tau2;
            zb = m(i,j,2,l) -jx0*(phi_bar(i,j,l) - phi_bar(i,jx,l))/hx2*tau2;
            z = za^2 + zb^2;
            
            alphaxt = rho(i,j,l)+tau1*(-Q(i,j) -1/ht*(phi1_bar(i,j)-phi_bar(i,j,l)));
            %alphaxt = rho(i,j,l)+tau1*(-Q(i,j) - c_kl*alpha_bar(i,j,l) -1/ht*(phi1_bar(i,j)-phi_bar(i,j,l)));
            d = tau2/2 * (z);
            b = alphaxt + tau1;
            cc=cubic_poly_solve(-b,0,-d);
            rho(i,j,l) = max(cc-tau1,0);
            m(i,j,1,l) = rho(i,j,l)/(rho(i,j,l) + tau2) * za;
            m(i,j,2,l) = rho(i,j,l) /(rho(i,j,l) + tau2) * zb;
            
        end
    end
    %for rho1
    %quadratic programming on a simplex
    y = rho1+0.01*(-c_terminal*beta_bar+c_terminal*log(rho_target)+phi1_bar);
    rho1= optimization_on_simplex_matrix(y,M1*M2);
    
end
toc
end