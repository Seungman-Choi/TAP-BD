%% TABD_Simulation_HighPhotonsCase

load(strcat(pwd, '\DataSet\USAF_simData.mat'))
GPU_usage = true;
if GPU_usage == true
    Const_pupil     = gpuArray(Const_pupil);
    GT_image        = gpuArray(GT_image);
    GT_PSF          = gpuArray(GT_PSF);
    GT_turbulence   = gpuArray(GT_turbulence);
    Iarr            = gpuArray(Iarr);
    psitpsi_freq    = gpuArray(psitpsi_freq);
    SLMarr          = gpuArray(SLMarr);
end
%% TABD (Simplified version; W/O. Poisson denoiser and FFT wave propagation simulation instead of angular spectrum method)
SLMnum_arr  = 10;%[10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

%%% Parameter Setting %%
Var_Maxiter     = 1000;
r1              = 1e-2; 
tau1            = 1e-5*r1;
ram1            = zeros(Num_img, Num_img, 2);
DenormH         = 2e-4;
DenormO         = 1e-3;
tau2            = 5e-5;

upd_Orecon = []; upd_turbrecon =[];
for q = 1:length(SLMnum_arr)
    %%% Initialization %%%
    Var_SLMNum      = SLMnum_arr(q);
    Dat_Intarr      = Iarr(:,:,1:Var_SLMNum);
    upd_Ifreq       = fft2(ifftshift((Dat_Intarr)));
    Dat_SLMPhase    = SLMarr(:,:,1:Var_SLMNum);

    % Initilization of Phi; Turbulence phase
    upd_phi         = sqrt(Const_pupil.^2/sum(Const_pupil(:).^2)); 
    upd_uarr        = fftshift(fft2(ifftshift(upd_phi.*exp(1j*angle(Dat_SLMPhase(:,:,1))))))/Num_img; % Forward propagation; Image plane efield
    upd_Harr        = abs(upd_uarr).^2; % PSF
    upd_Hfreq       = fft2(ifftshift(upd_Harr));
    % Initilization of O; Target
    upd_Ofreq       = sum(conj(upd_Hfreq).*upd_Ifreq, 3)./(sum(abs(upd_Hfreq).^2,3)+DenormO); 
    upd_O           = real(fftshift(ifft2(upd_Ofreq))); 
    upd_O(upd_O<0) = 0;
    % Initilization of Harr; PSFs
    upd_Hfreq       = (conj(upd_Ofreq).*upd_Ifreq)./(abs(upd_Ofreq).^2+DenormH); 
    upd_Harr        = real(fftshift(ifft2(upd_Hfreq)));
    upd_Harr        = upd_Harr./sum(upd_Harr, [1,2]);
    upd_Harr(upd_Harr<tau2) =0; % Hard Thresholding
    upd_Harr        = upd_Harr./sum(upd_Harr, [1,2]);
    % Initilization of TO; Target smoothness
    upd_TO          = sign(Psi(upd_O)-ram1).*max(0, abs(Psi(upd_O)-ram1)-tau1/r1);

    for iter = 1:Var_Maxiter
        %%% Blind deconvolution; O, TO and Harr update %%%
        % O update
        upd_Hfreq   = fft2(ifftshift(upd_Harr));
        upd_Onorm1  = sum(conj(upd_Hfreq).*upd_Ifreq, 3);
        upd_Onorm2  = fft2(ifftshift(PsiT(upd_TO+ram1)));
        upd_Odenorm1 = sum(abs(upd_Hfreq).^2,3);
        upd_Odenorm2 = psitpsi_freq;
        upd_Ofreq   = (upd_Onorm1+r1*upd_Onorm2)./(upd_Odenorm1+r1*upd_Odenorm2+DenormO);
        upd_O = real(fftshift(ifft2(upd_Ofreq))); upd_O(upd_O<0) = 0;
        % Harr update1
        upd_Ofreq   = fft2(ifftshift(upd_O));
        upd_Hfreq   = (conj(upd_Ofreq).*upd_Ifreq)./(abs(upd_Ofreq).^2+DenormH);
        upd_Harr    = real(fftshift(ifft2(upd_Hfreq)));
        upd_Harr    = upd_Harr./sum(upd_Harr, [1,2]);
        upd_Harr(upd_Harr<tau2) =0; % Hard Thresholding for simplicity
        upd_Harr    = upd_Harr./sum(upd_Harr, [1,2]);
        % TO update
        upd_TO = sign(Psi(upd_O)-ram1).*max(0, abs(Psi(upd_O)-ram1)-tau1/r1);
        % Dual update
        ram1 = ram1 + r1*(upd_TO - Psi(upd_O));

        %%% Phase retrieval; uarr, Phi and Harr update %%%
        %%% Backward Propagation; Phi update%%%
        % uarr update1
        upd_uarr    = abs(sqrt(upd_Harr)).*exp(1j*angle(upd_uarr));
        % Phi update
        upd_Phi_arr = ifftshift(ifft2(fftshift(upd_uarr)))*Num_img.*conj(Dat_SLMPhase);
        upd_phi     = sum(upd_Phi_arr, 3)/Var_SLMNum;
        upd_phi     = sqrt(Const_pupil.^2/sum(Const_pupil(:).^2)).*exp(1j*angle(upd_phi)); % Aperture Constraint
        %%% Forward Propagation; 
        % uarr update2
        upd_uarr    = fftshift(fft2(fftshift(upd_phi.*exp(1j*angle(Dat_SLMPhase)))))/Num_img;
        % Harr update2
        upd_Harr    = abs(upd_uarr).^2;
        
        if rem(iter, 100) ==0
        subplot(4,2,[1,2]);     imagesc(Dat_Intarr(range_img,range_img,1));          axis image; colorbar; ylabel('Measurement');
        title(strcat('SLNnum:', string(q), '/', string(length(SLMnum_arr))));
        xlabel(strcat('Iter:', string(iter), '/', string(Var_Maxiter)));
        subplot(4,2,3);     imagesc(GT_image(range_img,range_img));                 axis image; colorbar; ylabel('Target'); 
        subplot(4,2,4);     imagesc(upd_O(range_img,range_img));              axis image; colorbar; 
        subplot(4,2,5);     imagesc(GT_PSF(range_pupil,range_pupil)); axis image; colorbar; ylabel('Turbulence');
        subplot(4,2,6);     imagesc(upd_Harr(range_img,range_img,1));       axis image; colorbar;

        subplot(4,2,7);     imagesc(angle(GT_turbulence(range_pupil,range_pupil))); axis image; colorbar; ylabel('Turbulence');
        subplot(4,2,8);     imagesc(angle(upd_phi(range_pupil,range_pupil)));       axis image; colorbar;
        drawnow;
        end
    end
    %%%% Tip/tilt correction; [Optional for visualization]
    % CrossCorrelation based ImageShift
    cross_corr = xcorr2(GT_image, upd_O);
    [max_corr, idx] = max(cross_corr(:));
    [row_shift, col_shift] = ind2sub(size(cross_corr), idx);
    row_offset = row_shift - Num_img; col_offset = col_shift - Num_img;
    upd_Oaligned = circshift(upd_O, [+row_offset, +col_offset]);
    upd_Orecon(:,:,q) = upd_Oaligned;
    % Grating phase from ImageShift
    img_offset = zeros(Num_img,Num_img);
    img_offset(col_offset+Num_img/2, row_offset+Num_img/2) = 1;
    GratingPhase = fftshift(fft2(ifftshift(img_offset)));
    upd_turbrecon(:,:,q) = upd_phi.*GratingPhase;
end

%%%% Visualization %%%%
figure();
Nh = 2; 
Nw = length(SLMnum_arr);
gap = [0.01, 0.01];
marg_h = [0.05, 0.05];
marg_w = [0.05, 0.05];
subplot_handles = tight_subplot(Nh, Nw+1, gap, marg_h, marg_w);
axes(subplot_handles(1));       imagesc(GT_image(range_img,range_img)); axis image; axis off; title ('GT');
axes(subplot_handles(Nw+1+1));    imagesc(angle(GT_turbulence(range_pupil,range_pupil))); axis image; axis off;
for q = 1:length(SLMnum_arr)
axes(subplot_handles(q+1));       imagesc(upd_Orecon(range_img,range_img, q)); axis image; axis off; title(strcat('Recons (M=', string(SLMnum_arr(q)),')'));
axes(subplot_handles(q+Nw+1+1));    imagesc(angle(upd_turbrecon(range_pupil,range_pupil, q))); axis image; axis off;
end
% save(strcat(pwd, '\Fig1_TABD_Simulation\ReconDat.mat'), 'upd_Orecon', 'upd_turbrecon', 'GT_image', 'GT_turbulence')
%%


%%

function [GT_image, GT_turbulence, GT_PSF, SLMarr, Iarr, Const_pupil, Num_img, range_pupil, range_img, psitpsi_freq] = main(GPU_usage, DatFolder)
    %%% Main Variable
    Num_img = 256;
    Num_pupil = 144;
    range_pupil = (Num_img/2-Num_pupil/2:Num_img/2+Num_pupil/2-1);
    range_img = (Num_img/2-128/2:Num_img/2+128/2-1);
    Const_pupil = AperConst(Num_img, Num_img, Num_pupil);
    %%% GT Data Load
    [GT_image, GT_turbulence, GT_PSF] = loadGroundTruth(Num_img, Const_pupil, DatFolder);
    %%% DataSet Load
    [SLMarr, Iarr]  = CodedData(GT_image, Const_pupil, DatFolder);
    psitpsi_freq    = PsiTPsi_freq(Num_img);
    if GPU_usage
        GT_image = gpuArray(GT_image);
        GT_turbulence = gpuArray(GT_turbulence);
        Iarr = gpuArray(Iarr);
        SLMarr = gpuArray(SLMarr);
        Const_pupil = gpuArray(Const_pupil);
        psitpsi_freq = gpuArray(psitpsi_freq);
    end
end

function [GT_image, GT_turbulence, GT_PSF] = loadGroundTruth(Num_img, Const_pupil, DatFolder)
    datafolder = strcat(pwd, '\',DatFolder,'\');
    GT_image = im2double(imread(strcat(datafolder, 'USAF-1951.png')));
    GT_image = GT_image/sum(GT_image(:));
    
    turbfile = load(strcat(datafolder, 'Turbulence.mat'));
    GT_turbulence = double(turbfile.imsdata);
    
    GT_PSF      = abs(fftshift(fft2(ifftshift(sqrt(Const_pupil.^2/sum(Const_pupil(:).^2)).*GT_turbulence))/Num_img)).^2;  
end

function [SLMarr, Iarr] = CodedData(GT_image, Const_pupil, DatFolder)
    datafolder = strcat(pwd, '\',DatFolder,'\dataset\');
    SLMarr = [];
    Iarr = [];
    for i = 1:100
        SLM_imagefile = load(strcat(datafolder, 'SLM_sim', string(i), '.mat'));
        SLM_image = double(SLM_imagefile.proj_sim);
        SLMarr(:, :, i) = Const_pupil .* exp(1j * SLM_image);

        Measurementfile = load(strcat(datafolder, 'SLM_raw', string(i), '.mat'));
        I_image = double(Measurementfile.imsdata);
        Iarr(:, :, i) = I_image;
    end
    Iarr = Iarr ./ sum(Iarr, [1, 2]);% * sum(GT_image(:));
end

function r0_const = AperConst(numx, numy, diam_size)
[x0_grid,y0_grid] = meshgrid([1:numx]-round(numx/2), [1:numy]-round(numy/2)); 
r0_grid = sqrt(x0_grid.^2 + y0_grid.^2);
r0_const = zeros(numy, numx); r0_const(r0_grid<diam_size/2) = 1;
end
function psitpsi_freq = PsiTPsi_freq(full_size)
    psitpsi = zeros(full_size); psitpsi(1, 1) = 4; psitpsi(1, 2) = -1;
    psitpsi(2, 1) = -1; psitpsi(1, end) = -1; psitpsi(end, 1) = -1;
    psitpsi_freq = abs(fft2(psitpsi));
end
function psix = Psi(x)
    diff_x1 = circshift(x, [1,0]) -x;
    diff_x2 = circshift(x, [0,1]) -x;
    psix = cat(3, diff_x1, diff_x2);
end
function psiT = PsiT(x)
    diff1 = circshift(x(:,:,1), [-1,0]) -x(:,:,1);
    diff2 = circshift(x(:,:,2), [0,-1]) -x(:,:,2);
    psiT = diff1 + diff2;
end
function [ha, pos] = tight_subplot(Nh, Nw, gap, marg_h, marg_w)
    % tight_subplot creates "subplot" axes with adjustable gaps and margins
    %
    % [ha, pos] = tight_subplot(Nh, Nw, gap, marg_h, marg_w)
    %
    %   in:  Nh      number of axes in hight (vertical direction)
    %        Nw      number of axes in width (horizontaldirection)
    %        gap     gaps between the axes in normalized units (0...1)
    %                   or [gap_h gap_w] for different gaps in height and width 
    %        marg_h  margins in height in normalized units (0...1)
    %                   or [lower upper] for different lower and upper margins 
    %        marg_w  margins in width in normalized units (0...1)
    %                   or [left right] for different left and right margins 
    %
    %  out:  ha     array of handles of the axes objects
    %                   starting from upper left corner, going row-wise as in
    %                   subplot
    %        pos    positions of the axes objects
    %
    %  Example: ha = tight_subplot(3,2,[.01 .03],[.1 .01],[.01 .01])
    %           for ii = 1:6; axes(ha(ii)); plot(randn(10,ii)); end
    %           set(ha(1:4),'XTickLabel',''); set(ha,'YTickLabel','')
    
    % Pekka Kumpulainen 21.5.2012   @tut.fi
    % Tampere University of Technology / Automation Science and Engineering
    
    
    if nargin<3; gap = .02; end
    if nargin<4 || isempty(marg_h); marg_h = .05; end
    if nargin<5; marg_w = .05; end
    
    if numel(gap)==1; 
        gap = [gap gap];
    end
    if numel(marg_w)==1; 
        marg_w = [marg_w marg_w];
    end
    if numel(marg_h)==1; 
        marg_h = [marg_h marg_h];
    end
    
    axh = (1-sum(marg_h)-(Nh-1)*gap(1))/Nh; 
    axw = (1-sum(marg_w)-(Nw-1)*gap(2))/Nw;
    
    py = 1-marg_h(2)-axh; 
    
    % ha = zeros(Nh*Nw,1);
    ii = 0;
    for ih = 1:Nh
        px = marg_w(1);
        
        for ix = 1:Nw
            ii = ii+1;
            ha(ii) = axes('Units','normalized', ...
                'Position',[px py axw axh], ...
                'XTickLabel','', ...
                'YTickLabel','');
            px = px+axw+gap(2);
        end
        py = py-axh-gap(1);
    end
    if nargout > 1
        pos = get(ha,'Position');
    end
    ha = ha(:);
end
