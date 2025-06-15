%% TAPBD_Experiment_LowPhotonsCase

load(strcat(pwd, '\DataSet\', 'PurdueTrain_ExpData.mat'));
GPU_usage = true;
if GPU_usage == true
    Const_pupil     = gpuArray(Const_pupil);
    DatCoded1       = gpuArray(DatCoded1);
    DatSLM1         = gpuArray(DatSLM1);
    GT_Target       = gpuArray(GT_Target);
    GT_Turbulence   = gpuArray(GT_Turbulence);
    Lens            = gpuArray(Lens);
    psitpsi_freq    = gpuArray(psitpsi_freq);
end

%% Poisson Denoiser Processing
% [Reference] Shen, Yan, et al. Infrared Physics & Technology 93 (2018): 192-198.
AverPhoton  =  mean(DatCoded1(:));
tau0        = [1+1/AverPhoton 4+1/AverPhoton];
r0          = [0.4, 30];
[DatCoded1_norm, DatCoded2_norm, GT_Target_norm]   = Ascombe_Wavelet(DatCoded1, GT_Target, 200, tau0, r0);

CodedIndex = 1;
subplot(121); imagesc(DatCoded1_norm(range_img,range_img,CodedIndex));                axis image; axis off; colorbar(); title(strcat(string(CodedIndex), '-th coded-measurement'));
subplot(122); imagesc(DatCoded2_norm(range_img,range_img,CodedIndex));                axis image; axis off; colorbar(); title(strcat(string(CodedIndex), '-th coded-measurement'));
%% TABD_Experiment (W. Poisson denoiser)

%%% Parameter Setting %%%
Var_Maxiter     = 400;

r1              = 1e-2; % TO = PsiO;
tau1            = 1e-5*r1;
DenormH         = 5e-6;
DenormO         = 1e-7;
tau2            = 4.0e-2;

Var_SLMNum_arr = 48;% Number of phase diversity used.

upd_Orecon = []; upd_Hrecon = []; upd_turbrecon =[];
for Ind_SLMNum = 1: length(Var_SLMNum_arr)
    Var_SLMNum      = Var_SLMNum_arr(Ind_SLMNum);
    Dat_SLMPhase    = DatSLM1(:,:,1:Var_SLMNum);
    DatCoded1_temp  = DatCoded2_norm(:,:,1:Var_SLMNum);
    upd_Ifreq       = fft2(ifftshift((DatCoded1_temp)));
    %%% GT PSF Plot %%%
    Phase_Ground_arr =  sqrt(Const_pupil.^2/sum(Const_pupil(:).^2)).*GT_Turbulence;
    E0_Ground = AngularSpectrum3d(Phase_Ground_arr, Var_Dist, lambda, PixelSize);
    E1_Ground = AngularSpectrum3d(E0_Ground.*Lens, Var_FL, lambda, PixelSize);
    GT_PSF = abs(E1_Ground).^2; %PSF_Ground_arr = PSF_Ground_arr/sum(PSF_Ground_arr(:));

    %%% Initilization %%%
    % Initilization of Phi; Turbulence phase
    upd_Phi         = sqrt(Const_pupil.^2/sum(Const_pupil(:).^2));
    AiPhi_arr       = AngularSpectrum3d(upd_Phi.*Dat_SLMPhase, Var_Dist, lambda, PixelSize);
    upd_uarr        = AngularSpectrum3d(AiPhi_arr.*Lens, Var_FL, lambda, PixelSize);
    % Initialiation of Harr; PSFs
    upd_Harr        = abs(upd_uarr).^2; 
    upd_Hfreq       = fft2(ifftshift(upd_Harr));
    % Initialization of O; target
    upd_Ofreq       = sum(conj(upd_Hfreq).*upd_Ifreq, 3)./(sum(abs(upd_Hfreq).^2,3)+DenormO); 
    upd_O           = real(fftshift(ifft2(upd_Ofreq))); 
    upd_O(upd_O<0) = 0;
    % Initilization of ram1; Dual variable
    ram1            = zeros([num_pixel, num_pixel, 2]);
    % Initilization of TO; Target smoothness
    upd_TO          = sign(Psi(upd_O)-ram1).*max(0, abs(Psi(upd_O)-ram1)-tau1/r1);

    for iter = 1: Var_Maxiter
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
        upd_Harr(upd_Harr<0) =0;                                
        upd_Harr = sign(upd_Harr).*max(0, abs(upd_Harr)-max(upd_Harr, [], [1,2])*tau2);  % Soft Thresholding
        % upd_Harr = sign(upd_Harr).*max(0, abs(upd_Harr)-tau2);
        upd_Harr = upd_Harr.*AperConst(num_pixel, num_pixel, 120); % Limited cutoff frequency by pupil plane pixel.
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
        AiHui_arr0  = conj(Lens).*AngularSpectrum3d(upd_uarr, -Var_FL, lambda, PixelSize);
        AiHui_arr1  = conj(Dat_SLMPhase).*AngularSpectrum3d(AiHui_arr0, -Var_Dist, lambda, PixelSize);
        upd_Phi     = sum(AiHui_arr1, 3)/Var_SLMNum; upd_Phi = sqrt(Const_pupil.^2/sum(Const_pupil(:).^2)).*exp(1j*angle(upd_Phi));
        %%% Forward Propagation; uarr and Harr update %%%
        AiPhi_arr0  = AngularSpectrum3d(upd_Phi.*Dat_SLMPhase, Var_Dist, lambda, PixelSize);
        % uarr update2
        upd_uarr    = AngularSpectrum3d(AiPhi_arr0.*Lens, Var_FL, lambda, PixelSize);
        % Harr update2
        upd_Harr    = abs(upd_uarr).^2;
        if rem(iter, 50) == 0
            subplot(4,2,1); imagesc(DatCoded1_norm(range_img,range_img,1));             axis image; colorbar; title('raw measurement'); colormap('default');
            xlabel(strcat('Iter:', string(iter), '/', string(Var_Maxiter))); 
            subplot(4,2,2); imagesc(DatCoded1_temp(range_img,range_img,1));             axis image; colorbar; title('Denoised-measurement');
            xlabel(strcat('SLNnum index:', string(Ind_SLMNum), '/', string(length(Var_SLMNum_arr))));
            
            subplot(4,2,3);     imagesc(GT_Target_norm(range_img,range_img));                    axis image; colorbar; ylabel('Target'); title('GT');
            subplot(4,2,4);     imagesc(upd_O(range_img,range_img));                        axis image; colorbar; title('RT');
            subplot(4,2,5);     imagesc(GT_PSF(range_PSF,range_PSF, 1));            axis image; colorbar; ylabel('PSF');
            subplot(4,2,6);     imagesc(upd_Harr(range_PSF,range_PSF,1));                    axis image; colorbar;
            subplot(4,2,7);     imagesc(angle(GT_Turbulence(range_pupil,range_pupil)));     axis image; colorbar; ylabel('Turbulence');
            subplot(4,2,8);     imagesc(angle(upd_Phi(range_pupil,range_pupil)));           axis image; colorbar;
            drawnow;
        end
    end
    %%%% Tip/tilt correction; [Optional for visualization]
    % CrossCorrelation based ImageShift
    cross_corr = xcorr2(GT_Target_norm, upd_O);
    [max_corr, idx] = max(cross_corr(:));
    [col_shift, row_shift] = ind2sub(size(cross_corr), idx);
    col_offset = col_shift - num_pixel; row_offset = row_shift - num_pixel;
    upd_Oaligned = circshift(upd_O, [+col_offset, +row_offset]);
    upd_Haligned = circshift(upd_Harr(:,:,1), [-col_offset, -row_offset]);
    upd_Orecon(:,:,Ind_SLMNum) = upd_Oaligned;
    upd_Hrecon(:,:,Ind_SLMNum) = upd_Haligned;
    % Grating phase from ImageShift
    img_offset = zeros(num_pixel,num_pixel);
    img_offset(-col_offset+num_pixel/2, -row_offset+num_pixel/2) = 1;
    Gratingprop   = conj(Lens).*AngularSpectrum3d(img_offset, -Var_FL, lambda, PixelSize);
    GratingPhase  = AngularSpectrum3d(Gratingprop, -Var_Dist, lambda, PixelSize);
    upd_turbrecon(:,:,Ind_SLMNum) = upd_Phi.*GratingPhase;
end
%%%% Visualization %%%%
figure(); 
Nh = 3; 
Nw = length(Var_SLMNum_arr);
gap = [0.01, 0.01];
marg_h = [0.05, 0.05];
marg_w = [0.05, 0.05];
subplot_handles = tight_subplot(Nh, Nw+1, gap, marg_h, marg_w);
axes(subplot_handles(1));             imagesc(rot90(rot90(GT_Target(range_img,range_img)))); axis image; axis off; title ('GT'); %colormap('gray');
axes(subplot_handles((Nw+1)*1+1));    imagesc(abs(rot90(rot90(GT_PSF(range_pupil,range_pupil))))); axis image; axis off; 
axes(subplot_handles((Nw+1)*2+1));    imagesc(angle(rot90(rot90(GT_Turbulence(range_pupil,range_pupil))))); axis image; axis off; 

for q = 1:length(Var_SLMNum_arr)
axes(subplot_handles(q+1));             imagesc(rot90(rot90(upd_Orecon(range_img,range_img, q)))); axis image; axis off; title(strcat('Recons (M=', string(Var_SLMNum_arr(q)),')'));
axes(subplot_handles((Nw+1)*1+q+1));    imagesc(rot90(rot90(abs(upd_Hrecon(range_pupil,range_pupil, q))))); axis image; axis off; 
axes(subplot_handles((Nw+1)*2+q+1));    imagesc(rot90(rot90(angle(upd_turbrecon(range_pupil,range_pupil, q))))); axis image; axis off; 
end

%% Data Loading and processing
function [DatSLM0, DatSLM1, Lens, DatCoded0, DatCoded1, Const_pupil, psitpsi_freq, PixelSize, num_pixel, num_pupil, AverPhoton] = main(GPU_usage, Index_Exposure, Index_Scene, Index_Scaling, Var_FL, lambda, Exposure_arr, DatFolder)
    
    %%% Raw Data %%%
    % GroundTruth Turbulence
    datafile = strcat(pwd, '\', DatFolder, '\GT_Turbulence\Turbulence_DoverFried16');
    load(datafile, 'SLM0');
    SLM0 = SLM0(:,:,Index_Scene);
    % GroundTruth Image; Diffraction-limited image WO Turbulence effect
    datafile = strcat(pwd, '\', DatFolder, '\DiffractionLimitedImage.mat');
    load(datafile, 'im_img1');
    im_img1 = im_img1(:,:,1);
    % PhaseDiversity Patterns; Aper5.0e-03_Zern4_Var20
    datafile = strcat(pwd, '\', DatFolder, '\SLMDiversityPatterns\PhaseDiversity_Var20_Num128');
    load(datafile, 'SLM1');
    % CodedMeasurements
    datafile = strcat(pwd, '\', DatFolder, '\CodedMeasurement\Scene',string(Index_Scene), '_', string(Exposure_arr(Index_Exposure)), 'us');
    load(datafile, 'im_img2');
    % Darknoise;
    datafile = strcat(pwd, '\', DatFolder, '\DarkNoise\DarkNoise_', string(Exposure_arr(Index_Exposure)), 'us');
    load(datafile, 'im_img3');

    %%% DataProcessing %%%
    % DarknoiseSubtraction
    DarkNoise = mean(im_img3, 3);
    im_img1 = im_img1 - DarkNoise; im_img1(im_img1<0) = 0;
    im_img2 = im_img2 - DarkNoise; im_img2(im_img2<0) = 0;
    % ImageShift
    Shiftx = 256-190; Shifty = 256-185;
    im_img1 = circshift(im_img1, [Shifty, Shiftx, 0]);
    im_img2 = circshift(im_img2, [Shifty, Shiftx, 0]);
    % PhotonConversion
    Gain_CCD = 1/3.38;
    Gain_MCP = (500-5.48)/(100-1);
    Gain_Base = Gain_CCD*Gain_MCP;
    IntensifierGain = 5;
    Gain_Result = Gain_Base*IntensifierGain;
    TotalSignal = sum(im_img2(:))/size(im_img2,3);
    TotalPhoton = TotalSignal/Gain_Result;
    AverPhoton  = TotalPhoton/200^2;

    % ImageRescaling
    [DatSLM0, DatSLM1, DatCoded0, DatCoded1, PixelSize, num_pixel] = ImageScalingSwap(Index_Scaling, SLM0, SLM1, im_img1, im_img2);
    % DataFlip to match experimental condition
    DatSLM0 =    flip(DatSLM0, 1);
    DatSLM1 =    flip(DatSLM1, 1);

    num_pupil = round(3e-3/PixelSize/2)*2;
    Const_pupil = AperConst(num_pixel, num_pixel, num_pupil);
    DatSLM0 = Const_pupil.*exp(1j*DatSLM0);
    DatSLM1 = Const_pupil.*exp(1j*DatSLM1);
    % Normalization
    % DatCoded0 = DatCoded0./sum(DatCoded0(:));
    % DatCoded1 = DatCoded1./sum(DatCoded1, [1,2]);

    Lens            = Lens_phase(Const_pupil, Var_FL, lambda, PixelSize);
    psitpsi_freq    = PsiTPsi_freq(num_pixel);
    if GPU_usage
        DatSLM0 = gpuArray(DatSLM0);
        DatSLM1 = gpuArray(DatSLM1);
        DatCoded0 = gpuArray(DatCoded0);
        DatCoded1 = gpuArray(DatCoded1);
        Const_pupil = gpuArray(Const_pupil);
        Lens = gpuArray(Lens);
        psitpsi_freq = gpuArray(psitpsi_freq);
    end
   
end

function [DatSLM0, DatSLM1, DatCoded0, DatCoded1, PixelSize, num_pixel] = ImageScalingSwap(Method_Ind, SLM0, SLM1, im_img1, im_img2)
    % Define pixel sizes and dimensions
    PixelSize_SLM   = 8.0e-6;
    PixelSize_CCD   = 16e-6 * 1.48;
    numx_SLM = 1920; numy_SLM = 1200;
    numx_CCD = 512;  numy_CCD = 512;
    % Initialize outputs
    DatSLM0     = []; % GT Turbulence phase
    DatSLM1     = []; % SLM phase diversity
    DatCoded0   = []; % GT target
    DatCoded1   = []; % Coded measurements
    if Method_Ind == 0 % CCD coord matching
        ScalingFactor = PixelSize_SLM / PixelSize_CCD;
        new_numx_SLM = round(numx_SLM * ScalingFactor);
        new_numy_SLM = round(numy_SLM * ScalingFactor);
        
        % Interpolation grid
        [x_SLM, y_SLM] = meshgrid(1:numx_SLM, 1:numy_SLM);
        [xq, yq] = meshgrid(linspace(1, numx_SLM, new_numx_SLM), linspace(1, numy_SLM, new_numy_SLM));

        DatSLM0 = process_image(SLM0, x_SLM, y_SLM, xq, yq, numx_CCD, numy_CCD, false);
        % DatSLM0 = process_image(SLM0, x_SLM, y_SLM, xq, yq, numx_CCD, numy_CCD);
        DatSLM1 = process_image(SLM1, x_SLM, y_SLM, xq, yq, numx_CCD, numy_CCD);
        DatCoded0 = im_img1;
        DatCoded1 = im_img2;
        
        PixelSize = PixelSize_CCD;
        num_pixel = numx_CCD;

    elseif Method_Ind == 1 % SLM coord matching
        ScalingFactor = PixelSize_CCD / PixelSize_SLM;
        new_numx_CCD = round(numx_CCD * ScalingFactor);
        new_numy_CCD = round(numy_CCD * ScalingFactor);
        
        % Interpolation grid
        [x_CCD, y_CCD] = meshgrid(1:numx_CCD, 1:numy_CCD);
        [xq, yq] = meshgrid(linspace(1, numx_CCD, new_numx_CCD), linspace(1, numy_CCD, new_numy_CCD));
        
        DatCoded0 = process_image(im_img1, x_CCD, y_CCD, xq, yq, numx_SLM, numy_SLM, false);
        % DatCoded0 = process_image(im_img1, x_CCD, y_CCD, xq, yq, numx_SLM, numy_SLM);
        DatCoded1 = process_image(im_img2, x_CCD, y_CCD, xq, yq, numx_SLM, numy_SLM);
        DatSLM0 = SLM0;
        DatSLM1 = SLM1;
        % Croping to make square image
        DatCoded0 = DatCoded0(1:numy_SLM, numx_SLM/2-numy_SLM/2:numx_SLM/2+numy_SLM/2-1, :);
        DatCoded1 =  DatCoded1(1:numy_SLM, numx_SLM/2-numy_SLM/2:numx_SLM/2+numy_SLM/2-1, :);
        DatSLM0 = DatSLM0(1:numy_SLM, numx_SLM/2-numy_SLM/2:numx_SLM/2+numy_SLM/2-1, :);
        DatSLM1 = DatSLM1(1:numy_SLM, numx_SLM/2-numy_SLM/2:numx_SLM/2+numy_SLM/2-1, :);

        PixelSize = PixelSize_SLM;
        num_pixel = numy_SLM;
       
    end
end
function output = process_image(input, x, y, xq, yq, target_x, target_y, is3D)
    if nargin < 8
        is3D = true;
    end
    if is3D
        output = zeros(target_y, target_x, size(input, 3));
        for i = 1:size(input, 3)
            img_resized = interp2(x, y, input(:, :, i), xq, yq);
            output(:, :, i) = pad_or_crop(img_resized, target_x, target_y);
        end
    else
        img_resized = interp2(x, y, input, xq, yq);
        output = pad_or_crop(img_resized, target_x, target_y);
    end
end
function result = pad_or_crop(image, target_x, target_y)
    [current_y, current_x] = size(image);
    % Pad or crop in x dimension
    if current_x < target_x
        pad_x = ceil((target_x - current_x) / 2);
        image = padarray(image, [0 pad_x], 0, 'both');
        image = image(:, 1:target_x);
    elseif current_x > target_x
        crop_x = floor((current_x - target_x) / 2) + 1;
        image = image(:, crop_x:crop_x + target_x - 1);
    end
    % Pad or crop in y dimension
    if current_y < target_y
        pad_y = ceil((target_y - current_y) / 2);
        image = padarray(image, [pad_y 0], 0, 'both');
        image = image(1:target_y, :);
    elseif current_y > target_y
        crop_y = floor((current_y - target_y) / 2) + 1;
        image = image(crop_y:crop_y + target_y - 1, :);
    end
    result = image;
end
%%
function [x0arr, x1arr, GT_Target] = Ascombe_Wavelet(RawImage, GT_Target, Max_iter, tau0, r0)
    r1 = r0(1); r2 = r0(2); %r3 = r0(3); r4 = r0(4);
    tau1 = tau0(1); tau2 = tau0(2); %tau3 = tau0(3); tau4 = tau0(4);
    waveletName = 'sym4';
    copt = 0.108;
    for SLM_num = 1:size(RawImage,3)
        [LL, ~, ~, HH] = dwt2(RawImage(:,:,SLM_num), waveletName);
        % LL(LL<0)= 0; HH(HH<0)= 0; LH(LH<0)= 0; HL(HL<0)= 0;
        MeasuredT1 = 2*sqrt(complex(LL +copt));
        % MeasuredT2 = 2*sqrt(LH +copt);
        % MeasuredT3 = 2*sqrt(HL +copt);
        MeasuredT2 = 2*sqrt(complex(HH +copt)); 
        q1 = MeasuredT1; q2 = MeasuredT2; %q3 = MeasuredT3; q4 = MeasuredT4; % ground truth
        s1 = MeasuredT1; s2 = MeasuredT2; %s3 = MeasuredT3; s4 = MeasuredT4; % Estimation
        psitpsi_freq = PsiTPsi_freq(size(q1,1));
        for i = 1: Max_iter
            %%% Penalty %%%
            Ts1 = sign(Psi(s1)).*max(0, abs(Psi(s1))-tau1/r1);
            s1_freq = fft2(ifftshift((q1+ r1*PsiT(Ts1))))./(1+r1*psitpsi_freq);
            s1 = real(fftshift(ifft2(s1_freq))); %s1(s1<0) = 0;
        
            Ts2 = sign(Psi(s2)).*max(0, abs(Psi(s2))-tau2/r2);
            s2_freq = fft2(ifftshift((q2+ r2*PsiT(Ts2))))./(1+r2*psitpsi_freq);
            s2 = real(fftshift(ifft2(s2_freq))); %s2(s2<0) = 0;
        end

        z1 = (s1/2).^2 - copt; z2 = (s2/2).^2 - copt;
        x1 = idwt2(z1, zeros(size(z1)), zeros(size(z1)), z2, waveletName);
        x1 = x1- min(x1(:));
        % cutoff1 = size(x1,1)/3;
        % x1filter = fourier_filter(x1, 'low-pass', cutoff1); % Limited cutoff frequency by pupil plane pixel.
        x1filter = x1;
        x1arr(:,:,SLM_num) = x1filter;
    end
    % Normalization
    x0arr       = RawImage./sum(RawImage, [1,2]);
    x1arr       = x1arr./sum(x1arr, [1,2]);
    % x1arr(x1arr<mean(x1arr(1:40,1:40,:), 'all')*2) =0;
    x1arr       = x1arr./sum(x1arr, [1,2]);
    GT_Target   = GT_Target/sum(GT_Target(:));
end
function filtered_image = fourier_filter(image, filter_type, cutoff)
    F = fft2(double(image));
    F_shifted = fftshift(F);
    [rows, cols] = size(image);
    cx = round(rows / 2); 
    cy = round(cols / 2); 
    [x, y] = meshgrid(1:cols, 1:rows);
    dist = sqrt((x - cx).^2 + (y - cy).^2);
    if strcmp(filter_type, 'low-pass')
        filter = double(dist <= cutoff);
    elseif strcmp(filter_type, 'high-pass')
        filter = double(dist > cutoff);
    else
        error('Invalid filter type. Choose either "low-pass" or "high-pass".');
    end
    F_filtered = F_shifted .* filter;
    F_inv_shifted = ifftshift(F_filtered);
    filtered_image = real(ifft2(F_inv_shifted));

end

%%
function r0_const = AperConst(numx, numy, diam_size)
[x0_grid,y0_grid] = meshgrid([1:numx]-round(numx/2), [1:numy]-round(numy/2)); 
r0_grid = sqrt(x0_grid.^2 + y0_grid.^2);
r0_const = zeros(numy, numx); r0_const(r0_grid<diam_size/2) = 1;
end
function Lens_term = Lens_phase(U0, f, lambda, SLM_pixel)
    k = 2*pi/lambda;
    [SLM_ynum, SLM_xnum] = size(U0);
    Lensx = SLM_pixel*SLM_xnum; Lensy = SLM_pixel*SLM_ynum;
    dx = linspace(-Lensx/2, Lensx/2- SLM_pixel, SLM_xnum);
    dy = linspace(-Lensy/2, Lensy/2- SLM_pixel, SLM_ynum);
    [dx_grid, dy_grid] = meshgrid(dx, dy);
    dst = sqrt(dx_grid.^2 +dy_grid.^2);
    Lens_term = exp(1j*k/(2*f)*dst.^2);
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
function psitpsi_freq = PsiTPsi_freq(full_size)
    psitpsi = zeros(full_size); psitpsi(1, 1) = 4; psitpsi(1, 2) = -1;
    psitpsi(2, 1) = -1; psitpsi(1, end) = -1; psitpsi(end, 1) = -1;
    psitpsi_freq = abs(fft2(psitpsi));
end
function U2 = AngularSpectrum3d(u0, z, lambda, PixelSize) 
    k = 2*pi/lambda;
    [SLM_ynum, SLM_xnum, SLM_knum] = size(u0);
    SLM_ysize = PixelSize*SLM_ynum; SLM_xsize = PixelSize*SLM_xnum;
    
    dfx = 1/SLM_xsize; dfy = 1/SLM_ysize;
    fx = linspace(-0.5/PixelSize, 0.5/PixelSize - dfx, SLM_xnum);
    fy = linspace(-0.5/PixelSize, 0.5/PixelSize - dfy, SLM_ynum);
    [fx_grid, fy_grid] = meshgrid(fx, fy);

    U0 = fftshift(fft2(ifftshift(u0))); % 3D FFT
    H0 = exp(1j*pi*lambda*z*(fx_grid.^2+fy_grid.^2));
    H0_3D = repmat(H0, [1, 1, SLM_knum]);
    U1 = U0.*H0_3D;
    U2 = ifftshift(ifft2(fftshift(U1)));
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
