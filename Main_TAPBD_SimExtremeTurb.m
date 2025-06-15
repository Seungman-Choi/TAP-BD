% Data Preparation
load(strcat(pwd, '\DataSet\Sat_simData.mat'))
GPU_usage = true;
if GPU_usage == true
    GT_image        = gpuArray(GT_image);
    GT_turbulence   = gpuArray(GT_turbulence);
    SLMarr          = gpuArray(SLMarr);
    Iarr            = gpuArray(Iarr);
    Const_pupil     = gpuArray(Const_pupil);
end
range_img       = (Num_img/2-160/2:Num_img/2+160/2-1);
range_pupil     = (Num_img/2-Num_pupil/2:Num_img/2+Num_pupil/2-1);

Var_SLMNum      = 100;
Var_Maxiter     = 1000;

% Ground Truth image and PSF generation
GT_image        = GT_image./sum(GT_image(:));
GT_PSF          = abs(fftshift(fft2(ifftshift(sqrt(Const_pupil.^2/sum(Const_pupil(:).^2)).*GT_turbulence)))).^2;
GT_PSF          = GT_PSF/sum(GT_PSF(:));

% Required Data input
Dat_Intarr        = Iarr./sum(Iarr, [1,2]);
upd_Ifreq       = fft2(ifftshift((Dat_Intarr)));
Dat_SLMPhase    = SLMarr(:,:,1:Var_SLMNum);

% Hyper-parameter setting
tau2                = 2e-3;
tau2_arr            = linspace(1e-2, 4e-2, length(Var_SLMNum));
r3                  = 1e-2; 
tau3                = 1e-6*r3;
ram3                = zeros(Num_img, Num_img, 2);
r9                  = 2e-4; 
r10                 = 2e-4;
Threshold_Harr      = 6e-5;
Threshold_Oarr      = 0;
psitpsi_freq        = PsiTPsi_freq(Num_img);

%%%% Initilization %%%%
upd_phi     = sqrt(Const_pupil.^2/sum(Const_pupil(:).^2));
upd_uarr    = fftshift(fft2(ifftshift(upd_phi.*exp(1j*angle(Dat_SLMPhase(:,:,1))))))/Num_img;
upd_Harr    = abs(upd_uarr).^2;
upd_Hfreq   = fft2(ifftshift(upd_Harr));
upd_Ofreq   = sum(conj(upd_Hfreq).*upd_Ifreq, 3)./(sum(abs(upd_Hfreq).^2,3)+r10);
upd_O = real(fftshift(ifft2(upd_Ofreq))); upd_O(upd_O<0) = 0;
upd_O(upd_O<Threshold_Oarr) = 0;
upd_Hfreq   = (conj(upd_Ofreq).*upd_Ifreq)./(abs(upd_Ofreq).^2+r9); 
upd_Harr =      real(fftshift(ifft2(upd_Hfreq)));
upd_Harr = upd_Harr./sum(upd_Harr, [1,2]);
upd_Harr = sign(upd_Harr).*max(0, abs(upd_Harr)-max(upd_Harr, [], [1,2])*tau2);
upd_Harr = upd_Harr./sum(upd_Harr, [1,2]);
upd_TO = sign(Psi(upd_O)-ram3).*max(0, abs(Psi(upd_O)-ram3)-tau3/r3);
tic
for iter = 1:Var_Maxiter
    %%% O update %%%
    upd_Hfreq   = fft2(ifftshift(upd_Harr));
    upd_Onorm1  = sum(conj(upd_Hfreq).*upd_Ifreq, 3);
    upd_Onorm2  = fft2(ifftshift(PsiT(upd_TO+ram3)));
    upd_Odenorm1 = sum(abs(upd_Hfreq).^2,3);
    upd_Odenorm2 = psitpsi_freq;
    upd_Ofreq   = (upd_Onorm1+r3*upd_Onorm2)./(upd_Odenorm1+r3*upd_Odenorm2+r10);
    upd_O = real(fftshift(ifft2(upd_Ofreq))); upd_O(upd_O<0) = 0;
    upd_O(upd_O<Threshold_Oarr) = 0;
    %%% Harr update %%%
    upd_Ofreq   = fft2(ifftshift(upd_O));
    upd_Hfreq   = (conj(upd_Ofreq).*upd_Ifreq)./(abs(upd_Ofreq).^2+r9);
    upd_Harr    = real(fftshift(ifft2(upd_Hfreq)));
    upd_Harr    = upd_Harr./sum(upd_Harr, [1,2]);
    upd_Harr = sign(upd_Harr).*max(0, abs(upd_Harr)-max(upd_Harr, [], [1,2])*tau2);
    upd_Harr = upd_Harr.*AperConst(Num_img, Num_img, Num_img*3/4);
    upd_Harr    = upd_Harr./sum(upd_Harr, [1,2]);

    %%% TO update %%%
    upd_TO = sign(Psi(upd_O)-ram3).*max(0, abs(Psi(upd_O)-ram3)-tau3/r3);
    %%% uarr update %%%
    upd_uarr    = abs(sqrt(abs(upd_Harr))).*exp(1j*angle(upd_uarr));
    %%% Phi update %%%
    upd_Phi_arr = ifftshift(ifft2(fftshift(upd_uarr))).*conj(Dat_SLMPhase)*Num_img;
    upd_phi     = sum(upd_Phi_arr, 3)/Var_SLMNum;
    upd_phi     = sqrt(Const_pupil.^2/sum(Const_pupil(:).^2)).*exp(1j*angle(upd_phi));
    upd_uarr    = fftshift(fft2(fftshift(upd_phi.*exp(1j*angle(Dat_SLMPhase)))))/Num_img;
    upd_Harr    = abs(upd_uarr).^2;
    %%% Dual update %%%
    ram3 = ram3 + r3*(upd_TO - Psi(upd_O));
    %%% Visualization %%%
    if rem(iter, 20) == 0
        figure(1); clf;

        cmap_target = gray;
        cmap_PSF    = turbo;
        cmap_turb   = parula;
        cmap_other  = gray;

        % tiledlayout
        t = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
        set(gcf, 'Color', 'w');
        set(gcf, 'Units','normalized','Position',[0.1 0.1 0.8 0.6]);  % 창 크기

        % 1-1. Turbulent measurement (span two rows)
        ax1 = nexttile([2 1]);
        imagesc(Dat_Intarr(range_img,range_img,1));
        axis image; set(gca, 'XTick', [], 'YTick', []);
        title({'Turbulent measurement', '(D/r0 = 45)'}, 'FontSize', 17);
        colormap(ax1, cmap_other);
        c1 = colorbar;
        c1.FontSize = 12;
        c1.FontWeight = 'normal';

        % 1-2. GT Target
        ax2 = nexttile;
        imagesc(GT_image(range_img,range_img));
        axis image; set(gca, 'XTick', [], 'YTick', []);
        title("Target", 'FontSize', 17);
        ylabel('Ground truth', 'FontSize', 17, 'FontWeight', 'bold');
        colormap(ax2, cmap_target);
        c2 = colorbar;
        c2.FontSize = 12;
        c2.FontWeight = 'normal';

        % 1-3. GT PSF
        ax4 = nexttile;
        imagesc(GT_PSF(range_pupil,range_pupil));
        axis image; set(gca, 'XTick', [], 'YTick', []);
        title("PSF", 'FontSize', 17);
        colormap(ax4, cmap_PSF);
        c4 = colorbar;
        c4.FontSize = 12;
        c4.FontWeight = 'normal';

        % 1-4. GT Phase
        ax6 = nexttile;
        imagesc(angle(GT_turbulence(range_pupil,range_pupil)));
        axis image; set(gca, 'XTick', [], 'YTick', []);
        title("Turbulence phase", 'FontSize', 17);
        colormap(ax6, cmap_turb);
        c6 = colorbar;
        c6.FontSize = 12;
        c6.FontWeight = 'normal';

        % 2-2. Estimated Target
        ax3 = nexttile;
        imagesc(upd_O(range_img,range_img));
        axis image; set(gca, 'XTick', [], 'YTick', []);
        colormap(ax3, cmap_target);
        c3 = colorbar;
        c3.FontSize = 12;
        c3.FontWeight = 'normal';
        ylabel('Reconstruction', 'FontSize', 17, 'FontWeight', 'bold');

        text(0.05, 0.95, sprintf('Iteration: %d/%d', iter, Var_Maxiter), ...
            'Units', 'normalized', ...
            'FontSize', 17, ...
            'FontWeight', 'bold', ...
            'Color', 'w', ...
            'HorizontalAlignment', 'left', ...
            'VerticalAlignment', 'top');

        % 2-3. Estimated PSF
        ax5 = nexttile;
        imagesc(upd_Harr(range_pupil,range_pupil,1));
        axis image; set(gca, 'XTick', [], 'YTick', []);
        colormap(ax5, cmap_PSF);
        c5 = colorbar;
        c5.FontSize = 12;
        c5.FontWeight = 'normal';

        % 2-4. Estimated Phase
        ax7 = nexttile;
        imagesc(angle(upd_phi(range_pupil,range_pupil)));
        axis image; set(gca, 'XTick', [], 'YTick', []);
        colormap(ax7, cmap_turb);
        c7 = colorbar;
        c7.FontSize = 12;
        c7.FontWeight = 'normal';
        drawnow;
    end
end
toc

%%
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

% tightfig.m (https://www.mathworks.com/matlabcentral/fileexchange/34055-tightfig)
function hfig = tightfig1(hfig)
    if nargin == 0, hfig = gcf; end
    set(hfig, 'Units', 'centimeters');
    pos = get(hfig, 'Position');

    ti = [0 0 0 0];
    for i = 1:length(hfig.Children)
        try
            ax = hfig.Children(i);
            if strcmp(get(ax, 'Type'), 'axes')
                ti = max(ti, get(ax, 'TightInset'));
            end
        end
    end

    for i = 1:length(hfig.Children)
        try
            ax = hfig.Children(i);
            if strcmp(get(ax, 'Type'), 'axes')
                pos_ax = get(ax, 'Position');
                pos_ax(1) = pos_ax(1) - ti(1);
                pos_ax(2) = pos_ax(2) - ti(2);
                pos_ax(3) = pos_ax(3) + ti(1) + ti(3);
                pos_ax(4) = pos_ax(4) + ti(2) + ti(4);
                set(ax, 'Position', pos_ax);
            end
        end
    end
    drawnow;
end
%%
function tightfig(hfig)
    if nargin == 0
        hfig = gcf;
    end

    % Get all axes in figure
    ax = findall(hfig, 'type', 'axes');

    % Get their outer positions and tight insets
    for i = 1:length(ax)
        outerpos{i} = get(ax(i), 'OuterPosition');
        ti{i}       = get(ax(i), 'TightInset');
    end

    % Tighten each axes
    for i = 1:length(ax)
        left   = outerpos{i}(1) + ti{i}(1);
        bottom = outerpos{i}(2) + ti{i}(2);
        ax_width  = outerpos{i}(3) - ti{i}(1) - ti{i}(3);
        ax_height = outerpos{i}(4) - ti{i}(2) - ti{i}(4);
        set(ax(i), 'Position', [left, bottom, ax_width, ax_height]);
    end

    % Adjust figure size to fit tightly
    drawnow;
    outerpos_all = cell2mat(cellfun(@(p,t) [p(1)+t(1), p(2)+t(2), p(1)+p(3)-t(3), p(2)+p(4)-t(4)], outerpos, ti, 'UniformOutput', false));
    tight_rect = [min(outerpos_all(:,1)), min(outerpos_all(:,2)), ...
                  max(outerpos_all(:,3)), max(outerpos_all(:,4))];

    set(hfig, 'Units', 'normalized');
    fig_pos = get(hfig, 'Position');
    new_width = fig_pos(3) * (tight_rect(3) - tight_rect(1));
    new_height = fig_pos(4) * (tight_rect(4) - tight_rect(2));
    set(hfig, 'Position', [fig_pos(1), fig_pos(2), new_width, new_height]);
end