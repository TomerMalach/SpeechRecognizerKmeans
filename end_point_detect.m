function [I1, I2] = end_point_detect(s, fs, plotFlag, extension, threshFactor)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

persistent ttt
if isempty(ttt)
    ttt=0;
end
if nargin<5
    threshFactor = 50;
end
if nargin<4
    extension = 100*1e-3;
end

winForStat = 100e-3;
winForStat = round(winForStat*fs);
extension = round(extension*fs);
win = 20e-3;
win = round(fs*win) + [0 0];
x = movmean(abs(s), win );
x1 = x(1:winForStat);
x2 = x(end-winForStat+1:end);
thresh = max(median(x1),median(x2))*threshFactor;

I1 = find(x>thresh,1, 'first');
I2 = find(x>thresh,1, 'last');
I1 = max(I1 - extension, 1);
I2 = min(I2 + extension, length(s));

if plotFlag
    figure('name','VAD');
    t = (0:size(s,1)-1)'/fs;
    ax(1) = subplot(2,1,1);
    plot(t, s);
    hold on;
    try
    plot(t(I1)+[0 0], ylim(ax(1)), 'k--');
    plot(t(I2)+[0 0], ylim(ax(1)), 'k--');
    end
%     plot(t, abs(hilbert(s)));
%     legend('s', 'AM{s}');
   
    ax(2) = subplot(2,1,2);
    plot(t, 10*log10(x));
    
    hold on;
    plot(xlim(ax(2)), 10*log10(thresh)+[0 0], 'k--');
    add_audio_cm(s, fs, 'time', ax(2));
    try
    plot(t(I1)+[0 0], ylim(ax(2)), 'k--');
    plot(t(I2)+[0 0], ylim(ax(2)), 'k--');
    end
    linkaxes(ax, 'x');
    
%     subplo
%     t = (0:n-1)'/fs
end

if ( isempty(I1) && isempty(I2) ) || (I2-I1)/fs < 300e-3;
    if ttt>30
        I1 = 1;
        I2 = length(s);
        warning('bla');
        return
    end
    ttt = ttt+1;
    [I1, I2] = end_point_detect(s, fs, plotFlag, extension, threshFactor*0.8);
    return
end
ttt=0;
end

