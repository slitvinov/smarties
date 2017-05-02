load stats.txt;
str1 = {'epoch','avg mse','avg rel err','avg Q','min Q','max Q','errWeights','errWeights','errWeights','N','steps','dT'};

ind=[4,5,6];

for k=ind
    plot(stats(:,1),stats(:,k));
    hold on
end
legend(str1(ind))
ax=gca;
ax.FontSize=15;

load master_rewards.dat;
str2 = {'Iter','Mean reward','variance'};
