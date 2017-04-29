a = csvread('../2D_PICCS/ExperimentsCSV.csv');
a = a(1:12,2:6)
lambda1 = [0.1, 0.2, 0.3, 0.4];

format short g;

for i = 1:4
    data = a(3*(i-1)+1:3*(i),:)
    figure;
    hold on;
    plot(data(:,1),data(:,2),'b--o','Color','red')
    plot(data(:,1),data(:,3),'b--o','Color','green')
    plot(data(:,1),data(:,4),'b--o','Color','blue')
    plot(data(:,1),data(:,5),'b--o','Color','black')
    legend('FBP','CS','Prior - EigenSpace', 'Random Prior')
    fileName = sprintf('%s_Lambda1.png',num2str(lambda1(i),'%.1f'))
    print(fileName,'-dpng')
%     plot(data(:,1),data(:,2))
end