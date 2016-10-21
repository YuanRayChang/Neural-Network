x_axis = 1:500;
figure;
h=plot(x_axis,ac001,'r',x_axis,ac004,'b',x_axis,ac008,'g--');
set(h,'linewidth',1.5)
title('Yeast','FontSize', 16);
legend('0.01','0.04','0.08','location','southeast');
xlabel('epoch','FontSize', 16);
ylabel('accuracy','FontSize', 16);
set(gcf, 'PaperPosition', [0 0 15 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [15 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'yeastcurve', 'pdf') %Save figure