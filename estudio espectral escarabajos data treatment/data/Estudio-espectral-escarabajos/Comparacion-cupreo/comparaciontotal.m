%Script para graficar espectros promediados / estudio espectral 2018-2019
% Se cargan los archivos .csv
clear;
files = dir('*.csv'); %This extracts all .csv files in the current folder
[d1,d2]=size(files);
K=size(files);
J=K(1)/6;
lambda=importdata(files(5).name,',');
%%

%%
cupreoPCV=importdata(files(1).name,',');
cupreoT=importdata(files(2).name,',');
cupreoPCV_std=importdata(files(4).name,',');
cupreoT_std=importdata(files(3).name,',');
%%
plot(lambda,cupreoT,lambda,cupreoPCV)

%%
curve1PC = cupreoPCV + cupreoPCV_std;
curve2PC = cupreoPCV - cupreoPCV_std;

curve1T = cupreoT + cupreoT_std;
curve2T = cupreoT - cupreoT_std;
%figure;

% Dibujar el área sombreada entre promedio_mas_desviacion y promedio_menos_desviacion
%fill([lambda;flipud(lambda)], [curve1PC; flipud(curve2PC)], ...
%    [0.9 0.9 0.9], 'EdgeColor', 'none'); % Área sombreada
%fill([lambda;flipud(lambda)], [curve1T; flipud(curve2T)], ...
%    [0.8 0.8 0.8], 'EdgeColor', 'none'); % Área sombreada

hold on;

% Graficar la línea del promedio
plot(lambda, cupreoPCV, 'b', 'LineWidth',2);

% Graficar las líneas del promedio + desviación estándar y promedio - desviación estándar
plot(lambda, curve1PC, '--b', 'LineWidth', 1.5);
plot(lambda, curve2PC, '--b', 'LineWidth', 1.5);


plot(lambda, cupreoT, 'r', 'LineWidth', 2);
plot(lambda, curve1T, ['--r'], 'LineWidth', 1.5);
plot(lambda, curve2T, '--r', 'LineWidth', 1.5);

% Etiquetas y título
xlabel('\lambda (nm)');
ylabel('R (%)');
title('Comparison of average reflectance for \textit{C. cupreomarginata} specimens ','from two differnt regions', 'Interpreter', 'latex');
legend('std dev-PC', 'Average-PC', 'AveragePC ± std dev','std dev-T', 'Average-T', 'Average-T ± std dev');

hold off;


%%
writematrix(mean_data,"cupreo-averagetot.csv")
writematrix(std_data,"cupreotot-std.csv")