%Script para graficar espectros promediados / estudio espectral 2018-2019
% Se cargan los archivos .csv
clear;
files = dir('*.csv'); %This extracts all .csv files in the current folder
[d1,d2]=size(files);
K=size(files);
J=K(1)/6;
lambda=importdata(files(5).name,',');

%%
for p=1:2
    xtemp=importdata(files(p).name,',');
    main(:,p)=xtemp;
    xtemp=importdata(files(p+2).name,',');
    std(:,p)=xtemp;
end
%%

std_data =sum(std,2);
mean_data=mean(main,2);
%%
curve1 = mean_data + std_data;
curve2 = mean_data - std_data;
%figure;

% Dibujar el área sombreada entre promedio_mas_desviacion y promedio_menos_desviacion
fill([lambda;flipud(lambda)], [curve1; flipud(curve2)], ...
    [0.9 0.9 0.9], 'EdgeColor', 'none'); % Área sombreada

hold on;

% Graficar la línea del promedio
plot(lambda, mean_data, 'b', 'LineWidth', 2);

% Graficar las líneas del promedio + desviación estándar y promedio - desviación estándar
plot(lambda, curve1, '--r', 'LineWidth', 1.5);
plot(lambda, curve2, '--r', 'LineWidth', 1.5);

% Etiquetas y título
xlabel('\lambda (nm)');
ylabel('R (%)');
title('Average reflectance of the scutellum for N=2 \textit{Macraspis chrysis} specimens ','Interpreter', 'latex');
legend('std dev', 'Average', 'Average ± std dev');

hold off;


%%
writematrix(mean_data,"macraspis-chrysis-averagetot.csv")
writematrix(std_data,"macraspis-chrysis-std.csv")