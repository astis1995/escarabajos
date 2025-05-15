%Script para graficar espectros promediados / estudio espectral 2018-2019
% Se cargan los archivos .csv
clear;
files = dir('*.csv'); %This extracts all .csv files in the current folder
[d1,d2]=size(files);
K=size(files);
J=K(1)/6;

%%
for p=1:2
    xtemp=importdata(files(p).name,',',171).data;
    data_main(:,p)=xtemp;
end
%%
for p=3:4
    xtemp=importdata(files(p).name,',',171).data;
    data_std(:,p)=xtemp;
end
%%
%lambda=xtemp(:,1);
std_data =sum(data_std,2);
mean_data_main=mean(data_main,2);
%%
curve1 = mean_data_main + std_data;
curve2 = mean_data_main - std_data;
%figure;

% Dibujar el área sombreada entre promedio_mas_desviacion y promedio_menos_desviacion
fill([lambda;flipud(lambda)], [curve1; flipud(curve2)], ...
    [0.9 0.9 0.9], 'EdgeColor', 'none'); % Área sombreada

%hold on;

% Graficar la línea del promedio
plot( mean_data_main, 'b', 'LineWidth', 2);

% Graficar las líneas del promedio + desviación estándar y promedio - desviación estándar
plot(lambda, curve1, '--r', 'LineWidth', 1.5);
plot(lambda, curve2, '--r', 'LineWidth', 1.5);

% Etiquetas y título
xlabel('\lambda (nm)');
ylabel('R (%)');
title('Average reflectance for N=15 \textit{C. cupreomarginata} specimens ','from Talmanca Mountain Range, RHCP', 'Interpreter', 'latex');
legend('std dev', 'Average', 'Average ± std dev');

hold off;
%%
plot(curve2)

%%
writematrix(mean_data,"cupreoT-averageR.csv")
writematrix(std_data,"cupreoTR-std.csv")