%Script para graficar espectros promediados / estudio espectral 2018-2019
% Se cargan los archivos .csv
clear;
files = dir('*.csv'); %This extracts all .csv files in the current folder
[d1,d2]=size(files);
K=size(files);
J=K(1)/6;

%%
%lbls = filenames2labels("//home/marcelahj/Documents/Trabajo/Abejones/Cupreo-2019/",FileExtensions=".csv");
%lista(1)=[lbls(1)]
%for j=1:J
%    lista(j,1)=lbls(3*j-2)
%end

%writematrix(lista,"lista.csv")
%% 
%x=lbls(1)


%%
for p=1:K
    xtemp=importdata(files(p).name,',',171).data;
    data(:,p)=xtemp(:,2);
end
%%

lambda=xtemp(:,1);
std_data =std(data,1,2);
mean_data=mean(data,2);

curve1 = mean_data + std_data;
curve2 = mean_data - std_data;
figure;

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
title(['Average reflectance of the scutellum for N=2 blue \textit{Macraspis chrysis} specimens,RHCP '], 'Interpreter', 'latex');
legend('std dev', 'Average', 'Average ± std dev');

hold off;

%%
writematrix(mean_data,"macraspis-chrysis-averageR.csv")
writematrix(std_data,"macraspis-chrysis-stdR.csv")
writematrix(lambda,"lambda.csv")