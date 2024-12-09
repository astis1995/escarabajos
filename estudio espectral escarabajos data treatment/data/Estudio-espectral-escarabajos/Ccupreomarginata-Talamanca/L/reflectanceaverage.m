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
%%
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
title('Average reflectance for N=15 \textit{C. cupreomarginata} specimens ','from Talmanca Mountain Range, RHCP', 'Interpreter', 'latex');
legend('std dev', 'Average', 'Average ± std dev');

hold off;

%%

plot(lambda, mean_data, 'b', 'LineWidth', 2);
hold on;
plot(lambda, curve1, 'r', 'LineWidth', 2, 'LineStyle','--');
plot(lambda, curve2, 'r', 'LineWidth', 2, 'LineStyle','--');

%x2 = [lambda, fliplr(lambda)];
%inBetween = [curve1, fliplr(curve2)];
%fill(x2, inBetween,[0 20 20]);
%%
%x = 1:numel(mean_data);
x2 = [lambda,fliplr(lambda)];
inBetween = [curve1, fliplr(curve2)];
fill(x2, inBetween, 'g');
hold on;
plot(lambda, mean_data, 'r', 'LineWidth', 2);
%%
x = 1 : 20;
% curve1 and curve2 representing the data and its deviation
curve1 = sin(x);
curve2 = sin(x) + 0.5;
plot(x, curve1, 'r', 'LineWidth', 2, 'LineStyle','--');
hold on;
plot(x, curve2, 'b', 'LineWidth', 2);
%x2 = [x, fliplr(x)];
%inBetween = [curve1, fliplr(curve2)];
%fill(x2, inBetween, [1 0.7 1]);
%%
p=0
x1=importdata(files(p+1).name,',',171).data;
X=size(x1);
Y=X(1)-1;
acumulado=zeros(Y,1);
for j=1:J
  x1=importdata(files(p+1).name,',',171).data;
  x2=importdata(files(p+2).name,',',171).data;
  x3=importdata(files(p+3).name,',',171).data;
  
  for b=1:Y
    xprom(b,2)=(x1(b,2) +x2(b,2)+x3(b,2))/3;
    xprom(b,1)=x1(b,1);
  end
lbl=convertStringsToChars(files(2*j-1).name);
nombre=string(lbl(1:7));
writematrix(xprom,"nombre.csv")
datos=dir('nombre.csv');
datos2=importdata(datos.name,',',1 ).data;
acumulado=datos2(:,2)+acumulado;
p=p+3;
end
ondas(:,1)=datos2(:,1)
promedio=acumulado/3.;
    %%
plot(ondas,promedio)