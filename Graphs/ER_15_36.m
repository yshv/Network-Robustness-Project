load ER_15_36.mat;
ER_y = ER_15_36(:,2);
ER_x = ER_15_36(:,1);

p = polyfit(ER_x, ER_y, 1);
px = [min(ER_x), max(ER_x)];
py = polyval(p, px);

scatter(ER_x, ER_y, 'filled');
hold on;
plot(px, py, 'LineWidth', 2);

title('Wavelength assignement vs number of edges for a 15 Node 36 edge ER graph');
xlabel('Number of edges') 
ylabel('Number of wavelengths') 


