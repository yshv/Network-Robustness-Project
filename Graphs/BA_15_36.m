load BA_15_36.mat;
BA_y = BA_15_36(:,2);
BA_x = BA_15_36(:,1);

p = polyfit(BA_x, BA_y, 1);
px = [min(BA_x), max(BA_x)];
py = polyval(p, px);

scatter(BA_x, BA_y, 'filled')
hold on;
plot(px, py, 'LineWidth', 2)

title('Wavelength assignement vs number of edges for a 15 Node 36 edge BA graph');
xlabel('Number of edges') 
ylabel('Number of wavelengths') 


