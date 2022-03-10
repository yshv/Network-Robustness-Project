x = [2 1 4 68  4 5]
y = [5 4 2 23 21 50]
p = polyfit(x, y, 1);
px = [min(x) max(x)];
py = polyval(p, px);
scatter(x, y, 'filled')
hold on
plot(px, py, 'LineWidth', 2);