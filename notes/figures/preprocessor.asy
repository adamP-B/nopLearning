import deepNetwork;
size(800,0);

Network net = Network(3, 64, 64);
net.conv(16, 5, 2);
net.relu();
net.conv(16, 3, 1);
net.relu();
net.maxPool(2);
net.conv(16, 3, 1);
net.relu();
net.maxPool(2);


net.draw(10, 25);
net.init();
net.forward();
net.sequential();


