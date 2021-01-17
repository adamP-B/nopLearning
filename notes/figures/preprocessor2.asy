import deepNetwork;
size(800,0);

Network net = Network(16, 16, 16);
net.conv(16, 3, 1);
net.relu();
net.maxPool(2);
net.conv(16, 3, 1);
net.relu();
net.maxPool(2);
net.conv(16, 3, 1);
net.relu();
net.flatten();
net.linear(64);
net.relu();
net.linear(32);

net.draw(10, 14);
net.init();
net.forward();
net.sequential();

