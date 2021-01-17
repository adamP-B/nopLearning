import deepNetwork;
size(800,0);

Network net = Network(1,28,28);
net.conv(8,7);
net.maxPool(2);
net.relu();
net.conv(10,5);
net.maxPool(2);
net.relu();
net.flatten();
net.linear(32);
net.relu();
net.linear(6);
net.view(2,3);
net.draw(3);
net.init();
net.forward();
net.sequential();
