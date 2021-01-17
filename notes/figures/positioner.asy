import deepNetwork;
size(1000,0);

Network net = Network(18,16,16);
net.conv(16,3,1);
net.relu();
net.conv(16,3,1);
net.maxPool(2);
net.relu();
net.conv(16,3,1);
net.relu();
net.conv(16,3,1);
net.maxPool(2);
net.conv(16,3,1);
net.flatten();
net.linear(32);
net.relu();
net.linear(4);

net.draw(-8);
net.init();
net.forward();
net.sequential();
