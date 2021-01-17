import deepNetwork;
size(1000,0);

Network net = Network(3,16,16);
net.conv(16,5,2);
net.relu();
net.conv(16,3,1);
net.maxPool(2);
net.relu();
net.conv(16,3,1);
net.relu();
net.conv(16,3,1);
net.maxPool(2);
net.relu();
net.flatten();
net.linear(64);
net.relu();
net.linear(32);
net.relu();
net.linear(16);
net.gumbel_softmax();

net.draw(-8);
net.init();
net.forward();
net.sequential();
