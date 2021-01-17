
import deepNetwork;
size(200,0);

Network net = Network(64);
net.linear(64);
net.relu();
net.linear(32);
net.relu();
net.linear(15);


net.draw(10, 14);
net.init();
net.forward();
net.sequential();
