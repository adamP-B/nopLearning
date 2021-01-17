import myutil;
import block;
size(650,0);

pair[] loc = {(0,0), (0,5), (3,0), (6,0), (9,2), (9,5), (12,3.5), (15,3.5), (15,0), (19,0), (22,0), (24,0)};

pen imageCol = blue;
Image image = Image(loc[0], 1, imageCol);
image.add((-0.3, -0.2)--(-0.4,-0.3), imageCol);
image.add((-0.4, -0.2)--(-0.3,-0.3), imageCol);
image.add((-0.1,0.5)--(0.1,0.7), imageCol);
image.add((-0.1,0.7)--(0,0.6), imageCol);
image.add((0.6,0.1)--(0.7,0.1)--(0.6,-0.1)--(0.7,-0.1), imageCol);

Block label1 = Block(loc[1], "Label", blue);
label1.showFrame = false;

Block CNN0 = Block(loc[2], "CNN", black, lightgray);
CNN0.squashE();
Block CNN1 = CNN0.copy(loc[4]);
Block CNN2 = CNN0.copy(loc[9]);
CNN0.subscript("0");
CNN1.subscript("1");
CNN2.subscript("2");


Block3d tensor1 = Block3d(loc[3], 1.5, 0.5, 0.5, black, yellow);
Block3d tensor2 = tensor1.copy(loc[8]);
Block3d tensor3 = Block3d(loc[7], 1.5, 0.1, 0.1, black, yellow);
Block3d tensor4 = Block3d(tensor2.E()+(0.05,0), 0.1, 0.5, 0.5, blue, yellow);
Block3d tensor5 = Block3d(tensor2.E()+(0.15,0), 0.1, 0.5, 0.5, blue, yellow);


Block MLP0 = Block(loc[5], "MLP", black, lightgray);
MLP0.squashE(0.5);
MLP0.scale(0.8, 1.2);
Block MLP1 = MLP0.copy(loc[6]);
Block MLP2 = MLP0.copy(loc[10]);
MLP0.subscript("0");
MLP1.subscript("1");
MLP2.subscript("2");

Block output = label1.copy(loc[11]);
output.scale(0.5,0.3);
output.name = "\large $\binom{\bm{\mu}}{\bm{\sigma}}$";


pair gradHloc = tensor4.S() + (-0.7,-2);
Image gradH = Image(gradHloc, 0.5);
pair gradVloc = tensor5.S() + (0.7,-2);
Image gradV = Image(gradVloc, 0.5);

for(int i=-9; i<10; ++i) {
  real r  = 0.05*(10+i);
  gradH.add((-1,0.1*i)--(1,0.1*i), blue*r+(1.0-r)*white+linewidth(2.5));
  gradV.add((0.1*i,-1)--(0.1*i,1), blue*r+(1.0-r)*white+linewidth(2.5));
}

Point mid = Point((0.5*(CNN1.E().x+ MLP1.W().x), MLP1.W().y));

drawBlocks();

draw(connectEW(image, CNN0), Arrow);
draw(connectEW(CNN0, tensor1), Arrow);
draw(connectEW(tensor1, tensor2), Arrow);
draw(connectEW(tensor1, CNN1), Arrow);
draw(connectEW(label1, MLP0), Arrow);
draw(connectES(CNN1, mid));
draw(connectEN(MLP0, mid));
draw(connectEW(mid, MLP1), Arrow);
draw(connectEW(MLP1, tensor3), Arrow);
draw(connectSN(tensor3, tensor2), Arrow);
draw(connectNS(gradH, tensor4), Arrow);
draw(connectNS(gradV, tensor5), Arrow);
draw(connectEW(tensor5, CNN2), Arrow);
draw(connectEW(CNN2, MLP2), Arrow);
draw(connectEW(MLP2,output), Arrow);


Block attention = superBlock("Attention Block", red+dashed, 0.4, CNN1,
			      MLP0, MLP1);
attention.labelE(0.7);
attention.draw();

Block preprocess = superBlock("Preprocessor", red+dashed, 0.4, CNN0, CNN1);
preprocess.labelN();
preprocess.draw();

Block positioner = superBlock("Positioner", red+dashed, 0.4, CNN2, MLP2, output);
positioner.labelN();
positioner.draw();



label("cat", tensor5.W(), 4SE);
label("\large $\otimes$", tensor2.N(), 4NE);


label("\Large Localiser Architecture", (0.5*(image.W().x+output.E().x), gradV.S().y), 2S);
