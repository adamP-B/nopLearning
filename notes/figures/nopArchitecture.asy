import myutil;
import block;
size(700,0);

string output(int ss) {
  string sub = string(ss);
  return "\Large $\binom{\bm{\mu}_{"+sub+"}}{\bm{\sigma}_{"+sub+"}}$";
}

pair[] loc = {(0,0), (3,3), (6,0), (9,0), (15,3), (15,-1.5)};

Block localiser0 = Block(loc[1], "Localiser", black, lightgray);
localiser0.squashE();
Block localiser1 = localiser0.copy(loc[4]);
Block localiser2 = localiser0.copy(loc[5]);
localiser0.subscript("0");
localiser1.subscript("1");
localiser2.subscript("2");


pen imageCol = blue;
Image image = Image((0,0), 1, imageCol);
image.add((-0.3, -0.2)--(-0.4,-0.3), imageCol);
image.add((-0.4, -0.2)--(-0.3,-0.3), imageCol);
image.add((-0.1,0.5)--(0.1,0.7), imageCol);
image.add((-0.1,0.7)--(0,0.6), imageCol);
image.add((0.6,0.1)--(0.7,0.1)--(0.6,-0.1)--(0.7,-0.1), imageCol);

Image image1 = image.copy(loc[2]);
image1.add(box((0.55,-0.15),(0.75,0.15)), red);


draw(connectEW(image,image1), Arrow);

Point mid = Point((0.5*(image.E().x+localiser0.W().x), image.E().y));
Point localiser0inI = Point(localiser0.W(0.3));
Point localiser0inL = Point(localiser0.W(0.7));
Point localiser1inI = Point(localiser1.W(0.3));
Point localiser1inL = Point(localiser1.W(0.7));
Point localiser2inI = Point(localiser2.W(0.3));
Point localiser2inL = Point(localiser2.W(0.7));

Block randIn = Block((image.N().x, localiser0inL.W().y), "$\bm{\eta}\sim\mathcal{N}(\bm{0},\mat{I})$");
randIn.showFrame = false;


draw(connectNW(mid,localiser0inI), Arrow);
draw(connectEW(randIn,localiser0inL), Arrow);
draw(connectSW(mid,localiser2inI), Arrow);


pair out0pos = (image1.N().x, localiser0.E().y);

Block out0 = Block(out0pos, output(0));
out0.showFrame = false;
out0.scale(0.5,0.35);

draw(connectEW(localiser0,out0), Arrow);

Block out1 = out0.copy(localiser1.E()+(1.5,0));
out1.name = output(1);
draw(connectEW(localiser1,out1), Arrow);

Block out2 = out0.copy(localiser2.E()+(1.5,0));
out2.name = output(2);
draw(connectEW(localiser2,out2), Arrow);

pair gridPos = 0.5*(out0.S()+image1.N());
Block gridSampler = Block(gridPos, "Grid Sampler", black, lightgray);
gridSampler.squashN(1.3);
gridSampler.scale(1,0.3);

draw(connectSN(out0, gridSampler), Arrow);
draw(connectSN(gridSampler, image1), Arrow);

Block classifier = Block(loc[3], "Classifier", black, lightgray);
classifier.squashE();
classifier.scale(1,0.7);
draw(connectEW(image1, classifier), Arrow);

Point cp = Point(classifier.E() + (1,1));
draw(connectES(classifier, cp));
draw(connectNW(cp, localiser1inL), Arrow);

Point mid2 = Point(localiser2.W(0.3) - (2,0));
draw(connectNW(mid2, localiser1inI), Arrow);

Block nothing = Block(localiser2inL.W() - 1.0, "$\emptyset$");
nothing.scale(0.3,0.5);
nothing.showFrame = false;
draw(connectEW(nothing, localiser2inL), Arrow);

drawBlocks();

label("$\mathcal{L}_1 = \mathop{\mathrm{KL}}\left(\mathcal{N}(\bm{\mu}_1,\bm{\sigma}^2_1) \| \mathcal{N}(\bm{\mu}_0,\bm{\sigma}^2_0) \right)$", localiser1.S(), 5S);

label("$\mathcal{L}_2 = \mathop{\mathrm{KL}}\left(\mathcal{N}(\bm{\mu}_2,\bm{\sigma}^2_2) \| \mathcal{N}(\bm{\mu}_0,\bm{\sigma}^2_0) \right)$", localiser2.N(), 5N);

label("$\mathcal{L}_0 =\mathcal{L}_1 -\mathcal{L}_2$", (8,3));
label("\Large NoP-Learner Architecture", (0.5*(image.W().x+out2.E().x), localiser2.min().y),2S);

