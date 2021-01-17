import myutil;

size(1000,0);

void drawImage(pair p) {
  path f = (-1,-2)--(-1,2)--(1,1.5)--(1,-1.5)--cycle;
  draw(shift(p)*f);
  draw(p+(-0.5,-1)--p+(0,0.25), linewidth(3)+gray);
  draw(p+(0,-0.8)--p+(-0.4,0.3), linewidth(3)+gray);
  draw(p+(0.4,0.5)--p+(0.8,1.4), linewidth(5)+gray);
  draw(p+(0.4,1.5)--p+(0.6,1), linewidth(5)+gray);
  draw(shift(p)*((0.5,-0.8)--(0.9,-0.8)--(0.6,-1.2)--(0.9,-1.1)), linewidth(2)+gray);
}

void drawbb(pair p) {
  path bb = (-0.5,-1)--(0,-0.8)--(0,0.25)--(-0.4,0.3)--cycle;
  draw(shift(p)*bb, dashed);
}


void drawNetwork(pair p, string s1="", string s2="") {
  path f = box((-1,-1),(1,1));
  draw(shift(p)*f, linewidth(2));
  label(s1, p, N);
  label(s2, p, S);
}

void myArrow(pair p1, pair p2, real off1=0, real off2=0) {
  draw(p1+(off1,0)--p2+(off2,0), linewidth(3), Arrow(10));
}

path bb=box((-1.5,-6),(17,6));

draw(bb, pink);

pair[] pp = {(0,0), (3,0), (6,0), (9,0), (11,0), (9,4), (9,-4), (13,4), (13,0), (13,-4), (15,4), (15,0), (15,-4)};


drawImage(pp[0]);
label("\large Image from COCO", pp[0]+(0,-2), S, heavygreen);


label("\large (or scattered MNST)", pp[0]+(0,-2.3), S, heavygreen);
ship();
label("\large $\mathcal{N}(0,1)$", (1.5,2),N);
myArrow((1.5,2),(1.5,0));

myArrow(pp[0],pp[1],0,-1);
drawNetwork(pp[1], "Spatial", "Attention");
label("$(\bm{\mu},\bm{\sigma}^2)$", pp[1]+(1,0), NE, red);


myArrow(pp[1], pp[2], 1);
ship();

drawImage(pp[2]);
drawbb(pp[2]);
draw(shift(pp[2]+(-0.21,-0.27))*scale(0.2,0.54)*circle(0, 1), red);
label("\large $p=\mathcal{N}\!\left(\bm{x} \mid \bm{\mu}, \mathrm{diag}(\bm{\sigma}^2)\right)$", pp[2]-(0,2), S, red);
ship();

myArrow(pp[2],pp[3],0,-1);
drawNetwork(pp[3], "Gumbel Softmax", "Classifier");
myArrow(pp[3],pp[4],1,-0.2);
label("\large $\bm{y}$", pp[4]);
label("\large ``label''", pp[4]-(0,0.5), heavygreen);
ship();


drawImage(pp[5]);
drawImage(pp[6]);

myArrow(pp[5],pp[7], 1, -1);
myArrow(pp[4],pp[8], 0.2, -1);
myArrow(pp[6],pp[9], 1, -1);
myArrow(pp[4]+(0,0.3), pp[4]+(0,4));

drawNetwork(pp[7], "location", "module 1");
drawNetwork(pp[8], "location", "module 2");
drawNetwork(pp[9], "location", "module 3");
label("$(\bm{\mu}_1,\bm{\sigma}_1^2)$", pp[7]+(1,0), 2NE, red);
label("$(\bm{\mu}_2,\bm{\sigma}_2^2)$", pp[8]+(1,0), 2NE, red);
label("$(\bm{\mu}_3,\bm{\sigma}_3^2)$", pp[9]+(1,0), 2NE, red);

label("\large $q_1(\bm{x}| \bm{\mu}_1, \bm{\sigma}_1^2)$", pp[10], E);
label("\large $q_2(\bm{x}| \bm{\mu}_2, \bm{\sigma}_2^2)$", pp[11], E);
label("\large $q_3(\bm{x}| \bm{\mu}_3, \bm{\sigma}_3^2)$", pp[12], E);

myArrow(pp[7], pp[10], 1);
myArrow(pp[8], pp[11], 1);
myArrow(pp[9], pp[12], 1);

ship();

label("\Large $\mathcal{L}_1 = \mathrm{KL}(q_1,p)$", pp[7]+(0,1.5), blue);
label("\Large $\mathcal{L}_2 = \mathrm{KL}(q_2,p)$", pp[8]+(0,1.5), blue);
label("\Large $\mathcal{L}_3 = \mathrm{KL}(q_3,p)$", pp[9]+(0,1.5), blue);

ship();

label("\Large $\mathcal{L} = \mathcal{L}_1 - \mathcal{L}_2 - \mathcal{L}_3$", pp[1]+(0,1.5), blue);
ship();
