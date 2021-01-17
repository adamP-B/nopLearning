import myutil;
size(300,0);

pair inf = (2,0.5);

pair threeD(pair x) {
  return ((0,x.y)-inf)/(1+0.3*x.x);
}

guide project(pair[] p) {
  guide g;
  for(int i=0; i<p.length; ++i)
    g = g--threeD(p[i]);
  return g--cycle;
}

picture image = new picture;
size(image, 100);

pair[] imagebox = {(0,0), (1,0), (1,1), (0,1)};
draw(image, project(imagebox));
draw(image, threeD((0.3,0.2))--threeD((0.4,0.3)));
draw(image, threeD((0.4,0.2))--threeD((0.3,0.3)));
add(image);


pair[] selection = {(0.28,0.18), (0.42,0.18), (0.42,0.32), (0.28,0.32)};

draw(image, project(selection), red+linewidth(1));


