l = 1;
res = l/10.;

// Exterior Boundary
Point(1) = {0, 0, 0, res};
Point(2) = {0, l, 0, res};
Point(3) = {l, l, 0, res};
Point(4) = {l, 0, 0, res};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Physical Line(5) = {1, 2, 3, 4};
Line Loop(6) = {1, 2, 3, 4};
Plane Surface(7) = {6};
Physical Surface(8) = {7};
