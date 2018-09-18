ri = 1.0; 
ro = 30.0*ri;
reso = 1.0;
resi = 0.2;

Point(1) = {  0,   0, 0, resi};

// Interior Boundary
Point(2) = {+ri,   0, 0, resi};
Point(3) = {-ri,   0, 0, resi};
Point(4) = {  0, +ri, 0, resi};
Point(5) = {  0, -ri, 0, resi};
Circle(1) = {4, 1, 3};
Circle(2) = {3, 1, 5};
Circle(3) = {5, 1, 2};
Circle(4) = {2, 1, 4};
Line Loop(10) = {1, 2, 3, 4};
Physical Line(13) = {1, 2, 3, 4};

// Exterior Boundary
Point(6) = {+ro,   0, 0, reso};
Point(7) = {-ro,   0, 0, reso};
Point(8) = {  0, +ro, 0, reso};
Point(9) = {  0, -ro, 0, reso};
Circle(5) = {8, 1, 7};
Circle(6) = {7, 1, 9};
Circle(7) = {9, 1, 6};
Circle(8) = {6, 1, 8};
Line Loop(9) = {5, 6, 7, 8};
Physical Line(12) = {5, 6, 7, 8};

// Domain
Plane Surface(11) = {9, 10};
Physical Surface(14) = {11};
