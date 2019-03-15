ri = 0.001;   // Inner radius
ro = 0.1;     // Outer radius
resi = ri/5;  // Inner resolution
reso = 0.006; // Outer resolution

Point(1) = {  0,   0, 0, resi};
Point(2) = {+ro,   0, 0, reso};
Point(3) = {-ro,   0, 0, reso};
Point(4) = {  0, +ro, 0, reso};
Point(5) = {  0, -ro, 0, reso};
Point(6) = {+ri,   0, 0, resi};
Point(7) = {-ri,   0, 0, resi};
Point(8) = {  0, +ri, 0, resi};
Point(9) = {  0, -ri, 0, resi};

Circle(1) = {2, 1, 4};
Circle(2) = {4, 1, 3};
Circle(3) = {3, 1, 5};
Circle(4) = {5, 1, 2};
Circle(5) = {6, 1, 8};
Circle(6) = {8, 1, 7};
Circle(7) = {7, 1, 9};
Circle(8) = {9, 1, 6};

Physical Line(1) = {2, 1, 4, 3}; // Exterior
Physical Line(2) = {5, 6, 7, 8}; // Interior

// Optional
Line Loop(1) = {2, 3, 4, 1};
Line Loop(2) = {5, 6, 7, 8};
Plane Surface(1) = {1, 2};
Physical Surface(3) = {1};
