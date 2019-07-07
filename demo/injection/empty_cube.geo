l = 2;			// length of cube
reso = 0.2;		// outer resolution

// Outer boundary (box)
Point(1) = {0,0,0, reso};
Point(2) = {0,0,l, reso};
Point(3) = {0,l,0, reso};
Point(4) = {0,l,l, reso};
Point(5) = {l,0,0, reso};
Point(6) = {l,0,l, reso};
Point(7) = {l,l,0, reso};
Point(8) = {l,l,l, reso};


// Outer boundary
Line(1) = {2, 6};
Line(2) = {6, 8};
Line(3) = {8, 4};
Line(4) = {4, 2};
Line(5) = {2, 1};
Line(6) = {1, 3};
Line(7) = {3, 4};
Line(8) = {3, 7};
Line(9) = {7, 8};
Line(10) = {7, 5};
Line(11) = {5, 6};
Line(12) = {5, 1};

// Outer boundary
Line Loop(25) = {4, 1, 2, 3};
Plane Surface(26) = {25};
Line Loop(27) = {2, -9, 10, 11};
Plane Surface(28) = {27};
Line Loop(29) = {12, 6, 8, 10};
Plane Surface(30) = {29};
Line Loop(31) = {6, 7, 4, 5};
Plane Surface(32) = {31};
Line Loop(33) = {5, -12, 11, -1};
Plane Surface(34) = {33};
Line Loop(35) = {9, 3, -7, 8};
Plane Surface(36) = {35};


// Outer boundary
Physical Surface(53) = {36, 32, 28, 34, 30, 26};


// Outer boundary
Surface Loop(55) = {32, 30, 34, 28, 26, 36};

// Volume
Volume(57) = {55};
Physical Volume(58) = {57};
