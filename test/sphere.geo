ro = 20;		// outer radius
ri = 1;			// inner radius
reso = 6.0;		// outer resolution
resi = 0.2;		// inner resolution

// Center
Point(1) = {0, 0, 0, reso};

// Outer sphere
Point(2) = {ro, 0, 0, reso};
Point(3) = {0, ro, 0, reso};
Point(4) = {0, 0, ro, reso};
Point(5) = {-ro, 0, 0, reso};
Point(6) = {0, -ro, 0, reso};
Point(7) = {0, 0, -ro, reso};

// Inner sphere
Point(8) = {ri, 0, 0, resi};
Point(9) = {0, ri, 0, resi};
Point(10) = {0, 0, ri, resi};
Point(11) = {-ri, 0, 0, resi};
Point(12) = {0, -ri, 0, resi};
Point(13) = {0, 0, -ri, resi};

// Outer sphere
Circle(1) = {3, 1, 2};
Circle(2) = {2, 1, 6};
Circle(3) = {6, 1, 5};
Circle(4) = {5, 1, 3};
Circle(5) = {3, 1, 7};
Circle(6) = {7, 1, 6};
Circle(7) = {6, 1, 4};
Circle(8) = {4, 1, 3};
Circle(9) = {2, 1, 4};
Circle(10) = {4, 1, 5};
Circle(11) = {5, 1, 7};
Circle(12) = {7, 1, 2};

// Inner sphere
Circle(13) = {12, 1, 10};
Circle(14) = {10, 1, 9};
Circle(15) = {9, 1, 13};
Circle(16) = {13, 1, 12};
Circle(17) = {10, 1, 8};
Circle(18) = {8, 1, 13};
Circle(19) = {13, 1, 11};
Circle(20) = {11, 1, 10};
Circle(21) = {9, 1, 8};
Circle(22) = {8, 1, 12};
Circle(23) = {12, 1, 11};
Circle(24) = {11, 1, 9};

// Outer sphere
Line Loop(25) = {8, 1, 9};
Ruled Surface(26) = {25};
Line Loop(27) = {1, -12, -5};
Ruled Surface(28) = {27};
Line Loop(29) = {11, -5, -4};
Ruled Surface(30) = {29};
Line Loop(31) = {4, -8, 10};
Ruled Surface(32) = {31};
Line Loop(34) = {10, -3, 7};
Ruled Surface(35) = {34};
Line Loop(36) = {7, -9, 2};
Ruled Surface(37) = {36};
Line Loop(38) = {2, -6, 12};
Ruled Surface(39) = {38};
Line Loop(40) = {11, 6, 3};
Ruled Surface(41) = {40};

// Inner sphere
Line Loop(42) = {20, 14, -24};
Ruled Surface(43) = {42};
Line Loop(44) = {14, 21, -17};
Ruled Surface(45) = {44};
Line Loop(46) = {21, 18, -15};
Ruled Surface(47) = {46};
Line Loop(48) = {15, 19, 24};
Ruled Surface(49) = {48};
Line Loop(50) = {19, -23, -16};
Ruled Surface(51) = {50};
Line Loop(52) = {16, -22, 18};
Ruled Surface(53) = {52};
Line Loop(54) = {22, 13, 17};
Ruled Surface(55) = {54};
Line Loop(56) = {13, -20, -23};
Ruled Surface(57) = {56};

// Outer sphere
Physical Surface(58) = {32, 26, 28, 30, 41, 35, 37, 39};

// Inner sphere
Physical Surface(59) = {45, 47, 49, 43, 51, 55, 53, 57};

// Interior Volume
Surface Loop(60) = {45, 43, 57, 55, 53, 51, 49, 47};
Surface Loop(61) = {32, 30, 41, 39, 37, 35, 26, 28};
Volume(62) = {60, 61};
Physical Volume(63) = {62};
