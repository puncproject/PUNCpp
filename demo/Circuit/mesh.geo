sizeo = .3;
a = 6;
b = 3;
Point(1) = {-a, -b, 0, sizeo};
Point(2) = {a, -b, 0, sizeo};
Point(3) = {a, b, 0, sizeo};
Point(4) = {-a, b, 0, sizeo};

r = 1.0;
sizei = 0.1;
Point(5) = {-b, 0, 0, 1};
Point(6) = {-b-r, 0, 0, sizei};
Point(7) = {-b, -r, 0, sizei};
Point(8) = {-b+r, 0, 0, sizei};
Point(9) = {-b, r, 0, sizei};

Point(10) = {b, 0, 0, sizei};
Point(11) = {b-r, 0, 0, sizei};
Point(12) = {b, -r, 0, sizei};
Point(13) = {b+r, 0, 0, sizei};
Point(14) = {b, r, 0, sizei};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};


Circle(5) = {7, 5, 6};
Circle(6) = {6, 5, 9};
Circle(7) = {9, 5, 8};
Circle(8) = {8, 5, 7};


Circle(9) = {12, 10, 11};
Circle(10) = {11, 10, 14};
Circle(11) = {14, 10, 13};
Circle(12) = {13, 10, 12};


Physical Line(1) = {4, 1, 2, 3};

Physical Line(2) = {7, 6, 5, 8};

Physical Line(3) = {10, 9, 12, 11};


Line Loop(1) = {8, 5, 6, 7};
Line Loop(2) = {9, 10, 11, 12};
Line Loop(3) = {1, 2, 3, 4};

Plane Surface(1) = {1, 2, 3};

Physical Surface(4) = {1};
