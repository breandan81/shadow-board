// test_disk.scad
// Ground truth: 80mm diameter disk
// Validates: circular edge accuracy and radial scale uniformity

diameter = 80;
height   = 8;

cylinder(h=height, d=diameter, $fn=128);
