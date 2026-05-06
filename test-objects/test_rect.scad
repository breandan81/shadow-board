// test_rect.scad
// Ground truth: 100mm × 60mm rectangle, sharp corners
// Validates: scale accuracy on straight edges

width  = 100;
depth  = 60;
height = 8;

cube([width, depth, height]);
