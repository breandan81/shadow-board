// test_ring.scad
// Ground truth: annular ring, outer diameter 80mm, inner diameter 40mm
// Validates: hole/interior contour handling (two closed paths in the SVG)

outer_d = 80;
inner_d = 40;
height  = 8;

difference() {
    cylinder(h=height, d=outer_d, $fn=128);
    cylinder(h=height, d=inner_d, $fn=128);
}
