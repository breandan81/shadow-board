// test_lshape.scad
// Ground truth: L-shape with 80×80mm bounding box, 30mm arm width
// Validates: concave corner handling (the inner corner is the hard case)
//
// Geometry:
//   full bbox  80 × 80
//   horizontal arm: 80 × 30  (bottom)
//   vertical arm:   30 × 80  (left)
//   cutout:         50 × 50  (top-right)

bbox_x    = 80;
bbox_y    = 80;
arm_w     = 30;
height    = 8;

union() {
    cube([bbox_x, arm_w, height]);           // horizontal arm
    cube([arm_w,  bbox_y, height]);          // vertical arm
}
