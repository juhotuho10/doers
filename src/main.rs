mod factorial_design;
mod random_design;
mod response_surface_design;

use crate::response_surface_design::{Alpha, Face};

fn main() {
    let array_1 =
        response_surface_design::ccdesign(3, &[3, 3, 3], Alpha::Orthogonal, Face::Circumscribed);

    println!("{:?}", array_1);
}
