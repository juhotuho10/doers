mod factorial_design;
mod random_design;
mod response_surface_design;
use doers::factorial_design::gsd;
use std::vec;

fn main() {
    let levels = vec![3, 3, 3];
    let reductions = 8;
    let n_arrays = 1;
    let example_array = gsd(levels, reductions, n_arrays);

    println!("{:?}", example_array);
}
