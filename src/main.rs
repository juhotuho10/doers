mod factorial_design;
mod random_design;
mod response_surface_design;
use ndarray::arr1;

fn main() {
    let array_1 = arr1(&[1, 2, 3]);
    let array_2 = arr1(&[-3, 4, 5]);

    let array_3 = array_1 * array_2;

    println!("{}", array_3);
}
