mod factorial_design;
mod random_design;
mod response_surface_design;
use ndarray::array;

fn main() {
    let array_1 = array![1, 2, 3];
    let array_2 = array![-3, 4, 5];
    let array_3 = array_1 * array_2;

    println!("{}", array_3);
}
