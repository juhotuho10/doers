use ndarray::Array2;

mod factorial_design;
mod random_design;
mod response_surface_design;

fn main() {
    let mut array_1 = vec![Array2::from_elem((1, 1), 1)];

    for _ in 0..1000 {
        array_1 = factorial_design::gsd(&[5, 5, 5, 5, 5], 5, 3).expect("")
    }

    println!("{:?}", array_1);
}
