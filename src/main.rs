mod factorial_design;
mod random_design;
mod response_surface_design;
use ndarray::{s, Array, Array1, Axis};

fn sort_ndarray_array1(array: Array1<f32>) -> Array1<f32> {
    // Step 1: Convert to Vec<f32>
    let mut vec = array.to_vec();

    // Step 2: Sort the Vec<f32>. Here we use sort_unstable_by because f32 doesn't implement Ord
    // This also ignores NaN values or you might customize sorting logic to handle them differently
    vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Step 3: Convert the sorted Vec<f32> back to an ndarray::Array1<f32>
    Array1::from(vec)
}
fn main() {
    let samples = 6;
    let arr = random_design::lhs_centered(4, samples, 10);
    let mut cut: Array1<f64> = Array::linspace(0., 1., samples + 1);
    cut = cut.mapv(|x: f64| (x * 100.).round() / 100.); // rounding the variables to 0.01

    let a = cut.slice(s![..samples]);
    let b = cut.slice(s![1..samples + 1]);
    let mut center = ((&a + &b) / 2.).mapv(|x| x as f32);
    center = sort_ndarray_array1(center);

    for col in arr.axis_iter(Axis(1)) {
        assert_eq!(center, sort_ndarray_array1(col.to_owned()))
    }
}
