#![warn(
    clippy::complexity,
    clippy::correctness,
    clippy::nursery,
    clippy::perf,
    clippy::style,
    clippy::suspicious,
    clippy::pedantic
)]
// allowing individual pedantic lints
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::must_use_candidate,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::unreadable_literal
)]
pub mod factorial_design;
pub mod random_design;
pub mod response_surface_design;
