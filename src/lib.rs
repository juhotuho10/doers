// Lint groups to warn about
#![warn(
    clippy::complexity,
    clippy::correctness,
    clippy::nursery,
    clippy::perf,
    clippy::style,
    clippy::suspicious,
    clippy::pedantic,
    clippy::cargo
)]
// warn about individual lints
#![warn(
    clippy::suboptimal_flops,
    clippy::imprecise_flops,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::manual_clamp,
    clippy::modulo_arithmetic,
    clippy::approx_constant,
    clippy::integer_division,
    clippy::float_cmp
)]
// allowing individual pedantic lints
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::must_use_candidate,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::needless_pass_by_value
)]
pub mod factorial_design;
pub mod random_design;
pub mod response_surface_design;
