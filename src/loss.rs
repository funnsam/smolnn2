use crate::*;

/// Computes the squared error (`(result - expected)²`)
pub fn squared_error<D: Dimension>(
    result: Tensor<D>,
    expected: &Tensor<D>,
) -> Tensor<D> {
    (result - expected).map_each(|i| i * i)
}

pub fn squared_error_derivative<D: Dimension>(
    result: Tensor<D>,
    expected: &Tensor<D>,
) -> Tensor<D> {
    (result - expected) * 2.0
}

/// Computes the categorical cross entropy (`-ln(result) × expected`)
pub fn categorical_cross_entropy<D: Dimension>(
    result: Tensor<D>,
    expected: &Tensor<D>,
) -> Tensor<D> {
    result.map_zip_ref(expected, |r, e| -(r.ln() * e))
}

pub fn categorical_cross_entropy_derivative<D: Dimension>(
    result: Tensor<D>,
    expected: &Tensor<D>,
) -> Tensor<D> {
    result - expected
}
