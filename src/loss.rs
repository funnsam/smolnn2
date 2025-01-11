use crate::*;

/// Computes the squared error (`(result - expected)²`)
pub fn squared_error<const W: usize, const H: usize>(
    result: Matrix<W, H>,
    expected: &Matrix<W, H>,
) -> Matrix<W, H> {
    (result - expected).map_each(|i| *i *= *i)
}

pub fn squared_error_derivative<const W: usize, const H: usize>(
    result: Matrix<W, H>,
    expected: &Matrix<W, H>,
) -> Matrix<W, H> {
    (result - expected) * 2.0
}

/// Computes the categorical cross entropy (`-ln(result) × expected`)
pub fn categorical_cross_entropy<const W: usize, const H: usize>(
    result: Matrix<W, H>,
    expected: &Matrix<W, H>,
) -> Matrix<W, H> {
    result.map_zip_ref(expected, |(r, e)| *r = -(r.ln() * *e))
}

pub fn categorical_cross_entropy_derivative<const W: usize, const H: usize>(
    result: Matrix<W, H>,
    expected: &Matrix<W, H>,
) -> Matrix<W, H> {
    result - expected
}
