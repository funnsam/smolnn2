use crate::*;

macro_rules! activation {
    ($name:tt $fmla:expr, $der:expr) => {
        impl Collectable for $name {
            type Collector = ();
        }

        impl<const W: usize, const H: usize> Layer<W, H, W, H> for $name {
            fn forward(&self, input: &Matrix<W, H>) -> Matrix<W, H> {
                input.clone().map_each(|i| *i = $fmla(self, *i))
            }

            fn derivative(
                &self,
                dc: &Matrix<W, H>,
                input: &Matrix<W, H>,
            ) -> Matrix<W, H> {
                dc.clone().map_zip_ref(input, |(i, input)| *i *= $der(self, *input))
            }

            fn back_propagate(
                &self,
                _collector: &mut Self::Collector,
                _derivative: Matrix<W, H>,
                _input: &Matrix<W, H>,
            ) {}
        }
    };
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClippedRelu;
activation!(ClippedRelu
    |_, i: f32| i.min(1.0).max(0.0),
    |_, i: f32| if 0.0 < i && i < 1.0 { 1.0 } else { 0.0 }
);

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LeakyRelu(pub f32);
activation!(LeakyRelu
    |s: &Self, i: f32| i.max(s.0 * i),
    |s: &Self, i: f32| if 0.0 <= i { 1.0 } else { s.0 }
);

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tanh;
activation!(Tanh
    |_, i: f32| i.tanh(),
    |_, i: f32| i.cosh().powi(2).recip()
);

/// Performs the softmax algorithm on it's inputs. When training, you **must** use the Categorical
/// Cross Entropy loss function instead.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Softmax;

impl Collectable for Softmax {
    type Collector = ();
    }

impl<const W: usize, const H: usize> Layer<W, H, W, H> for Softmax {
    fn forward(&self, input: &Matrix<W, H>) -> Matrix<W, H> {
        let mut input = input.clone();
        let max = input.inner.iter().flatten().fold(f32::NEG_INFINITY, |c, i| c.max(*i));
        input.map_each_in_place(|i| *i = (*i - max).exp());
        let sum = input.inner.iter().flatten().sum::<f32>();
        input / sum
    }

    fn derivative(
        &self,
        dc: &Matrix<W, H>,
        _input: &Matrix<W, H>,
    ) -> Matrix<W, H> {
        dc.clone()
    }

    fn back_propagate(
        &self,
        _collector: &mut Self::Collector,
        _derivative: Matrix<W, H>,
        _input: &Matrix<W, H>,
    ) {}
}
