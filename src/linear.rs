use crate::*;

#[macro_export]
macro_rules! fcnn_make_optimizers {
    ($opt:expr) => { ($opt, $opt) };
}

pub use fcnn_make_optimizers as make_optimizers;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// A fully connected linear layer
pub struct Linear<const I: usize, const O: usize> where
    [f32; 1 * I * O]: Sized,
    [f32; 1 * 1 * O]: Sized,
{
    pub weight: Matrix<I, O>,
    pub bias: Vector<O>,
}

#[derive(Debug, Clone)]
/// A weight accumulator for `Linear`
pub struct LinearCollector<const I: usize, const O: usize> where
    [f32; 1 * I * O]: Sized,
    [f32; 1 * 1 * O]: Sized,
{
    pub weight: Matrix<I, O>,
    pub bias: Vector<O>,
}

impl<const I: usize, const O: usize> Linear<I, O> where
    [f32; 1 * I * O]: Sized,
    [f32; 1 * 1 * O]: Sized,
{
    pub fn new<F: Fn() -> f32>(f: F) -> Self {
        Self {
            weight: Matrix::from_iter(core::iter::repeat_with(&f)),
            bias: Matrix::from_iter(core::iter::repeat_with(&f)),
        }
    }

    /// Get uniform distributed random `f32`s in the range 0..1 and return a fully connected neural
    /// network layer with Xavier uniform initialization
    pub fn new_xavier_uniform<F: Fn() -> f32>(f: F) -> Self {
        let r = (6.0_f32 / (I + O) as f32).sqrt();

        Self {
            weight: Matrix::from_iter(core::iter::repeat_with(|| f() * 2.0 * r - r)),
            bias: Matrix::from_iter(core::iter::repeat_with(|| f() * 2.0 * r - r)),
        }
    }

    /// Get uniform distributed random `f32`s in the range 0..1 and return a fully connected neural
    /// network layer with He uniform initialization
    pub fn new_he_uniform<F: Fn() -> f32>(f: F) -> Self {
        let r = (3.0_f32 / I as f32).sqrt();

        Self {
            weight: Matrix::from_iter(core::iter::repeat_with(|| f() * 2.0 * r - r)),
            bias: Matrix::from_iter(core::iter::repeat_with(|| f() * 2.0 * r - r)),
        }
    }

    pub fn update<
        W: Optimizer<Dim2<I, O>>,
        B: Optimizer<Dim2<1, O>>,
    >(&mut self, c: &LinearCollector<I, O>, opts: &mut (W, B)) {
        opts.0.update(&mut self.weight, c.weight.clone());
        opts.1.update(&mut self.bias  , c.bias  .clone());
    }
}

impl<const I: usize, const O: usize> Collectable for Linear<I, O> where
    [f32; 1 * I * O]: Sized,
    [f32; 1 * 1 * O]: Sized,
{
    type Collector = LinearCollector<I, O>;
}

impl<const I: usize, const O: usize> Layer<Dim2<1, I>, Dim2<1, O>> for Linear<I, O> where
    [f32; 1 * I * O]: Sized,
    [f32; 1 * 1 * O]: Sized,
{
    fn forward(&self, input: &Vector<I>) -> Vector<O> where
        [f32; 1 * 1 * I]: Sized,
    {
        &self.weight * input + &self.bias
    }

    fn derivative(
        &self,
        dc: &Vector<O>,
        _input: &Vector<I>,
    ) -> Vector<I> where
        [f32; 1 * O * I]: Sized,
        [f32; 1 * 1 * I]: Sized,
    {
        &self.weight.transpose() * dc
    }

    fn back_propagate(
        &self,
        collector: &mut Self::Collector,
        last_derivative: Vector<O>,
        input: &Vector<I>,
    ) where
        [f32; 1 * 1 * I]: Sized,
        [f32; 1 * I * 1]: Sized,
    {
        collector.bias += &last_derivative;
        collector.weight += &(&last_derivative * &input.transpose());
    }
}

impl<const I: usize, const O: usize> LinearCollector<I, O> where
    [f32; 1 * I * O]: Sized,
    [f32; 1 * 1 * O]: Sized,
{
    pub fn new() -> Self {
        Self {
            weight: Matrix::new_filled(0.0),
            bias: Matrix::new_filled(0.0),
        }
    }

    pub fn reset(&mut self) {
        self.weight.fill(0.0);
        self.bias.fill(0.0);
    }
}

impl<const I: usize, const O: usize> core::ops::DivAssign<f32> for LinearCollector<I, O> where
    [f32; 1 * I * O]: Sized,
    [f32; 1 * 1 * O]: Sized,
{
    fn div_assign(&mut self, rhs: f32) {
        self.weight /= rhs;
        self.bias /= rhs;
    }
}
