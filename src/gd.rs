use crate::*;

pub struct GradDesc(pub f32);

impl<const W: usize, const H: usize> Optimizer<W, H> for GradDesc {
    fn update(&mut self, w: &mut Matrix<W, H>, g: Matrix<W, H>) {
        *w -= &(g * self.0);
    }
}
