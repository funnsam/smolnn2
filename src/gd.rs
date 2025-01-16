use crate::*;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GradDesc(pub f32);

impl<D: Dimension> Optimizer<D> for GradDesc {
    fn update(&mut self, w: &mut Tensor<D>, g: Tensor<D>) {
        *w -= &(g * self.0);
    }
}
