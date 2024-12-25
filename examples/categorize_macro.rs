use smolmatrix::*;
use smolnn2::*;

model! {
    #[derive(Debug, Clone)]
    pub CategorizeModel: 1, 1 => 1, 2

    => fcnn::Fcnn<1, 2>
    => activation::Tanh
    => fcnn::Fcnn<2, 2>
    => activation::Softmax
}

fn main() {
    let mut data = Vec::new();
    for i in 0..=100 {
        let x = i as f32 / 10.0 - 5.0;
        data.push((vector!(1 [x]), gen_data(x)));
    }

    let mut model = CategorizeModel {
        l1: fcnn::Fcnn::new(|| fastrand::f32() * 0.5),
        l2: activation::Tanh,
        l3: fcnn::Fcnn::new(|| fastrand::f32() * 0.5),
        l4: activation::Softmax,
    };
    let mut opt_l1w = adam::Adam::new(0.9, 0.999, 1.0);
    let mut opt_l1b = adam::Adam::new(0.9, 0.999, 1.0);
    let mut opt_l3w = adam::Adam::new(0.9, 0.999, 1.0);
    let mut opt_l3b = adam::Adam::new(0.9, 0.999, 1.0);

    loop {
        let mut c1 = fcnn::FcnnCollector::new();
        let mut c3 = fcnn::FcnnCollector::new();
        let mut err = 0.0;

        for (i, o) in data.iter() {
            let out = model.back_propagate(i, o, loss::categorical_cross_entropy_derivative, &mut c1, &mut (), &mut c3, &mut ());
            err += loss::squared_error(out, o)[0];
        }

        opt_l1w.update(&mut model.l1.weight, c1.weight / data.len() as f32);
        opt_l1b.update(&mut model.l1.bias  , c1.bias   / data.len() as f32);
        opt_l3w.update(&mut model.l3.weight, c3.weight / data.len() as f32);
        opt_l3b.update(&mut model.l3.bias  , c3.bias   / data.len() as f32);

        println!("error: {}", err / data.len() as f32);
    }
}

fn gen_data(x: f32) -> Vector<2> {
    let cat = (3.0 * x - 1.0).tanh() + 2.0 * (5.0 * x + 2.0).tanh() > 0.0;
    vector!(2 [cat as u8 as f32, !cat as u8 as f32])
    // vector!(1 [3.0 * x])
}
