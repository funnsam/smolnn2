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
        l1: fcnn::Fcnn::new_xavier_uniform(fastrand::f32),
        l2: activation::Tanh,
        l3: fcnn::Fcnn::new_xavier_uniform(fastrand::f32),
        l4: activation::Softmax,
    };
    let mut o1 = fcnn::make_optimizers!(adam::Adam::new(0.9, 0.999, 0.001));
    let mut o3 = fcnn::make_optimizers!(adam::Adam::new(0.9, 0.999, 0.001));

    let mut c1 = fcnn::FcnnCollector::new();
    let mut c3 = fcnn::FcnnCollector::new();

    loop {
        c1.reset();
        c3.reset();
        let mut err = 0.0;

        for (i, o) in data.iter() {
            let out = model.back_propagate(i, o, loss::categorical_cross_entropy_derivative, &mut c1, &mut (), &mut c3, &mut ());
            err += loss::squared_error(out, o)[0];
        }

        c1 /= data.len() as f32;
        model.l1.update(&c1, &mut o1);
        c3 /= data.len() as f32;
        model.l3.update(&c3, &mut o3);

        println!("error: {}", err / data.len() as f32);
    }
}

fn gen_data(x: f32) -> Vector<2> {
    let cat = (3.0 * x - 1.0).tanh() + 2.0 * (5.0 * x + 2.0).tanh() > 0.0;
    vector!(2 [cat as u8 as f32, !cat as u8 as f32])
    // vector!(1 [3.0 * x])
}
