use smolmatrix::*;
use smolnn2::*;

model! {
    #[derive(Debug, Clone)]
    pub SimpleModel: 1, 1 => 1, 1

    => fcnn::Fcnn<1, 2>
    => activation::Tanh
    => fcnn::Fcnn<2, 1>
}

fn main() {
    let mut data = Vec::new();
    for i in 0..=100 {
        let x = i as f32 / 50.0 - 1.0;
        data.push((vector!(1 [x]), gen_data(x)));
    }

    let mut model = SimpleModel {
        l1: fcnn::Fcnn::new_xavier_uniform(fastrand::f32),
        l2: activation::Tanh,
        l3: fcnn::Fcnn::new_xavier_uniform(fastrand::f32),
    };
    // model.l1.weight = matrix!(1 x 2 [3.0] [5.0]);
    // model.l1.bias = matrix!(1 x 2 [-1.0] [2.0]);
    // model.l3.weight = matrix!(2 x 1 [1.0, 2.0]);
    // model.l3.bias = matrix!(1 x 1 [0.0]);

    let mut o1 = fcnn::make_optimizers!(adam::Adam::new(0.9, 0.999, 0.01));
    let mut o3 = fcnn::make_optimizers!(adam::Adam::new(0.9, 0.999, 0.01));

    loop {
        let mut c1 = fcnn::FcnnCollector::new();
        let mut c3 = fcnn::FcnnCollector::new();
        let mut err = 0.0;

        for (i, o) in data.iter() {
            let out = model.back_propagate(i, o, loss::squared_error_derivative, &mut c1, &mut (), &mut c3);
            err += loss::squared_error(out, o)[0];
        }

        model.l1.update(c1, data.len(), &mut o1);
        model.l3.update(c3, data.len(), &mut o3);

        println!("error: {}", err / data.len() as f32);
    }
}

fn gen_data(x: f32) -> Vector<1> {
    vector!(1 [(3.0 * x - 1.0).tanh() + 2.0 * (5.0 * x + 2.0).tanh()])
}
