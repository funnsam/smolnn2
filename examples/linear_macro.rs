use smolmatrix::*;
use smolnn2::*;

model! {
    #[derive(Debug, Clone)]
    pub LinearModel: 1, 1 => 1, 1

    => fcnn::Fcnn<1, 1>
}

fn main() {
    let mut data = Vec::new();
    for i in 0..=100 {
        let x = i as f32 / 10.0 - 5.0;
        data.push((vector!(1 [x]), gen_data(x)));
    }

    let mut model = LinearModel {
        l1: fcnn::Fcnn::new_xavier_uniform(fastrand::f32),
    };
    let mut o1 = fcnn::make_optimizers!(adam::Adam::new(0.9, 0.999, 0.01));

    loop {
        let mut c1 = fcnn::FcnnCollector::new();
        let mut err = 0.0;

        for (i, o) in data.iter() {
            let out = model.back_propagate(i, o, loss::squared_error_derivative, &mut c1);
            err += loss::squared_error(out, o)[0];
        }

        model.l1.update(c1, data.len(), &mut o1);

        println!("error: {}", err / data.len() as f32);
    }
}

fn gen_data(x: f32) -> Vector<1> {
    vector!(1 [3.0 * x])
}
