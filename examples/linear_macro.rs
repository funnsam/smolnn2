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
        l1: fcnn::Fcnn::new(|| fastrand::f32() * 0.5),
    };
    let mut opt_l1w = adam::Adam::new(0.9, 0.999, 1.0);
    let mut opt_l1b = adam::Adam::new(0.9, 0.999, 1.0);

    loop {
        let mut c1 = fcnn::FcnnCollector::new();
        let mut err = 0.0;

        for (i, o) in data.iter() {
            let out = model.back_propagate(i, o, &mut c1);
            err += loss::squared_error(out, o)[0];
        }

        opt_l1w.update(&mut model.l1.weight, c1.weight / data.len() as f32);
        opt_l1b.update(&mut model.l1.bias  , c1.bias   / data.len() as f32);

        println!("error: {}", err / data.len() as f32);
    }
}

fn gen_data(x: f32) -> Vector<1> {
    vector!(1 [3.0 * x])
}
