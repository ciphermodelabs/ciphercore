use num_traits::AsPrimitive;

pub fn pearson_correlation<T: AsPrimitive<f64> + std::ops::Mul<Output = T>>(
    predicted: &[T],
    labels: &[T],
) -> f64 {
    let mut exy = 0.0;
    let mut ex = 0.0;
    let mut ey = 0.0;
    let mut ex2 = 0.0;
    let mut ey2 = 0.0;
    let mut den = 0.0;
    for (x, y) in predicted.iter().cloned().zip(labels.iter().cloned()) {
        exy += (x * y).as_();
        ex += x.as_();
        ey += y.as_();
        ex2 += (x * x).as_();
        ey2 += (y * y).as_();
        den += 1.0;
    }
    exy /= den;
    ex /= den;
    ey /= den;
    ex2 /= den;
    ey2 /= den;

    (exy - ex * ey) / ((ex2 - ex * ex) * (ey2 - ey * ey)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_pearson_correlation() {
        let predicted = vec![1i64, 2, 3, 4, 5];
        let labels = vec![1, 2, 3, 4, 5];
        let corr = pearson_correlation(&predicted, &labels);
        assert_eq!(corr, 1.0);
    }
}
