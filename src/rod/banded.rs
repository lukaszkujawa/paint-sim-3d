pub struct Banded {
    pub n: usize,
    pub b: usize,
    pub a: Vec<f64>,
}

impl Banded {
    pub fn new(n: usize, b: usize) -> Self {
        Self { n, b, a: vec![0.0; n * (b + 1)] }
    }

    pub fn from_scratch(mut a: Vec<f64>, n: usize, b: usize) -> Self {
        a.clear();
        a.resize(n * (b + 1), 0.0);
        Self { n, b, a }
    }

    pub fn into_scratch(self) -> Vec<f64> {
        self.a
    }

    #[inline]
    fn at(&self, i: usize, j: usize) -> usize {
        i * (self.b + 1) + (self.b + j - i)
    }

    #[inline]
    pub fn add(&mut self, i: usize, j: usize, v: f64) {
        if j > i {
            return;
        }
        if i - j > self.b {
            return;
        }
        let k = self.at(i, j);
        self.a[k] += v;
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let (i, j) = if j > i { (j, i) } else { (i, j) };
        if i - j > self.b {
            return 0.0;
        }
        self.a[self.at(i, j)]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        let (i, j) = if j > i { (j, i) } else { (i, j) };
        if i - j > self.b {
            return;
        }
        let k = self.at(i, j);
        self.a[k] = v;
    }

    pub fn fix_dof(&mut self, k: usize, value: f64, rhs: &mut [f64]) {
        let lo = k.saturating_sub(self.b);
        let hi = (k + self.b + 1).min(self.n);
        for r in lo..hi {
            if r == k {
                continue;
            }
            let c = self.get(r, k);
            if c != 0.0 {
                rhs[r] -= c * value;
                self.set(r, k, 0.0);
            }
        }
        self.set(k, k, 1.0);
        rhs[k] = value;
    }

    pub fn factor(&mut self) {
        let n = self.n;
        let b = self.b;
        for i in 0..n {
            let lo = i.saturating_sub(b);
            for j in lo..i {
                let mut s = self.get(i, j);
                let klo = i.saturating_sub(b).max(j.saturating_sub(b));
                for k in klo..j {
                    s -= self.get(i, k) * self.get(k, k) * self.get(j, k);
                }
                let d = self.get(j, j);
                let v = if d.abs() > 1e-300 { s / d } else { 0.0 };
                self.set(i, j, v);
            }
            let mut d = self.get(i, i);
            for k in lo..i {
                let l = self.get(i, k);
                d -= l * l * self.get(k, k);
            }
            if d <= 1e-300 {
                d = 1e-300;
            }
            self.set(i, i, d);
        }
    }

    pub fn solve(&self, rhs: &mut [f64]) {
        let n = self.n;
        let b = self.b;
        for i in 0..n {
            let lo = i.saturating_sub(b);
            let mut s = rhs[i];
            for k in lo..i {
                s -= self.get(i, k) * rhs[k];
            }
            rhs[i] = s;
        }
        for i in 0..n {
            rhs[i] /= self.get(i, i);
        }
        for i in (0..n).rev() {
            let hi = (i + b + 1).min(n);
            let mut s = rhs[i];
            for k in (i + 1)..hi {
                s -= self.get(k, i) * rhs[k];
            }
            rhs[i] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banded_solve_matches_dense() {
        let n = 12;
        let b = 4;
        let mut m = Banded::new(n, b);
        for i in 0..n {
            m.add(i, i, 10.0 + i as f64);
            for j in i.saturating_sub(b)..i {
                m.add(i, j, 0.3 * ((i * 7 + j * 3) % 5) as f64 - 0.5);
            }
        }
        let x_true: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                rhs[i] += m.get(i, j) * x_true[j];
            }
        }
        m.factor();
        m.solve(&mut rhs);
        for i in 0..n {
            assert!((rhs[i] - x_true[i]).abs() < 1e-9, "{} {} {}", i, rhs[i], x_true[i]);
        }
    }
}
