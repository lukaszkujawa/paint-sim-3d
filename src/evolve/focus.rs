
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FocusShape {
    Off,
    Circle,
    Rect,
}

#[derive(Clone, Copy, PartialEq)]
pub struct Focus {
    pub shape: FocusShape,
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub soft: f32,
    pub outside: f32,
}

impl Focus {
    pub fn rect(&self) -> [f32; 4] {
        [self.a[0].min(self.b[0]), self.a[1].min(self.b[1]), self.a[0].max(self.b[0]), self.a[1].max(self.b[1])]
    }

    pub(super) fn weight(&self, x: f32, y: f32, aspect: f32) -> f32 {
        let (sd, scale) = match self.shape {
            FocusShape::Off => return 1.0,
            FocusShape::Circle => {
                let r = ((self.b[0] - self.a[0]).powi(2) + ((self.b[1] - self.a[1]) / aspect).powi(2)).sqrt().max(1e-4);
                ((((x - self.a[0]).powi(2) + ((y - self.a[1]) / aspect).powi(2)).sqrt() - r), r)
            }
            FocusShape::Rect => {
                let r = self.rect();
                let (hx, hy) = (0.5 * (r[2] - r[0]), 0.5 * (r[3] - r[1]) / aspect);
                let (dx, dy) = ((x - 0.5 * (r[0] + r[2])).abs() - hx, ((y - 0.5 * (r[1] + r[3])) / aspect).abs() - hy);
                let out = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
                (out + dx.max(dy).min(0.0), hx.min(hy).max(1e-4))
            }
        };
        let f = (self.soft * scale).max(1e-4);
        let t = ((sd + 0.5 * f) / f).clamp(0.0, 1.0);
        let s = t * t * (3.0 - 2.0 * t);
        1.0 - s * (1.0 - self.outside)
    }
}
